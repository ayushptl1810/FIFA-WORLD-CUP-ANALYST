import os
import sys
import json
import math
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_pipeline.loader import CACHE_DIR

# Import modular features sub-calculators from pipeline utilities
from data_pipeline.utils.match_context import (
    get_score_diff,
    get_subs_used,
    get_xg_diff,
    get_possession_pct,
    get_formation_vec,
    get_red_card_factor,
    get_yellow_card_count,
    get_outcome_15m
)

from data_pipeline.utils.team_shape import (
    get_team_compactness,
    get_def_line_height,
    get_attack_width,
    get_max_player_drift,
    get_candidate_drift
)

from data_pipeline.utils.match_intensity import (
    get_press_intensity,
    get_momentum_15m,
    get_box_entries
)

from data_pipeline.utils.substitution_proxies import (
    get_voronoi_area_change,
    get_zone_overload_score,
    get_centrality_delta,
    get_candidate_involvement,
    get_candidate_hist_impact
)

logger = logging.getLogger(__name__)
MINUTES = list(range(5, 91, 5))   # [5, 10, 15, ..., 90]

# Centralized declarative Feature Schema mapping vector index to names.
FEATURE_SCHEMA = [
    "score_diff", "minute_norm", "subs_used", "xg_diff", "possession_pct", "press_intensity", "team_compactness", "def_line_height",
    "attack_width", "momentum_15m", "formation_vec", "max_player_drift", "red_card_factor", "yellow_card_count", "box_entries",
    "voronoi_area_change", "zone_overload_score", "centrality_delta", "candidate_involvement", "candidate_drift", "candidate_hist_impact"
]

# Global cache dictionary for Feature 21 (pre-computed once upfront)
PLAYER_HIST_IMPACT: Dict[int, float] = {}


def precompute_player_historical_impact():
    """
    Loops through all matches in the cache, parses every substitution event,
    computes the home vs away xG delta in the subsequent 15 minutes,
    and stores the average impact per player_id.
    """

    global PLAYER_HIST_IMPACT
    logger.info("Pre-computing historical player substitution impacts...")
    
    events_dir = os.path.join(CACHE_DIR, "events")
    if not os.path.exists(events_dir):
        logger.warning("Events cache directory not found. Cannot pre-compute historical impacts.")
        return
        
    files = [f for f in os.listdir(events_dir) if f.endswith(".json")]
    player_deltas = defaultdict(list)
    
    for filename in sorted(files):
        filepath = os.path.join(events_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                events = json.load(f)
                
            by_minute = defaultdict(list)
            for e in events:
                m = e.get('minute', 0)
                by_minute[int(m)].append(e)
                
            for e in events:
                if e.get('type', {}).get('name') == 'Substitution':
                    team_name = e.get('team', {}).get('name')
                    sub_data = e.get('substitution', {})
                    subbed_on_id = sub_data.get('replacement', {}).get('id')
                    sub_min = int(e.get('minute', 0))
                    
                    if not subbed_on_id:
                        continue
                        
                    future = [evt for m, evts in by_minute.items()
                              if sub_min < m <= sub_min + 15
                              for evt in evts]
                    
                    sub_team_xg = sum(evt.get('shot', {}).get('statsbomb_xg', 0) or 0
                                      for evt in future
                                      if evt.get('type', {}).get('name') == 'Shot'
                                      and evt.get('team', {}).get('name') == team_name)

                    opp_team_xg = sum(evt.get('shot', {}).get('statsbomb_xg', 0) or 0
                                      for evt in future
                                      if evt.get('type', {}).get('name') == 'Shot'
                                      and evt.get('team', {}).get('name') != team_name)
                    
                    xg_delta = sub_team_xg - opp_team_xg
                    player_deltas[subbed_on_id].append(xg_delta)
                    
        except Exception:
            continue
            
    for pid, deltas in player_deltas.items():
        PLAYER_HIST_IMPACT[pid] = round(sum(deltas) / len(deltas), 4)
        
    logger.info(f"Pre-computation complete! Profiled {len(PLAYER_HIST_IMPACT)} players.")


def compute_states(match_id: int, match_meta: dict, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each 5-minute interval, compute the full 21-feature RL state vector.
    Returns a list of state dicts ready for Neo4j + Qdrant.
    """
    # -- Pre-index events by minute for fast slicing --
    by_minute = defaultdict(list)
    for e in events:
        m = e.get('minute', 0)
        by_minute[int(m)].append(e)

    home_team = match_meta.get('home_team')
    away_team = match_meta.get('away_team')

    # -- Extract chronological substitutions that actually happened for the home team --
    subs = []
    for e in events:
        if (e.get('type', {}).get('name') == 'Substitution' 
            and e.get('team', {}).get('name') == home_team):

            sub_min = int(e.get('minute', 0))
            off_id = e.get('player', {}).get('id')
            on_id = e.get('substitution', {}).get('replacement', {}).get('id')

            if off_id and on_id:
                subs.append({
                    'minute': sub_min,
                    'off_id': off_id,
                    'on_id': on_id
                })

    subs.sort(key=lambda x: x['minute'])

    # -- Pre-index 360 freeze frames for spatial Voronoi calculations --
    event_uuid_space = {}
    frames_path = os.path.join(CACHE_DIR, "three-sixty", f"{match_id}.json")

    if os.path.exists(frames_path):
        try:
            with open(frames_path, "r", encoding="utf-8") as f:
                frames = json.load(f)

            if frames:
                for fr in frames:
                    uuid_str = fr.get('event_uuid')
                    coords = fr.get('visible_player_coords', [])
                    actor = next((p for p in coords if p.get('actor')), None)

                    if actor and uuid_str:
                        ax, ay = actor.get('x'), actor.get('y')
                        opps = [p for p in coords if not p.get('teammate')]

                        if ax is not None and ay is not None and opps:
                            min_dist = min(
                                math.sqrt((ax - o.get('x', 0))**2 + (ay - o.get('y', 0))**2)
                                for o in opps if o.get('x') is not None
                            )

                            # Space proxy proportional to circle area around actor
                            event_uuid_space[uuid_str] = round(math.pi * (min_dist ** 2), 2)
        except Exception:
            pass

    snapshots = []

    for cutoff in MINUTES:
        # Sliced play-by-play events up to this minute
        window = [e for m, evts in by_minute.items()
                  if m <= cutoff for e in evts]

        # Chronological matching to identify candidate/new/replaced players relative to current cutoff
        candidate_player_id = None
        future_replacement_id = None
        for s in subs:
            if s['minute'] > cutoff:
                candidate_player_id = s['off_id']
                future_replacement_id = s['on_id']
                break

        completed_sub = None
        for s in reversed(subs):
            if cutoff >= s['minute']:
                completed_sub = s
                break

        # Calculate values dynamically using focused sub-calculators from pipeline utilities
        features = {
            "score_diff":            get_score_diff(window, home_team, away_team),
            "minute_norm":           round(cutoff / 90.0, 4),
            "subs_used":             get_subs_used(window, home_team),
            "xg_diff":               get_xg_diff(window, home_team, away_team),
            "possession_pct":        get_possession_pct(window, home_team),
            "press_intensity":       get_press_intensity(window, home_team, away_team),
            "team_compactness":      get_team_compactness(window, home_team),
            "def_line_height":       get_def_line_height(window, home_team),
            "attack_width":          get_attack_width(window, home_team),
            "momentum_15m":          get_momentum_15m(by_minute, cutoff, home_team, away_team),
            "formation_vec":         get_formation_vec(window, home_team),
            "max_player_drift":      get_max_player_drift(window, home_team),
            "red_card_factor":       get_red_card_factor(window, home_team, away_team),
            "yellow_card_count":     get_yellow_card_count(window, home_team),
            "box_entries":           get_box_entries(window, home_team),
            "voronoi_area_change":   get_voronoi_area_change(completed_sub, events, event_uuid_space),
            "zone_overload_score":   get_zone_overload_score(completed_sub, by_minute, home_team, away_team),
            "centrality_delta":      get_centrality_delta(completed_sub, by_minute, home_team),
            "candidate_involvement": get_candidate_involvement(candidate_player_id, by_minute, cutoff, home_team),
            "candidate_drift":       get_candidate_drift(window, candidate_player_id),
            "candidate_hist_impact": get_candidate_hist_impact(future_replacement_id, PLAYER_HIST_IMPACT)
        }

        # Dynamically map floats according to FEATURE_SCHEMA index ordering
        vector = [features[name] for name in FEATURE_SCHEMA]

        snapshots.append({
            'state_id':       f"{match_id}_m{cutoff}",
            'match_id':       match_id,
            'minute':         cutoff,
            'score_diff':     features['score_diff'],
            'xg_diff':        features['xg_diff'],
            'outcome_15m':    get_outcome_15m(by_minute, cutoff, home_team, away_team),
            'vector':         vector,
        })

    return snapshots
