import os
import sys
import uuid
import json
import math
import logging
from collections import defaultdict
from typing import List, Dict, Any

# Add the project root directory to sys.path to enable absolute imports when executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from db.neo4j_db import get_neo4j_driver
from pipeline.loader import CACHE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MINUTES = list(range(5, 91, 5))   # [5, 10, 15, ..., 90]


def compute_states(match_id: int, match_meta: dict, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each 5-minute interval, compute the full 12-feature RL state vector.
    Returns a list of state dicts ready for Neo4j + Qdrant.
    """
    # -- pre-index events by minute for fast slicing --
    by_minute = defaultdict(list)
    for e in events:
        m = e.get('minute', 0)
        by_minute[int(m)].append(e)

    home_team = match_meta.get('home_team')
    away_team = match_meta.get('away_team')

    snapshots = []

    for cutoff in MINUTES:
        # All events up to this minute
        window = [e for m, evts in by_minute.items()
                  if m <= cutoff for e in evts]

        # -- 1. score_diff --
        home_goals = sum(1 for e in window
                         if e.get('type', {}).get('name') == 'Shot'
                         and e.get('shot', {}).get('outcome', {}).get('name') == 'Goal'
                         and e.get('team', {}).get('name') == home_team)
        away_goals = sum(1 for e in window
                         if e.get('type', {}).get('name') == 'Shot'
                         and e.get('shot', {}).get('outcome', {}).get('name') == 'Goal'
                         and e.get('team', {}).get('name') == away_team)
        score_diff = home_goals - away_goals

        # -- 2. minute_norm --
        minute_norm = round(cutoff / 90.0, 4)

        # -- 3. subs_used --
        subs_used = sum(1 for e in window
                        if e.get('type', {}).get('name') == 'Substitution'
                        and e.get('team', {}).get('name') == home_team)

        # -- 4. xg_diff --
        home_xg = sum(e.get('shot', {}).get('statsbomb_xg', 0) or 0
                      for e in window
                      if e.get('type', {}).get('name') == 'Shot'
                      and e.get('team', {}).get('name') == home_team)
        away_xg = sum(e.get('shot', {}).get('statsbomb_xg', 0) or 0
                      for e in window
                      if e.get('type', {}).get('name') == 'Shot'
                      and e.get('team', {}).get('name') == away_team)
        xg_diff = round(home_xg - away_xg, 4)

        # -- 5. possession_pct --
        home_poss = sum(1 for e in window if e.get('possession_team', {}).get('name') == home_team)
        total_poss = len(window) if window else 1
        possession_pct = round(home_poss / total_poss, 4)

        # -- 6. press_intensity (PPDA proxy) --
        home_pressures = sum(1 for e in window
                             if e.get('type', {}).get('name') == 'Pressure'
                             and e.get('team', {}).get('name') == home_team)
        home_passes_allowed = sum(1 for e in window
                                  if e.get('type', {}).get('name') == 'Pass'
                                  and e.get('team', {}).get('name') == away_team
                                  and e.get('pass', {}).get('outcome') is None)
        press_intensity = round(
            home_pressures / max(home_passes_allowed, 1), 4
        )

        # -- 7 & 8. team_compactness + def_line_height --
        defender_y = []
        forward_x  = []

        # Approximate from carry event locations of home team defenders
        for e in window:
            if e.get('type', {}).get('name') != 'Carry' or e.get('team', {}).get('name') != home_team:
                continue
            pos = e.get('position', {}).get('name', '')
            loc = e.get('location', [])
            if not loc:
                continue
            if 'Back' in str(pos):
                defender_y.append(loc[1])
            if 'Forward' in str(pos) or 'Wing' in str(pos):
                forward_x.append(loc[0])

        team_compactness = round(_std(defender_y), 4) if len(defender_y) > 1 else 0.5
        def_line_height  = round(
            sum(defender_y) / len(defender_y) / 80.0, 4
        ) if defender_y else 0.5

        # -- 9. attack_width --
        attack_width = round(_std(forward_x) / 120.0, 4) if len(forward_x) > 1 else 0.5

        # -- 10. momentum_15m (rolling xG delta last 15 min) --
        recent = [e for m, evts in by_minute.items()
                  if max(0, cutoff - 15) < m <= cutoff
                  for e in evts]
        recent_home_xg = sum(e.get('shot', {}).get('statsbomb_xg', 0) or 0
                             for e in recent
                             if e.get('type', {}).get('name') == 'Shot'
                             and e.get('team', {}).get('name') == home_team)
        recent_away_xg = sum(e.get('shot', {}).get('statsbomb_xg', 0) or 0
                             for e in recent
                             if e.get('type', {}).get('name') == 'Shot'
                             and e.get('team', {}).get('name') == away_team)
        momentum_15m = round(
            max(-1.0, min(1.0, recent_home_xg - recent_away_xg)), 4
        )

        # -- 11. formation_vec (one-hot, 8 formations) --
        FORMATIONS = [442, 433, 4231, 451, 352, 343, 532, 541]
        home_formation_int = 433  # default
        for e in window:
            if e.get('type', {}).get('name') == 'Starting XI' and e.get('team', {}).get('name') == home_team:
                home_formation_int = e.get('tactics', {}).get('formation', 433)
                break
        
        formation_vec_idx = FORMATIONS.index(home_formation_int) if home_formation_int in FORMATIONS else 1

        # -- 12. max_player_drift --
        player_locs = defaultdict(list)
        for e in window:
            if e.get('type', {}).get('name') == 'Carry' and e.get('team', {}).get('name') == home_team:
                pid = e.get('player', {}).get('id')
                loc = e.get('location', [])
                if pid and loc:
                    player_locs[pid].append(loc[0])

        drifts = [_std(xs) for xs in player_locs.values() if len(xs) > 2]
        max_player_drift = round(max(drifts) / 120.0, 4) if drifts else 0.0

        # -- 13. red_card_factor --
        home_reds = sum(1 for e in window
                        if e.get('team', {}).get('name') == home_team
                        and (e.get('foul_committed', {}).get('card', {}).get('name') in ('Second Yellow', 'Red Card')
                             or e.get('bad_behaviour', {}).get('card', {}).get('name') in ('Second Yellow', 'Red Card')))
        away_reds = sum(1 for e in window
                        if e.get('team', {}).get('name') == away_team
                        and (e.get('foul_committed', {}).get('card', {}).get('name') in ('Second Yellow', 'Red Card')
                             or e.get('bad_behaviour', {}).get('card', {}).get('name') in ('Second Yellow', 'Red Card')))
        red_card_factor = home_reds - away_reds

        # -- 14. yellow_card_count (home team active yellows) --
        home_yellows = sum(1 for e in window
                           if e.get('team', {}).get('name') == home_team
                           and (e.get('foul_committed', {}).get('card', {}).get('name') == 'Yellow Card'
                                or e.get('bad_behaviour', {}).get('card', {}).get('name') == 'Yellow Card'))

        # -- 15. box_entries (deep completions into opponent's box) --
        home_box_entries = 0
        for e in window:
            if (e.get('type', {}).get('name') == 'Pass'
                and e.get('team', {}).get('name') == home_team
                and e.get('pass', {}).get('outcome') is None):
                end_loc = e.get('pass', {}).get('end_location')
                if end_loc and len(end_loc) >= 2:
                    if end_loc[0] >= 102.0 and 18.0 <= end_loc[1] <= 62.0:
                        home_box_entries += 1

        # -- outcome label (did home xG improve in the next 15 min?) --
        future = [e for m, evts in by_minute.items()
                  if cutoff < m <= cutoff + 15
                  for e in evts]
        fut_home = sum(e.get('shot', {}).get('statsbomb_xg', 0) or 0
                       for e in future
                       if e.get('type', {}).get('name') == 'Shot'
                       and e.get('team', {}).get('name') == home_team)
        fut_away = sum(e.get('shot', {}).get('statsbomb_xg', 0) or 0
                       for e in future
                       if e.get('type', {}).get('name') == 'Shot'
                       and e.get('team', {}).get('name') == away_team)
        outcome_15m = round(fut_home - fut_away, 4)

        vector = [
            float(score_diff),
            minute_norm,
            float(subs_used),
            xg_diff,
            possession_pct,
            press_intensity,
            team_compactness,
            def_line_height,
            attack_width,
            momentum_15m,
            formation_vec_idx / 7.0,   # normalise to [0,1]
            max_player_drift,
            float(red_card_factor),
            float(home_yellows),
            float(home_box_entries),
        ]

        snapshots.append({
            'state_id':       f"{match_id}_m{cutoff}",
            'match_id':       match_id,
            'minute':         cutoff,
            'score_diff':     score_diff,
            'xg_diff':        xg_diff,
            'outcome_15m':    outcome_15m,
            'vector':         vector,
        })

    return snapshots


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def is_match_processed(match_id: int, driver, q_client) -> bool:
    """
    Checks if a match's game states are already fully stored in BOTH Neo4j and Qdrant.
    """
    # 1. Check Neo4j
    neo_query = """
    MATCH (m:Match {match_id: $match_id})-[r:HAS_STATE]->()
    RETURN count(r) > 0 AS processed
    LIMIT 1
    """
    neo4j_processed = False
    try:
        with driver.session() as s:
            result = s.run(neo_query, match_id=match_id)
            record = result.single()
            neo4j_processed = record["processed"] if record else False
    except Exception as e:
        logger.warning(f"Neo4j resume check failed for Match ID {match_id}: {e}")

    # 2. Check Qdrant
    qdrant_processed = False
    try:
        res = q_client.scroll(
            collection_name="historical_states",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="match_id",
                        match=MatchValue(value=match_id)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False
        )
        qdrant_processed = len(res[0]) > 0
    except Exception as e:
        logger.warning(f"Qdrant resume check failed for Match ID {match_id}: {e}")

    return neo4j_processed and qdrant_processed


def load_all_match_meta() -> Dict[int, Dict[str, Any]]:
    """
    Scans the matches cache directory dynamically and builds a lookup map of match_id -> match_meta.
    """
    match_meta_lookup = {}
    matches_dir = os.path.join(CACHE_DIR, "matches")
    if not os.path.exists(matches_dir):
        return match_meta_lookup
        
    for comp_id in os.listdir(matches_dir):
        comp_path = os.path.join(matches_dir, comp_id)
        if not os.path.isdir(comp_path):
            continue
        for season_file in os.listdir(comp_path):
            if not season_file.endswith(".json"):
                continue
            filepath = os.path.join(comp_path, season_file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    matches = json.load(f)
                for m in matches:
                    mid = m.get("match_id")
                    if mid:
                        match_meta_lookup[mid] = {
                            'home_team': m.get('home_team', {}).get('home_team_name'),
                            'away_team': m.get('away_team', {}).get('away_team_name')
                        }
            except Exception as e:
                logger.warning(f"Failed to load match meta from {filepath}: {e}")
    return match_meta_lookup


def populate_all_game_states():
    """
    Scans the local events cache, loads match events, computes 5-minute game states,
    and inserts GameState nodes & HAS_STATE relationships into Neo4j and upserts vectors to Qdrant.
    """
    events_dir = os.path.join(CACHE_DIR, "events")
    if not os.path.exists(events_dir):
        logger.error(f"No events directory found in cache: {events_dir}")
        return
        
    files = [f for f in os.listdir(events_dir) if f.endswith(".json")]
    if not files:
        logger.warning(f"No cached events files (.json) found in {events_dir}")
        return
        
    logger.info("Building match metadata lookup from local cache...")
    match_meta_lookup = load_all_match_meta()
    
    logger.info(f"Found {len(files)} cached matches. Beginning game states construction in Neo4j and Qdrant...")
    
    # Initialize Neo4j Driver
    driver = get_neo4j_driver()
    
    # Initialize Qdrant Client
    qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    qdrant_grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    q_client = QdrantClient(url=qdrant_url, check_compatibility=False, prefer_grpc=True, grpc_port=qdrant_grpc_port)
    
    total_states_written = 0
    
    # Batch writing query for Neo4j
    query = """
    UNWIND $rows AS r
    MERGE (m:Match {match_id: r.match_id})
    MERGE (g:GameState {state_id: r.state_id})
    SET
      g.minute           = r.minute,
      g.score_diff       = r.score_diff,
      g.xg_diff          = r.xg_diff,
      g.outcome_15m      = r.outcome_15m,
      g.vector           = r.vector,
      g.minute_norm      = r.vector[1],
      g.subs_used        = r.vector[2],
      g.possession_pct   = r.vector[4],
      g.press_intensity  = r.vector[5],
      g.team_compactness = r.vector[6],
      g.def_line_height  = r.vector[7],
      g.attack_width     = r.vector[8],
      g.momentum_15m     = r.vector[9],
      g.formation_vec    = r.vector[10],
      g.max_player_drift = r.vector[11],
      g.red_card_factor  = r.vector[12],
      g.yellow_card_count= r.vector[13],
      g.box_entries      = r.vector[14]
    MERGE (m)-[rel:HAS_STATE]->(g)
    """
    
    for idx, filename in enumerate(sorted(files), 1):
        match_id_str = filename.split(".json")[0]
        try:
            match_id = int(match_id_str)
        except ValueError:
            continue
            
        # Resume logic: skip processing if the match's game states are already in both Neo4j and Qdrant
        try:
            if is_match_processed(match_id, driver, q_client):
                logger.info(f"[{idx}/{len(files)}] Match ID {match_id}: Already processed in both DBs. Skipping.")
                continue
        except Exception as e:
            logger.warning(f"Failed to check resume status for Match ID {match_id}: {e}. Processing anyway.")
            
        match_meta = match_meta_lookup.get(match_id)
        if not match_meta:
            logger.warning(f"[{idx}/{len(files)}] Match ID {match_id}: Metadata not found in cache. Skipping.")
            continue
            
        filepath = os.path.join(events_dir, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                events = json.load(f)
                
            states = compute_states(match_id, match_meta, events)
            
            if states:
                # 1. Write to Neo4j
                with driver.session() as s:
                    s.run(query, rows=states)
                
                # 2. Write to Qdrant
                points = []
                for s_dict in states:
                    # Generate deterministic UUID based on state_id
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, s_dict['state_id']))
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector=s_dict['vector'],
                        payload={
                            'state_id':    s_dict['state_id'],
                            'match_id':    s_dict['match_id'],
                            'minute':      s_dict['minute'],
                            'score_diff':  s_dict['score_diff'],
                            'xg_diff':     s_dict['xg_diff'],
                            'outcome_15m': s_dict['outcome_15m']
                        }
                    ))
                
                q_client.upsert(
                    collection_name="historical_states",
                    points=points
                )
                
                total_states_written += len(states)
                logger.info(f"[{idx}/{len(files)}] Match ID {match_id}: Built networks and {len(states)} snapshots (Neo4j & Qdrant).")
            else:
                logger.warning(f"[{idx}/{len(files)}] Match ID {match_id}: No states generated.")
        except Exception as e:
            logger.error(f"Failed to build game states for Match ID {match_id}: {e}")
            
    driver.close()
    logger.info("=" * 60)
    logger.info(f"Done! Successfully wrote a total of {total_states_written} GameState snapshots to Neo4j & Qdrant.")
    logger.info("=" * 60)


if __name__ == "__main__":
    populate_all_game_states()