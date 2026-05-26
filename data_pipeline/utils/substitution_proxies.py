from collections import defaultdict
from typing import List, Dict, Any, Optional
from data_pipeline.zones import get_zone_id
from data_pipeline.utils.helpers import get_minute_window_events

def get_voronoi_area_change(
    completed_sub: Optional[Dict[str, Any]], 
    events: List[Dict[str, Any]], 
    event_uuid_space: Dict[str, float]
) -> float:

    if not completed_sub or not event_uuid_space:
        return 0.0
    
    new_id = completed_sub['on_id']
    repl_id = completed_sub['off_id']
    
    new_areas = []
    repl_areas = []
    
    for e in events:
        eid = e.get('id')
        if eid in event_uuid_space:
            pid = e.get('player', {}).get('id')

            if pid == new_id:
                new_areas.append(event_uuid_space[eid])
            elif pid == repl_id:
                repl_areas.append(event_uuid_space[eid])
                
    avg_new = sum(new_areas) / len(new_areas) if new_areas else 0.0
    avg_repl = sum(repl_areas) / len(repl_areas) if repl_areas else 0.0
    return round(avg_new - avg_repl, 4)


def get_zone_overload_score(
    completed_sub: Optional[Dict[str, Any]], 
    by_minute: Dict[int, List[Dict[str, Any]]], 
    home_team: str, 
    away_team: str
) -> float:

    if not completed_sub:
        return 0.0
        
    sub_min = completed_sub['minute']
    before_evts = get_minute_window_events(by_minute, sub_min - 15, sub_min)
    after_evts = get_minute_window_events(by_minute, sub_min, sub_min + 15)
    
    def calculate_max_gap(evts):
        hc = defaultdict(int)
        ac = defaultdict(int)

        for evt in evts:
            if evt.get('type', {}).get('name') in ('Carry', 'Pressure'):
                loc = evt.get('location')
                team = evt.get('team', {}).get('name')

                if loc and len(loc) >= 2:
                    zid = get_zone_id(loc[0], loc[1])

                    if team == home_team:
                        hc[zid] += 1
                    elif team == away_team:
                        ac[zid] += 1

        ht = sum(hc.values()) or 1
        at = sum(ac.values()) or 1

        gaps = [abs((hc[zid] / ht) - (ac[zid] / at)) for zid in range(15)]
        return max(gaps) if gaps else 0.0

    max_before = calculate_max_gap(before_evts)
    max_after = calculate_max_gap(after_evts)
    return round(max_after - max_before, 4)


def get_centrality_delta(
    completed_sub: Optional[Dict[str, Any]], 
    by_minute: Dict[int, List[Dict[str, Any]]], 
    home_team: str
) -> float:

    if not completed_sub:
        return 0.0
        
    sub_min = completed_sub['minute']
    new_id = completed_sub['on_id']
    repl_id = completed_sub['off_id']

    # 1. Calculate replaced player's passing centrality before sub
    before_evts = get_minute_window_events(by_minute, sub_min - 15, sub_min)
    total_passes_before = 0
    repl_involved = 0

    for evt in before_evts:
        if (evt.get('type', {}).get('name') == 'Pass' 
            and evt.get('team', {}).get('name') == home_team 
            and evt.get('pass', {}).get('outcome') is None):

            total_passes_before += 1
            passer = evt.get('player', {}).get('id')
            rec = evt.get('pass', {}).get('recipient', {}).get('id')

            if passer == repl_id or rec == repl_id:
                repl_involved += 1
    centrality_before = repl_involved / max(total_passes_before, 1)

    # 2. Calculate incoming player's passing centrality after sub
    after_evts = get_minute_window_events(by_minute, sub_min, sub_min + 15)
    total_passes_after = 0
    new_involved = 0
    
    for evt in after_evts:
        if (evt.get('type', {}).get('name') == 'Pass' 
            and evt.get('team', {}).get('name') == home_team 
            and evt.get('pass', {}).get('outcome') is None):

            total_passes_after += 1
            passer = evt.get('player', {}).get('id')
            rec = evt.get('pass', {}).get('recipient', {}).get('id')

            if passer == new_id or rec == new_id:
                new_involved += 1

    centrality_after = new_involved / max(total_passes_after, 1)
    return round(centrality_after - centrality_before, 4)


def get_candidate_involvement(
    candidate_player_id: Optional[int], 
    by_minute: Dict[int, List[Dict[str, Any]]], 
    cutoff: int, 
    home_team: str
) -> float:

    if not candidate_player_id:
        return 0.0
    recent_evts = get_minute_window_events(by_minute, max(0, cutoff - 15), cutoff)
    cand_touches = sum(1 for e in recent_evts if e.get('player', {}).get('id') == candidate_player_id)
    team_touches = sum(1 for e in recent_evts if e.get('team', {}).get('name') == home_team and e.get('player', {}).get('id') is not None)
    return round(cand_touches / max(team_touches, 1), 4)


def get_candidate_hist_impact(
    future_replacement_id: Optional[int], 
    player_hist_impact_cache: Dict[int, float]
) -> float:

    if not future_replacement_id:
        return 0.0
        
    return player_hist_impact_cache.get(future_replacement_id, 0.0)
