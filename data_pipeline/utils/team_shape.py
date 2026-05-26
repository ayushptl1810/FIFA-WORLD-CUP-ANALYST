from typing import List, Dict, Any, Optional
from data_pipeline.utils.helpers import _std, get_position_locations
from collections import defaultdict

def get_team_compactness(window: List[Dict[str, Any]], home_team: str) -> float:
    defender_y = get_position_locations(window, home_team, ["Back"], 1)
    return round(_std(defender_y), 4) if len(defender_y) > 1 else 0.5


def get_def_line_height(window: List[Dict[str, Any]], home_team: str) -> float:
    defender_y = get_position_locations(window, home_team, ["Back"], 1)
    return round(sum(defender_y) / len(defender_y) / 80.0, 4) if defender_y else 0.5


def get_attack_width(window: List[Dict[str, Any]], home_team: str) -> float:
    forward_x = get_position_locations(window, home_team, ["Forward", "Wing"], 0)
    return round(_std(forward_x) / 120.0, 4) if len(forward_x) > 1 else 0.5


def get_max_player_drift(window: List[Dict[str, Any]], home_team: str) -> float:
    player_locs = defaultdict(list)
    for e in window:
        if e.get('type', {}).get('name') == 'Carry' and e.get('team', {}).get('name') == home_team:
            
            pid = e.get('player', {}).get('id')
            loc = e.get('location', [])

            if pid and loc:
                player_locs[pid].append(loc[0])

    drifts = [_std(xs) for xs in player_locs.values() if len(xs) > 2]
    return round(max(drifts) / 120.0, 4) if drifts else 0.0


def get_candidate_drift(window: List[Dict[str, Any]], candidate_player_id: Optional[int]) -> float:
    if not candidate_player_id:
        return 0.0
        
    cand_carries = [
        e.get('location', [])[0] for e in window 
        if e.get('type', {}).get('name') == 'Carry' 
        and e.get('player', {}).get('id') == candidate_player_id 
        and e.get('location')
    ]
    return round(_std(cand_carries) / 120.0, 4) if len(cand_carries) > 1 else 0.0
