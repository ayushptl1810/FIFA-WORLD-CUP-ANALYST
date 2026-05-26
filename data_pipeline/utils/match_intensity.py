from typing import List, Dict, Any

def get_press_intensity(window: List[Dict[str, Any]], home_team: str, away_team: str) -> float:
    home_pressures = sum(1 for e in window
                         if e.get('type', {}).get('name') == 'Pressure'
                         and e.get('team', {}).get('name') == home_team)

    home_passes_allowed = sum(1 for e in window
                              if e.get('type', {}).get('name') == 'Pass'
                              and e.get('team', {}).get('name') == away_team
                              and e.get('pass', {}).get('outcome') is None)

    return round(home_pressures / max(home_passes_allowed, 1), 4)


def get_momentum_15m(by_minute: Dict[int, List[Dict[str, Any]]], cutoff: int, home_team: str, away_team: str) -> float:
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

    return round(max(-1.0, min(1.0, recent_home_xg - recent_away_xg)), 4)


def get_box_entries(window: List[Dict[str, Any]], home_team: str) -> float:
    home_box_entries = 0
    for e in window:
        if (e.get('type', {}).get('name') == 'Pass'
            and e.get('team', {}).get('name') == home_team
            and e.get('pass', {}).get('outcome') is None):

            end_loc = e.get('pass', {}).get('end_location')
            if end_loc and len(end_loc) >= 2:
                # StatsBomb penalty area boundaries: x >= 102, 18 <= y <= 62
                if end_loc[0] >= 102.0 and 18.0 <= end_loc[1] <= 62.0:
                    home_box_entries += 1
                    
    return float(home_box_entries)
