from typing import List, Dict, Any

def get_score_diff(window: List[Dict[str, Any]], home_team: str, away_team: str) -> float:
    home_goals = sum(1 for e in window
                     if e.get('type', {}).get('name') == 'Shot'
                     and e.get('shot', {}).get('outcome', {}).get('name') == 'Goal'
                     and e.get('team', {}).get('name') == home_team)

    away_goals = sum(1 for e in window
                     if e.get('type', {}).get('name') == 'Shot'
                     and e.get('shot', {}).get('outcome', {}).get('name') == 'Goal'
                     and e.get('team', {}).get('name') == away_team)
    return float(home_goals - away_goals)


def get_subs_used(window: List[Dict[str, Any]], home_team: str) -> float:
    subs_used = sum(1 for e in window
                    if e.get('type', {}).get('name') == 'Substitution'
                    and e.get('team', {}).get('name') == home_team)

    return float(subs_used)


def get_xg_diff(window: List[Dict[str, Any]], home_team: str, away_team: str) -> float:
    home_xg = sum(e.get('shot', {}).get('statsbomb_xg', 0) or 0
                  for e in window
                  if e.get('type', {}).get('name') == 'Shot'
                  and e.get('team', {}).get('name') == home_team)

    away_xg = sum(e.get('shot', {}).get('statsbomb_xg', 0) or 0
                  for e in window
                  if e.get('type', {}).get('name') == 'Shot'
                  and e.get('team', {}).get('name') == away_team)

    return round(home_xg - away_xg, 4)


def get_possession_pct(window: List[Dict[str, Any]], home_team: str) -> float:
    home_poss = sum(1 for e in window if e.get('possession_team', {}).get('name') == home_team)
    total_poss = len(window) if window else 1
    return round(home_poss / total_poss, 4)


def get_formation_vec(window: List[Dict[str, Any]], home_team: str) -> float:
    FORMATIONS = [442, 433, 4231, 451, 352, 343, 532, 541]
    home_formation_int = 433  # default

    for e in window:
        if e.get('type', {}).get('name') == 'Starting XI' and e.get('team', {}).get('name') == home_team:
            home_formation_int = e.get('tactics', {}).get('formation', 433)
            break
    
    formation_vec_idx = FORMATIONS.index(home_formation_int) if home_formation_int in FORMATIONS else 1
    return formation_vec_idx / 7.0


def get_red_card_factor(window: List[Dict[str, Any]], home_team: str, away_team: str) -> float:
    home_reds = sum(1 for e in window
                    if e.get('team', {}).get('name') == home_team
                    and (e.get('foul_committed', {}).get('card', {}).get('name') in ('Second Yellow', 'Red Card')
                         or e.get('bad_behaviour', {}).get('card', {}).get('name') in ('Second Yellow', 'Red Card')))
    
    away_reds = sum(1 for e in window
                    if e.get('team', {}).get('name') == away_team
                    and (e.get('foul_committed', {}).get('card', {}).get('name') in ('Second Yellow', 'Red Card')
                         or e.get('bad_behaviour', {}).get('card', {}).get('name') in ('Second Yellow', 'Red Card')))
    
    return float(home_reds - away_reds)


def get_yellow_card_count(window: List[Dict[str, Any]], home_team: str) -> float:
    home_yellows = sum(1 for e in window
                       if e.get('team', {}).get('name') == home_team
                       and (e.get('foul_committed', {}).get('card', {}).get('name') == 'Yellow Card'
                            or e.get('bad_behaviour', {}).get('card', {}).get('name') == 'Yellow Card'))
    
    return float(home_yellows)


def get_outcome_15m(by_minute: Dict[int, List[Dict[str, Any]]], cutoff: int, home_team: str, away_team: str) -> float:
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
                   
    return round(fut_home - fut_away, 4)
