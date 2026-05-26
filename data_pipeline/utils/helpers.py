import math
from typing import List, Dict, Any

def _std(values: list) -> float:
    """
    Helper function to calculate the population standard deviation of a numeric list.
    Returns 0.0 for lists with fewer than 2 elements.
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def get_position_locations(window: List[Dict[str, Any]], team: str, position_keywords: List[str], coord_idx: int) -> List[float]:
    """
    Filters carry events for a team and specific positional keywords, 
    returning a list of coordinates at coord_idx (0 for x, 1 for y).
    """
    coords = []
    for e in window:
        if e.get('type', {}).get('name') == 'Carry' and e.get('team', {}).get('name') == team:
            pos = str(e.get('position', {}).get('name', ''))
            loc = e.get('location', [])
            if loc and any(kw in pos for kw in position_keywords):
                coords.append(loc[coord_idx])
    return coords


def get_minute_window_events(by_minute: Dict[int, List[Dict[str, Any]]], start_min: int, end_min: int) -> List[Dict[str, Any]]:
    """
    Extracts a flat list of events occurring between start_min (exclusive) and end_min (inclusive).
    """
    return [evt for m, evts in by_minute.items() if start_min < m <= end_min for evt in evts]
