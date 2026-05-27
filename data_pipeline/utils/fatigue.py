from typing import List, Dict, Any, Optional

def get_involvement_slope(
    candidate_player_id: Optional[int],
    involvement_history: Dict[int, List[float]],
    recent_involvement: float
) -> float:
    """
    Calculates the 15-minute rate of change (slope) of the candidate player's involvement rate.
    """
    if not candidate_player_id or candidate_player_id not in involvement_history:
        return 0.0
        
    history = involvement_history[candidate_player_id]
    # 15 minutes lookback is 3 steps back (5m, 10m, 15m)
    prev_idx = len(history) - 4
    prev_involvement = history[prev_idx] if prev_idx >= 0 else 0.0
    return round(recent_involvement - prev_involvement, 4)


def get_pressing_load(
    press_intensity: float,
    possession_pct: float,
    cutoff: int
) -> float:
    """
    Calculates the pressing load factor, modeling physical exhaustion from ineffective pressing:
    press_intensity * opponent_possession * normalized_minute
    """
    opponent_possession = 1.0 - possession_pct
    normalized_min = cutoff / 90.0
    return round(press_intensity * opponent_possession * normalized_min, 4)


def get_fatigue_state(
    candidate_player_id: Optional[int],
    cutoff: int,
    recent_involvement: float,
    involvement_history: Dict[int, List[float]],
    drift: float,
    pressing_load: float
) -> float:
    """
    Computes the Continuous Fatigue Accumulator F in [0.0, 1.0] as a weighted physical load index.
    F = 0.3 * TimeLoad + 0.3 * InvolvementDecay + 0.2 * DriftLoad + 0.2 * PressingLoad
    """
    if not candidate_player_id or candidate_player_id not in involvement_history:
        return 0.0
        
    history = involvement_history[candidate_player_id]
    match_avg_involvement = sum(history) / len(history) if history else 0.0
    
    # 1. Cardiovascular baseline fatigue
    T_load = cutoff / 90.0
    
    # 2. Drop in involvement
    I_decay = max(0.0, 1.0 - (recent_involvement / max(match_avg_involvement, 0.01)))
    
    # 3. Position drift breakdown
    D_load = min(1.0, drift / 0.8)
    
    # 4. Pressing exhaustion load
    P_load = min(1.0, pressing_load)
    
    # Weighted summation
    F = 0.3 * T_load + 0.3 * I_decay + 0.2 * D_load + 0.2 * P_load
    return round(min(1.0, max(0.0, F)), 4)
