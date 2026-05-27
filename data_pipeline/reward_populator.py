import os
import sys
import uuid
import json
import logging
from typing import List, Dict, Any, Optional

# Bootstrap project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from db.neo4j_db import get_neo4j_driver
from data_pipeline.loader import CACHE_DIR
from data_pipeline.state_calculators import FEATURE_SCHEMA

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def map_position_to_role(pos_name: str) -> int:
    """
    Maps a StatsBomb position name to our action space roles:
    1: Attacker (Forward, Winger, Striker)
    2: Midfielder
    3: Defender (Center Back, Fullback, Wingback, Goalkeeper)
    0: Wait (default)
    """
    pos_lower = pos_name.lower()
    if any(k in pos_lower for k in ["forward", "wing", "striker", "centre forward", "secondary striker", "left wing", "right wing"]):
        return 1
    if any(k in pos_lower for k in ["midfielder", "midfield", "centre midfielder", "defensive midfielder", "attacking midfielder"]):
        return 2
    if any(k in pos_lower for k in ["back", "def", "fullback", "wingback", "center back", "left back", "right back", "goalkeeper"]):
        return 3
    return 2  # Fallback to midfielder


def compute_match_rewards(match_id: int, home_team: str, events: List[Dict[str, Any]], states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes possession-adjusted EDG continuous rewards and action mappings for all 18 states in a match.
    """
    # 1. Chronological substitutions of the home team
    subs = []
    for e in events:
        if (e.get('type', {}).get('name') == 'Substitution' 
            and e.get('team', {}).get('name') == home_team):
            
            sub_min = int(e.get('minute', 0))
            off_id = e.get('player', {}).get('id')
            on_id = e.get('substitution', {}).get('replacement', {}).get('id')
            pos_name = e.get('position', {}).get('name', 'Unknown')
            
            if off_id and on_id:
                subs.append({
                    'minute': sub_min,
                    'off_id': off_id,
                    'on_id': on_id,
                    'pos_name': pos_name,
                    'role': map_position_to_role(pos_name)
                })
    subs.sort(key=lambda x: x['minute'])

    # 2. Layer 1 & 2: Calculate possession-adjusted outcomes & trends per sub
    sub_attributions = {}
    sub_trends = {}
    sub_outcomes = {}
    
    for s in subs:
        sub_min = s['minute']
        on_id = s['on_id']
        
        # Find first touch at or after sub_min
        window_start = sub_min
        for e in events:
            e_min = int(e.get('minute', 0))
            if e_min >= sub_min and e.get('player', {}).get('id') == on_id:
                window_start = e_min
                break
        
        # Post-sub 15m window from window_start
        future = [evt for evt in events if window_start < int(evt.get('minute', 0)) <= window_start + 15]
        
        sub_team_xg = sum(evt.get('shot', {}).get('statsbomb_xg', 0) or 0
                          for evt in future
                          if evt.get('type', {}).get('name') == 'Shot'
                          and evt.get('team', {}).get('name') == home_team)

        opp_team_xg = sum(evt.get('shot', {}).get('statsbomb_xg', 0) or 0
                          for evt in future
                          if evt.get('type', {}).get('name') == 'Shot'
                          and evt.get('team', {}).get('name') != home_team)
        
        outcome = sub_team_xg - opp_team_xg
        sub_outcomes[on_id] = round(outcome, 4)

        # Pre-sub 15m window
        past = [evt for evt in events if sub_min - 15 <= int(evt.get('minute', 0)) < sub_min]
        
        pre_sub_team_xg = sum(evt.get('shot', {}).get('statsbomb_xg', 0) or 0
                              for evt in past
                              if evt.get('type', {}).get('name') == 'Shot'
                              and evt.get('team', {}).get('name') == home_team)

        pre_sub_opp_team_xg = sum(evt.get('shot', {}).get('statsbomb_xg', 0) or 0
                                  for evt in past
                                  if evt.get('type', {}).get('name') == 'Shot'
                                  and evt.get('team', {}).get('name') != home_team)
        
        trend = pre_sub_team_xg - pre_sub_opp_team_xg
        sub_trends[on_id] = round(trend, 4)
        
        # Attribution
        sub_attributions[on_id] = round(outcome - trend, 4)

    # 3. Compute rewards for each 5-minute state cutoff
    updated_states = []
    
    for s_dict in states:
        cutoff = s_dict['minute']
        vec = s_dict['vector']
        
        # Determine empirical action taken in the next 5 minutes [cutoff, cutoff + 5]
        action_taken = 0
        matched_sub = None
        for s in subs:
            if cutoff <= s['minute'] < cutoff + 5:
                action_taken = s['role']
                matched_sub = s
                break
        
        reward = 0.0
        
        # Layer 3 Reward Calculations
        if action_taken > 0 and matched_sub:
            # Action = Substitution
            on_id = matched_sub['on_id']
            attribution = sub_attributions.get(on_id, 0.0)
            
            # Proxy scores from the vector indices:
            v_change = vec[15] # voronoi_area_change
            z_overload = vec[16] # zone_overload_score
            c_delta = vec[17] # centrality_delta
            proxy_score = (v_change + z_overload + c_delta) / 3.0
            
            # Continuous EDG Reward Formula:
            reward = attribution + 0.5 * proxy_score
            
            # Off-ball proxy bonus
            if attribution > 0.1 and proxy_score > 0.2:
                reward += 0.4
            elif attribution < -0.1:
                reward -= 0.2
                
            # Early Sub Penalty with Crisis Exemptions
            if cutoff < 45:
                # Exemption checks:
                momentum = vec[9] # momentum_15m
                red_cards = vec[12] # red_card_factor
                involvement = vec[18] # candidate_involvement
                drift = vec[19] # candidate_drift
                
                crisis_triggered = (
                    drift > 0.6 or 
                    involvement < 0.05 or 
                    momentum < -0.3 or 
                    red_cards < 0.0
                )
                
                if not crisis_triggered:
                    reward -= 0.5  # early sub penalty
        else:
            # Action = Wait (0)
            reward = 0.0
            if cutoff > 60:
                # 1. Inaction fatigue penalty
                drift = vec[19] # candidate_drift
                if drift > 0.7:
                    reward -= 0.2
                
                # 2. Inaction tactical penalty
                momentum = vec[9] # momentum_15m
                subs_used = vec[2] # subs_used
                if momentum < -0.3 and subs_used < 3.0:
                    reward -= 0.3
            
            # 3. Critical Fatigue Inaction Penalty (Antigravity Extension)
            fatigue_state = vec[23] # fatigue_state
            if fatigue_state >= 0.8:
                reward -= 0.6
            elif 0.5 <= fatigue_state < 0.8:
                momentum = vec[9] # momentum_15m
                if momentum < -0.2:
                    reward -= 0.3
        
        # Round reward for storage neatness
        reward = round(reward, 4)
        
        updated_states.append({
            'state_id': s_dict['state_id'],
            'action_taken': action_taken,
            'reward': reward,
            'vector': vec
        })
        
    return updated_states


def populate_rewards():
    """
    Main orchestrator to load existing states, compute continuous rewards,
    and update Qdrant and Neo4j databases.
    """
    logger.info("Initializing connection to databases...")
    driver = get_neo4j_driver()
    
    qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    qdrant_grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    q_client = QdrantClient(url=qdrant_url, check_compatibility=False, prefer_grpc=True, grpc_port=qdrant_grpc_port)
    
    # 1. Query all matches with states from Neo4j
    matches_query = """
    MATCH (m:Match)-[:HAS_STATE]->(g:GameState)
    RETURN DISTINCT m.match_id as match_id
    """
    match_ids = []
    try:
        with driver.session() as s:
            result = s.run(matches_query)
            match_ids = [record["match_id"] for record in result]
    except Exception as e:
        logger.error(f"Failed to query match IDs from Neo4j: {e}")
        driver.close()
        return
        
    logger.info(f"Found {len(match_ids)} matches to process in Neo4j.")
    
    events_dir = os.path.join(CACHE_DIR, "events")
    total_updated = 0
    
    for idx, match_id in enumerate(sorted(match_ids), 1):
        # Retrieve all states for this match from Neo4j
        states_query = """
        MATCH (m:Match {match_id: $match_id})-[rel:HAS_STATE]->(g:GameState)
        RETURN g.state_id as state_id, g.minute as minute, g.vector as vector, g.score_diff as score_diff, g.xg_diff as xg_diff, g.outcome_15m as outcome_15m
        """
        
        states = []
        try:
            with driver.session() as s:
                result = s.run(states_query, match_id=match_id)
                for record in result:
                    states.append({
                        'state_id': record['state_id'],
                        'minute': record['minute'],
                        'vector': record['vector'],
                        'score_diff': record['score_diff'],
                        'xg_diff': record['xg_diff'],
                        'outcome_15m': record['outcome_15m']
                    })
        except Exception as e:
            logger.error(f"[{idx}/{len(match_ids)}] Match ID {match_id}: Failed to load states from Neo4j: {e}")
            continue
            
        if not states:
            continue
            
        # Get team name to determine substitutions (match events file)
        events_path = os.path.join(events_dir, f"{match_id}.json")
        if not os.path.exists(events_path):
            logger.warning(f"[{idx}/{len(match_ids)}] Match ID {match_id}: Events file not found. Skipping.")
            continue
            
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                events = json.load(f)
            
            # Find the home team name
            # Starting XI contains the home team details
            home_team = None
            for e in events:
                if e.get('type', {}).get('name') == 'Starting XI':
                    home_team = e.get('team', {}).get('name')
                    break
                    
            if not home_team:
                logger.warning(f"[{idx}/{len(match_ids)}] Match ID {match_id}: Home team could not be determined. Skipping.")
                continue
                
            # Compute actions and rewards
            updated_states = compute_match_rewards(match_id, home_team, events, states)
            
            # Update Neo4j
            neo_update_query = """
            UNWIND $rows AS r
            MATCH (g:GameState {state_id: r.state_id})
            SET g.action_taken = r.action_taken,
                g.reward = r.reward
            """
            with driver.session() as s:
                s.run(neo_update_query, rows=updated_states)
                
            # Update Qdrant
            points = []
            for u in updated_states:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, u['state_id']))
                
                # Fetch original payload to avoid wiping metadata
                res = q_client.retrieve(
                    collection_name="historical_states",
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=False
                )
                
                payload = {}
                if res and res[0].payload:
                    payload = res[0].payload
                
                # Add action and reward
                payload['action_taken'] = u['action_taken']
                payload['reward'] = u['reward']
                
                points.append(PointStruct(
                    id=point_id,
                    vector=u['vector'],
                    payload=payload
                ))
                
            q_client.upsert(
                collection_name="historical_states",
                points=points
            )
            
            total_updated += len(updated_states)
            logger.info(f"[{idx}/{len(match_ids)}] Match ID {match_id}: Computed rewards for {len(updated_states)} states.")
            
        except Exception as e:
            logger.error(f"[{idx}/{len(match_ids)}] Match ID {match_id}: Failed to process rewards: {e}")
            
    driver.close()
    logger.info("=" * 60)
    logger.info(f"Populator Complete! Successfully updated action_taken and reward for {total_updated} states in Qdrant & Neo4j.")
    logger.info("=" * 60)


if __name__ == "__main__":
    populate_rewards()
