import os
import sys
import uuid
import json
import logging
from typing import List, Dict, Any, Optional
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from db.neo4j_db import get_neo4j_driver
from data_pipeline.loader import CACHE_DIR

# Import calculators, helper schemas, and global player impact profiling
from data_pipeline.state_calculators import (
    FEATURE_SCHEMA,
    precompute_player_historical_impact,
    compute_states
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        
    # Run the upfront global pre-computations first
    precompute_player_historical_impact()

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
    
    # Batch writing query for Neo4j built dynamically from FEATURE_SCHEMA mapping to eliminate repetitive hardcoding
    set_statements = [f"g.{name} = r.vector[{idx}]" for idx, name in enumerate(FEATURE_SCHEMA)]
    set_clause = ",\n      ".join(set_statements)
    
    query = f"""
    UNWIND $rows AS r
    MERGE (m:Match {{match_id: r.match_id}})
    MERGE (g:GameState {{state_id: r.state_id}})
    SET
      g.minute      = r.minute,
      g.score_diff  = r.score_diff,
      g.xg_diff     = r.xg_diff,
      g.outcome_15m = r.outcome_15m,
      g.vector      = r.vector,
      {set_clause}
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
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, s_dict['state_id']))
                    
                    # Construct payload dynamically from FEATURE_SCHEMA
                    point_payload = {
                        'state_id':    s_dict['state_id'],
                        'match_id':    s_dict['match_id'],
                        'minute':      s_dict['minute'],
                        'score_diff':  s_dict['score_diff'],
                        'xg_diff':     s_dict['xg_diff'],
                        'outcome_15m': s_dict['outcome_15m']
                    }
                    for f_idx, name in enumerate(FEATURE_SCHEMA):
                        point_payload[name] = s_dict['vector'][f_idx]
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector=s_dict['vector'],
                        payload=point_payload
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