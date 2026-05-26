import os
import sys
import json
import logging
from collections import defaultdict
from typing import List, Dict, Any
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from neo4j import GraphDatabase
from db.neo4j_db import get_neo4j_driver
from data_pipeline.loader import CACHE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_pass_network(match_id: int, events: List[Dict[str, Any]], driver) -> int:
    edges = defaultdict(lambda: {
        'count': 0, 'success': 0,
        'start_x': [], 'start_y': [],
        'end_x':   [], 'end_y': []
    })

    for e in events:
        # Verify event type is 'Pass'
        if e.get('type', {}).get('name') != 'Pass':
            continue

        passer_id = e.get('player', {}).get('id')
        passer_name = e.get('player', {}).get('name')
        team = e.get('team', {}).get('name')
        
        # Recipient details are nested inside 'pass'
        pass_data = e.get('pass', {})
        recipient_id = pass_data.get('recipient', {}).get('id')
        recipient_name = pass_data.get('recipient', {}).get('name')
        
        location = e.get('location')  # [x, y]
        end_loc = pass_data.get('end_location')  # [x, y]
        outcome = pass_data.get('outcome', {}).get('name')  # None = successful in StatsBomb

        # Basic validation: ensure essential IDs and coordinates are present
        if not passer_id or not recipient_id or not location or not end_loc:
            continue

        key = (passer_id, passer_name, recipient_id, recipient_name, team, match_id)
        edges[key]['count'] += 1
        edges[key]['success'] += 1 if outcome is None else 0
        edges[key]['start_x'].append(location[0])
        edges[key]['start_y'].append(location[1])
        edges[key]['end_x'].append(end_loc[0])
        edges[key]['end_y'].append(end_loc[1])

    if not edges:
        return 0

    # Build batch of relationship dictionaries for UNWIND parameter
    rels = []
    for (pid, pname, rid, rname, team_name, mid), data in edges.items():
        n = data['count']
        rels.append({
            'match_id':       mid,
            'passer_id':      pid,
            'passer_name':    pname,
            'recipient_id':   rid,
            'recipient_name':  rname,
            'team':           team_name,
            'count':          n,
            'success_rate':   round(data['success'] / n, 3),
            'avg_start_x':    round(sum(data['start_x']) / n, 2),
            'avg_start_y':    round(sum(data['start_y']) / n, 2),
            'avg_end_x':      round(sum(data['end_x']) / n, 2),
            'avg_end_y':      round(sum(data['end_y']) / n, 2),
        })

    # Write to Neo4j in one transaction (idempotent ON CREATE/MATCH and SET queries)
    query = """
    UNWIND $rels AS r
    MERGE (p:Player {player_id: r.passer_id})
      ON CREATE SET p.name = r.passer_name, p.team_id = r.team
      ON MATCH SET p.team_id = r.team
    MERGE (q:Player {player_id: r.recipient_id})
      ON CREATE SET q.name = r.recipient_name, q.team_id = r.team
      ON MATCH SET q.team_id = r.team
    MERGE (p)-[rel:PASSED_TO {match_id: r.match_id}]->(q)
    SET
      rel.count        = r.count,
      rel.success_rate = r.success_rate,
      rel.avg_start_x  = r.avg_start_x,
      rel.avg_start_y  = r.avg_start_y,
      rel.avg_end_x    = r.avg_end_x,
      rel.avg_end_y    = r.avg_end_y
    """
    with driver.session() as s:
        s.run(query, rels=rels)

    return len(rels)


def is_match_processed(match_id: int, driver) -> bool:
    query = """
    MATCH ()-[r:PASSED_TO {match_id: $match_id}]->()
    RETURN count(r) > 0 AS processed
    LIMIT 1
    """
    with driver.session() as s:
        result = s.run(query, match_id=match_id)
        record = result.single()
        return record["processed"] if record else False


def populate_all_pass_networks():    
    events_dir = os.path.join(CACHE_DIR, "events")
    if not os.path.exists(events_dir):
        logger.error(f"No events directory found in cache: {events_dir}")
        return
        
    files = [f for f in os.listdir(events_dir) if f.endswith(".json")]
    if not files:
        logger.warning(f"No cached events files (.json) found in {events_dir}")
        return
        
    logger.info(f"Found {len(files)} cached matches. Beginning passing network construction in Neo4j...")
    
    driver = get_neo4j_driver()
    total_rels_written = 0
    
    for idx, filename in enumerate(sorted(files), 1):
        match_id_str = filename.split(".json")[0]
        try:
            match_id = int(match_id_str)
        except ValueError:
            continue
            
        try:
            if is_match_processed(match_id, driver):
                logger.info(f"[{idx}/{len(files)}] Match ID {match_id}: Already built in Neo4j. Skipping.")
                continue
        except Exception as e:
            logger.warning(f"Failed to check resume status for Match ID {match_id}: {e}. Processing anyway.")
            
        filepath = os.path.join(events_dir, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                events = json.load(f)
                
            rels_count = build_pass_network(match_id, events, driver)
            total_rels_written += rels_count
            logger.info(f"[{idx}/{len(files)}] Match ID {match_id}: Built passing network with {rels_count} connections.")
        except Exception as e:
            logger.error(f"Failed to build passing network for Match ID {match_id}: {e}")
            
    driver.close()
    logger.info("=" * 60)
    logger.info(f"Done! Successfully wrote a total of {total_rels_written} PASSED_TO relationships in Neo4j.")
    logger.info("=" * 60)


if __name__ == "__main__":
    populate_all_pass_networks()