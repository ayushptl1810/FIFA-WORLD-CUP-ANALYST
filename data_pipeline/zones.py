import os
import sys
import json
import logging
from collections import defaultdict
from typing import List, Dict, Any

# Add the project root directory to sys.path to enable absolute imports when executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from neo4j import GraphDatabase
from db.neo4j_db import get_neo4j_driver
from pipeline.loader import CACHE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pitch is 120 x 80 in StatsBomb coordinates
# We divide into a 5-column x 3-row grid = 15 zones
COLS = 5
ROWS = 3
COL_SIZE = 120.0 / COLS   # 24 units wide
ROW_SIZE = 80.0  / ROWS   # 26.67 units tall


def get_zone_id(x: float, y: float) -> int:
    col = min(int(x / COL_SIZE), COLS - 1)
    row = min(int(y / ROW_SIZE), ROWS - 1)
    return row * COLS + col   # 0–14


def zone_label(zone_id: int) -> str:
    row = zone_id // COLS
    col = zone_id  % COLS
    row_names = ['Defensive', 'Middle', 'Attacking']
    col_names = ['Left', 'Centre-Left', 'Centre', 'Centre-Right', 'Right']
    return f"{row_names[row]} {col_names[col]}"


def build_zone_control(match_id: int, events: List[Dict[str, Any]], driver) -> int:
    """
    Count how many Carry + Pressure events each player has per zone.
    """
    control = defaultdict(int)
    player_meta = {}  # player_id → {name, team}

    for e in events:
        # Verify event type is either Carry or Pressure
        event_type = e.get('type', {}).get('name')
        if event_type not in ('Carry', 'Pressure'):
            continue
            
        loc = e.get('location')  # [x, y]
        pid = e.get('player', {}).get('id')
        pname = e.get('player', {}).get('name')
        team_name = e.get('team', {}).get('name')

        if not loc or not pid:
            continue

        zone_id = get_zone_id(loc[0], loc[1])
        control[(pid, zone_id)] += 1
        player_meta[pid] = {
            'name': pname,
            'team': team_name
        }

    if not control:
        return 0

    # Compute density score per player: events_in_zone / total_events_for_player
    totals = defaultdict(int)
    for (pid, _), cnt in control.items():
        totals[pid] += cnt

    rows = []
    for (pid, zid), cnt in control.items():
        rows.append({
            'match_id':      match_id,
            'player_id':     pid,
            'player_name':   player_meta[pid]['name'],
            'team':          player_meta[pid]['team'],
            'zone_id':       f"{match_id}_z{zid}",
            'zone_num':      zid,
            'zone_label':    zone_label(zid),
            'x_min':  (zid % COLS)     * COL_SIZE,
            'x_max':  (zid % COLS + 1) * COL_SIZE,
            'y_min':  (zid // COLS)    * ROW_SIZE,
            'y_max':  (zid // COLS + 1)* ROW_SIZE,
            'density': round(cnt / totals[pid], 4),
        })

    # Idempotent write query utilizing MERGE and SET properties
    query = """
    UNWIND $rows AS r
    MERGE (z:Zone {zone_id: r.zone_id})
      ON CREATE SET
        z.zone_num   = r.zone_num,
        z.zone_label = r.zone_label,
        z.x_min      = r.x_min,
        z.x_max      = r.x_max,
        z.y_min      = r.y_min,
        z.y_max      = r.y_max
    MERGE (p:Player {player_id: r.player_id})
      ON CREATE SET p.name = r.player_name, p.team_id = r.team
      ON MATCH SET p.team_id = r.team
    MERGE (p)-[rel:CONTROLS_ZONE {match_id: r.match_id}]->(z)
    SET rel.density_score = r.density
    """
    with driver.session() as s:
        s.run(query, rows=rows)

    return len(rows)


def is_match_processed(match_id: int, driver) -> bool:
    query = """
    MATCH ()-[r:CONTROLS_ZONE {match_id: $match_id}]->()
    RETURN count(r) > 0 AS processed
    LIMIT 1
    """
    with driver.session() as s:
        result = s.run(query, match_id=match_id)
        record = result.single()
        return record["processed"] if record else False


def populate_all_zone_controls():
    events_dir = os.path.join(CACHE_DIR, "events")
    if not os.path.exists(events_dir):
        logger.error(f"No events directory found in cache: {events_dir}")
        return
        
    files = [f for f in os.listdir(events_dir) if f.endswith(".json")]
    if not files:
        logger.warning(f"No cached events files (.json) found in {events_dir}")
        return
        
    logger.info(f"Found {len(files)} cached matches. Beginning zone control construction in Neo4j...")
    
    driver = get_neo4j_driver()
    total_rels_written = 0
    
    for idx, filename in enumerate(sorted(files), 1):
        match_id_str = filename.split(".json")[0]
        try:
            match_id = int(match_id_str)
        except ValueError:
            continue
            
        # Resume logic: skip processing if the match's zone control is already in Neo4j
        try:
            if is_match_processed(match_id, driver):
                logger.info(f"[{idx}/{len(files)}] Match ID {match_id}: Already processed in Neo4j. Skipping.")
                continue
        except Exception as e:
            logger.warning(f"Failed to check resume status for Match ID {match_id}: {e}. Processing anyway.")
            
        filepath = os.path.join(events_dir, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                events = json.load(f)
                
            rels_count = build_zone_control(match_id, events, driver)
            total_rels_written += rels_count
            logger.info(f"[{idx}/{len(files)}] Match ID {match_id}: Built zone control with {rels_count} connections.")
        except Exception as e:
            logger.error(f"Failed to build zone control for Match ID {match_id}: {e}")
            
    driver.close()
    logger.info("=" * 60)
    logger.info(f"Done! Successfully wrote a total of {total_rels_written} CONTROLS_ZONE relationships in Neo4j.")
    logger.info("=" * 60)


if __name__ == "__main__":
    populate_all_zone_controls()