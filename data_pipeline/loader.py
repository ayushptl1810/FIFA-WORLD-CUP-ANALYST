import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base URL for StatsBomb open-data GitHub repository
STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"

# Determine the project cache directory dynamically (workspace/data/statsbomb_cache)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "statsbomb_cache"))

TOURNAMENTS: List[Tuple[int, int, str]] = [
    # FIFA Men's World Cups
    (43, 106, "FIFA World Cup 2022"),
    (43, 3, "FIFA World Cup 2018"),
    
    # UEFA Men's Euros
    (55, 43, "UEFA Euro 2020"),
    (55, 282, "UEFA Euro 2024"),
    
    # Copa America
    (223, 282, "Copa America 2024"),
    
    # African Cup of Nations
    (1267, 107, "African Cup of Nations 2023"),
    
    # Champions League (UCL Finals/Knockouts in StatsBomb Open Data)
    (16, 4, "Champions League 2018/2019"),
    (16, 1, "Champions League 2017/2018"),
    (16, 2, "Champions League 2016/2017"),
    (16, 27, "Champions League 2015/2016"),
    (16, 26, "Champions League 2014/2015"),
    (16, 25, "Champions League 2013/2014"),
    (16, 24, "Champions League 2012/2013"),
    (16, 23, "Champions League 2011/2012"),
    (16, 22, "Champions League 2010/2011"),
    (16, 21, "Champions League 2009/2010"),
    (16, 41, "Champions League 2008/2009"),
    (16, 39, "Champions League 2006/2007"),
    (16, 37, "Champions League 2004/2005"),
    (16, 44, "Champions League 2003/2004"),
    (16, 76, "Champions League 1999/2000"),
    
    # High-Stakes Women's Tournaments (to hit the ~500 high-quality match goal)
    (72, 107, "Women's World Cup 2023"),
    (72, 30, "Women's World Cup 2019"),
    (53, 106, "UEFA Women's Euro 2022")
]

def _fetch_and_cache(url_path: str, cache_rel_path: str, allow_missing: bool = False) -> Optional[List[Any]]:
    """
    Checks the local cache for the specified file. If it exists, loads and returns it.
    Otherwise, fetches from StatsBomb raw open-data, caches it locally, and returns the data.
    """
    cache_file = os.path.join(CACHE_DIR, cache_rel_path)

    # 1. Check local cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}. Re-fetching from network...")

    # 2. Fetch from GitHub raw data
    url = STATSBOMB_BASE_URL + url_path
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 404:
            if allow_missing:
                # Cache the 404 absence as null so we don't query the network again
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(None, f)
                return None
            response.raise_for_status()
            
        response.raise_for_status()
        data = response.json()
        
        # 3. Write to local cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return data

    except requests.RequestException as e:
        logger.error(f"Network error occurred while fetching {url}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from {url}: {e}")
        raise


def get_matches(competition_id: int, season_id: int) -> List[Dict[str, Any]]:
    """
    Fetches the list of all matches for a specific competition and season.
    """
    path = f"matches/{competition_id}/{season_id}.json"
    result = _fetch_and_cache(path, path, allow_missing=False)
    return result if result is not None else []


def get_events(match_id: int) -> List[Dict[str, Any]]:
    """
    Fetches every event (passes, shots, substitutions, etc.) for a single match.
    """
    path = f"events/{match_id}.json"
    result = _fetch_and_cache(path, path, allow_missing=False)
    return result if result is not None else []


def get_lineups(match_id: int) -> List[Dict[str, Any]]:
    """
    Fetches the starting XI and squad rosters for both teams in a match.
    """
    path = f"lineups/{match_id}.json"
    result = _fetch_and_cache(path, path, allow_missing=False)
    return result if result is not None else []


def get_frames(match_id: int) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches the 360 freeze frame spatial positions for a match.
    """
    path = f"three-sixty/{match_id}.json"
    return _fetch_and_cache(path, path, allow_missing=True)


def is_match_cached(match_id: int) -> bool:
    """
    Checks if a match is fully cached locally on disk without loading/parsing the JSONs.
    A match is considered cached if its lineups, events, and three-sixty files
    exist on disk (even if three-sixty is a cached 404 'null').
    """
    lineups_path = os.path.join(CACHE_DIR, "lineups", f"{match_id}.json")
    events_path = os.path.join(CACHE_DIR, "events", f"{match_id}.json")
    three_sixty_path = os.path.join(CACHE_DIR, "three-sixty", f"{match_id}.json")
    return os.path.exists(lineups_path) and os.path.exists(events_path) and os.path.exists(three_sixty_path)

def populate_cache():
    """
    Sequentially pre-caches all matches, lineups, events, and 360 frames
    for the selected high-stakes tournament corpus to build the RL training set.
    """
    logger.info("Starting StatsBomb local cache population for the RL training corpus...")
    logger.info("This will pre-cache matches, lineups, events, and 360 frames.")
    
    total_matches_cached = 0
    total_tournaments = len(TOURNAMENTS)
    
    for idx, (comp_id, season_id, name) in enumerate(TOURNAMENTS, 1):
        logger.info(f"[{idx}/{total_tournaments}] Fetching match list for {name}...")
        try:
            matches = get_matches(comp_id, season_id)
        except Exception as e:
            logger.error(f"Failed to fetch matches for {name}: {e}. Skipping tournament.")
            continue
            
        match_count = len(matches)
        logger.info(f"Found {match_count} matches in {name}.")
        
        for m_idx, match in enumerate(matches, 1):
            match_id = match.get("match_id")
            if not match_id:
                continue
                
            if is_match_cached(match_id):
                total_matches_cached += 1
                continue
                
            home_team = match.get("home_team", {}).get("home_team_name", "Home")
            away_team = match.get("away_team", {}).get("away_team_name", "Away")
            stage = match.get("competition_stage", {}).get("name", "Unknown Stage")
            
            logger.info(
                f"  -> Caching match {m_idx}/{match_count} (ID: {match_id}): "
                f"{home_team} vs {away_team} ({stage})"
            )
            
            try:
                # Cache lineages
                get_lineups(match_id)
                
                # Cache events
                get_events(match_id)
                
                # Cache 360 frames (if available)
                get_frames(match_id)
                
                total_matches_cached += 1
                
            except Exception as e:
                logger.warning(f"  [Error] Failed to cache match ID {match_id}: {e}. Continuing...")
                
            # Add a small delay between requests to be a good citizen
            time.sleep(0.05)
            
    logger.info("=" * 60)
    logger.info(f"Population complete! Successfully cached metadata and details for {total_matches_cached} matches.")
    logger.info("=" * 60)


if __name__ == "__main__":
    populate_cache()
