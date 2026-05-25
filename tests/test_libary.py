"""
Interactive Demonstrations for `soccerdata` and `datafc` Libraries
==================================================================

This file provides clean, runnable examples showcasing exactly how to use:
1. `datafc` - A modern, fast, pandas-oriented Sofascore API wrapper (perfect for World Cup analysis!).
2. `soccerdata` - A unified scraping interface for ClubElo, Understat, FBref, and more.
"""

import os
import pandas as pd
import soccerdata as sd
import datafc
from statsbombpy import sb

# ==============================================================================
# SECTION 1: DATAFC DEMONSTRATIONS (Sofascore)
# ==============================================================================

def demo_datafc_search_and_seasons():
    """
    Demonstrates using datafc to search for tournaments and find season IDs.
    """
    print("\n" + "="*50)
    print("DEMO: datafc - Search and Season Discovery")
    print("="*50)

    # 1. Search for 'World Cup' to find its tournament ID
    print("[1] Searching for 'World Cup' tournament...")
    search_df = datafc.search_data('World Cup', entity_type='tournament')
    
    print("\nTop tournament search results:")
    print(search_df[['entity_id', 'entity_name', 'entity_type']].head(5))

    # FIFA World Cup tournament ID is 16
    world_cup_id = 16
    
    # 2. Get all historical seasons of the FIFA World Cup
    print(f"\n[2] Fetching historical seasons for tournament ID {world_cup_id}...")
    seasons_df = datafc.seasons_data(tournament_id=world_cup_id)
    
    print("\nRecent FIFA World Cup seasons:")
    print(seasons_df[['season_id', 'season_name', 'season_year']].head(5))


def demo_datafc_world_cup_match_and_stats():
    """
    Demonstrates fetching matches, match-level statistics, and shots coordinates
    for the 2022 World Cup using datafc.
    """
    print("\n" + "="*50)
    print("DEMO: datafc - FIFA World Cup Matches & Statistics")
    print("="*50)

    # 2022 World Cup has tournament_id 16 and season_id 41087
    world_cup_id = 16
    season_2022_id = 41087

    # 1. Fetch match data for the Final stage of 2022 World Cup
    print("[1] Fetching 2022 World Cup Final match details...")
    match_df = datafc.match_data(
        tournament_id=world_cup_id, 
        season_id=season_2022_id, 
        tournament_type='world_cup',
        tournament_stage='final'
    )

    print("\nFinal Match Details:")
    print(match_df[['game_id', 'home_team', 'away_team', 'home_score_display', 'away_score_display']])

    # 2. Fetch team match-level statistics (e.g. Possession, xG, Passes)
    print("\n[2] Fetching team match-level statistics...")
    stats_df = datafc.match_stats_data(match_df)
    
    print("\nSample Team Match Stats (Possession & Expected Goals):")
    # Filter to show some key stats for the full match
    key_stats = stats_df[
        (stats_df['period'] == 'ALL') & 
        (stats_df['stat_name'].isin(['Ball possession', 'Expected goals', 'Total shots']))
    ]
    print(key_stats[['stat_name', 'home_team_stat', 'away_team_stat']])

    # 3. Fetch coordinates-level shots data (with shot type, coordinates, xG, xGOT)
    print("\n[3] Fetching shot map details (coordinates & xG)...")
    shots_df = datafc.shots_data(match_df)
    
    print("\nSample Shot Map entries (showing xG, coordinates, shooter):")
    print(shots_df[['time', 'player_name', 'incident_type', 'xg', 'player_coordinates_x', 'player_coordinates_y']].head(5))


def demo_datafc_lineups_and_player_ratings():
    """
    Demonstrates fetching match lineups and 66+ player metrics (e.g., passes, tackles, ratings).
    """
    print("\n" + "="*50)
    print("DEMO: datafc - Player Lineups, Ratings & Individual Stats")
    print("="*50)

    world_cup_id = 16
    season_2022_id = 41087

    # Get the final match DataFrame
    match_df = datafc.match_data(
        tournament_id=world_cup_id, 
        season_id=season_2022_id, 
        tournament_type='world_cup',
        tournament_stage='final'
    )

    # 1. Fetch detailed player lineups and match actions
    print("[1] Fetching lineups and 60+ individual player metrics...")
    lineups_df = datafc.lineups_data(match_df)

    print("\nExample: Lionel Messi's key statistics in the Final:")
    messi_stats = lineups_df[
        (lineups_df['player_name'] == 'Lionel Messi') & 
        (lineups_df['stat_name'].isin(['rating', 'goals', 'totalPass', 'accuratePass', 'keyPass', 'wasFouled']))
    ]
    print(messi_stats[['stat_name', 'stat_value']])


# ==============================================================================
# SECTION 2: SOCCERDATA DEMONSTRATIONS (Multi-Source Scraper)
# ==============================================================================

def demo_soccerdata_clubelo():
    """
    Demonstrates using soccerdata.ClubElo to pull historic Elo ratings for clubs.
    """
    print("\n" + "="*50)
    print("DEMO: soccerdata - Club Elo History")
    print("="*50)

    print("[1] Fetching Club Elo history for Real Madrid...")
    try:
        elo = sd.ClubElo()
        df = elo.read_team_history('Real Madrid')
        
        print("\nReal Madrid Elo Ratings (recent entries):")
        print(df[['rank', 'elo']].tail(5))
    except Exception as e:
        print(f"Skipping ClubElo demo due to scraping exception: {e}")


def demo_soccerdata_understat():
    """
    Demonstrates using soccerdata.Understat to fetch match schedules and shots coordinates.
    """
    print("\n" + "="*50)
    print("DEMO: soccerdata - Understat xG and Shot Maps")
    print("="*50)

    print("[1] Fetching Understat schedule for Premier League 2023-24...")
    try:
        us = sd.Understat(leagues='ENG-Premier League', seasons='2023-24')
        schedule_df = us.read_schedule()
        
        print("\nUnderstat Premier League Schedule (First 3 matches):")
        print(schedule_df[['league_id', 'url']].head(3))
        
        # Match ID 22275 is Burnley vs Manchester City
        print("\n[2] Fetching Understat shot events for match ID 22275...")
        shots_df = us.read_shot_events(match_id=22275)
        
        # Reset the multi-index to turn 'player' index level into a normal column
        shots_flat = shots_df.reset_index()
        
        print("\nUnderstat shot events sample:")
        print(shots_flat[['minute', 'player', 'xg', 'result']].head(5))
    except Exception as e:
        print(f"Skipping Understat demo due to scraping exception: {e}")


# ==============================================================================
# SECTION 3: STATSBOMB DEMONSTRATIONS (Open Event Data)
# ==============================================================================

def demo_statsbomb_worldcup():
    """
    Demonstrates using statsbombpy to fetch World Cup 2022 matches and deep event streams.
    """
    print("\n" + "="*50)
    print("DEMO: StatsBomb - World Cup 2022 Matches & Shot Events")
    print("="*50)

    # 1. Fetch matches for World Cup 2022 (competition_id=43, season_id=106)
    print("[1] Fetching StatsBomb match list for World Cup 2022...")
    matches_df = sb.matches(competition_id=43, season_id=106)
    
    print("\nSample World Cup 2022 Matches:")
    print(matches_df[['match_id', 'home_team', 'away_team', 'home_score', 'away_score']].head(5))

    # 2. Fetch deep event streams (e.g. Shots) for Argentina vs Australia (match_id=3869151)
    match_id = 3869151
    print(f"\n[2] Fetching complete match event stream for match ID {match_id} (Argentina vs Australia)...")
    events_df = sb.events(match_id=match_id)
    
    # Filter for Shot events
    shots_df = events_df[events_df['type'] == 'Shot']
    
    print("\nShot events in Argentina vs Australia (showing xG and outcomes):")
    print(shots_df[['minute', 'player', 'team', 'shot_outcome', 'shot_statsbomb_xg']].head(5))

# ==============================================================================
# MAIN RUNNER
# ==============================================================================

if __name__ == '__main__':
    print("Starting soccerdata, datafc, and statsbomb demonstrations...")
    
    try:
        demo_datafc_search_and_seasons()
        demo_datafc_world_cup_match_and_stats()
        demo_datafc_lineups_and_player_ratings()
        demo_soccerdata_clubelo()
        demo_soccerdata_understat()
        demo_statsbomb_worldcup()
        print("\nAll demonstrations completed successfully!")
    except Exception as ex:
        print(f"\nAn error occurred during execution: {ex}")
