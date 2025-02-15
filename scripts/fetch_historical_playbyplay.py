import os
import time
import pandas as pd
import random
import requests
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2
from requests.exceptions import ReadTimeout, ConnectionError

# Global settings
MAX_RETRIES = 5
COOLDOWN_THRESHOLD = 3  # Number of consecutive failures before pausing
COOLDOWN_TIME = 300  # Pause duration (5 minutes)

def get_all_games(season="2024-25", season_type="Regular Season"):
    """
    Fetches all games for the 2024-25 NBA season.
    :return: DataFrame with only games that do NOT have stored play-by-play data.
    """
    print(f"🔍 Checking NBA games for {season} season...")

    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season, season_type_nullable=season_type)
    games = game_finder.get_data_frames()[0]

    # Keep only relevant columns
    games = games[['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_DATE']]
    games = games.drop_duplicates(subset=['GAME_ID'])

    # Filter out games that are already stored
    existing_files = set(os.listdir("data/"))
    missing_games = games[~games['GAME_ID'].apply(lambda game_id: f"playbyplay_{game_id}.csv" in existing_files)]

    print(f"✅ Found {len(missing_games)} new games to fetch.")

    return missing_games


def get_game_playbyplay(game_id):
    """
    Fetches play-by-play data for a given NBA game with retries & cooldown.
    """
    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            print(f"🔄 Attempt {attempts + 1}/{MAX_RETRIES} - Fetching play-by-play for Game ID: {game_id}")

            # Fetch play-by-play data
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            pbp_data = pbp.get_data_frames()[0]

            if pbp_data.empty:
                print(f"⚠️ Warning: No play-by-play data found for Game ID {game_id}.")
                return None

            # Keep relevant columns
            pbp_data = pbp_data[['GAME_ID', 'EVENTNUM', 'PCTIMESTRING', 'PERIOD', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION', 'PLAYER1_NAME', 'PLAYER2_NAME', 'PLAYER3_NAME']]

            # Rename columns for clarity
            pbp_data.rename(columns={
                'PCTIMESTRING': 'time_remaining',
                'PERIOD': 'quarter',
                'HOMEDESCRIPTION': 'home_event',
                'VISITORDESCRIPTION': 'away_event',
                'PLAYER1_NAME': 'player_1',
                'PLAYER2_NAME': 'player_2',
                'PLAYER3_NAME': 'player_3'
            }, inplace=True)

            return pbp_data

        except (ReadTimeout, ConnectionError):
            print(f"❌ Timeout error for Game ID {game_id}. Retrying ({attempts + 1}/{MAX_RETRIES})...")
            attempts += 1
            time.sleep(random.uniform(5, 15))  # Wait before retrying
        except requests.exceptions.RequestException as e:
            print(f"❌ Critical error for Game ID {game_id}: {e}")
            return None

    print(f"❌ Failed to fetch play-by-play for Game ID {game_id} after {MAX_RETRIES} attempts.")
    return None


if __name__ == "__main__":
    print("📡 Fetching new games from the 2024-25 season...")

    # Get only games that haven't been collected yet
    new_games_df = get_all_games()

    if new_games_df.empty:
        print("🎉 All available games have been collected. No new games to fetch.")
    else:
        failed_games = []
        consecutive_failures = 0

        for index, row in new_games_df.iterrows():
            game_id = row['GAME_ID']
            filename = f"data/playbyplay_{game_id}.csv"

            pbp_df = get_game_playbyplay(game_id)

            if pbp_df is not None:
                pbp_df.to_csv(filename, index=False)
                print(f"✅ Play-by-play data saved: {filename}")
                consecutive_failures = 0
            else:
                failed_games.append(game_id)
                consecutive_failures += 1

            # **Pause execution if too many consecutive failures occur**
            if consecutive_failures >= COOLDOWN_THRESHOLD:
                print(f"⏸️ Too many failed requests ({consecutive_failures}). Pausing for {COOLDOWN_TIME // 60} minutes...")
                time.sleep(COOLDOWN_TIME)
                consecutive_failures = 0

            time.sleep(random.uniform(3, 7))  # Reduce API rate limiting

        # Save failed game IDs for manual retry
        if failed_games:
            failed_games_df = pd.DataFrame(failed_games, columns=['GAME_ID'])
            failed_games_df.to_csv("data/failed_games_2024_25.csv", index=False)
            print(f"❌ Some games failed. Saved failed game IDs to data/failed_games_2024_25.csv.")
