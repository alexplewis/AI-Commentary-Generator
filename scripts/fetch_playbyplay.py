import json
import pandas as pd
import time
from nba_api.live.nba.endpoints import playbyplay
from nba_api.live.nba.endpoints import scoreboard

def get_live_games():
    """
    Fetches live NBA games currently being played.
    :return: A list of active game IDs.
    """
    games = scoreboard.ScoreBoard().get_dict()['scoreboard']['games']
    live_game_ids = [game['gameId'] for game in games if game['gameStatusText'] != 'Final']
    
    return live_game_ids


def get_game_playbyplay(game_id):
    """
    Fetches play-by-play data for a given NBA game.
    :param game_id: The unique Game ID for an NBA match.
    :return: DataFrame with play-by-play events.
    """
    pbp = playbyplay.PlayByPlay(game_id)

    # Print raw response for debugging
    print("Raw API Response:", pbp.nba_response._response)

    try:
        pbp_data = pbp.get_dict()
    except json.decoder.JSONDecodeError:
        print(f"❌ Error: JSONDecodeError for Game ID {game_id}. The response might be empty or malformed.")
        return pd.DataFrame()  # Return empty DataFrame to prevent script crash
    
    # Extract event list
    events = pbp_data.get("game", {}).get("actions", [])

    # Convert to DataFrame
    df = pd.DataFrame(events)

    if df.empty:
        print(f"⚠️ Warning: No play-by-play data found for Game ID {game_id}.")
        return df

    # Keep relevant columns
    df = df[['clock', 'period', 'description', 'teamTricode', 'playerNameI']]

    # Rename columns for clarity
    df.rename(columns={
        'clock': 'time_remaining',
        'period': 'quarter',
        'description': 'event_description',
        'teamTricode': 'team',
        'playerNameI': 'player'
    }, inplace=True)

    return df


if __name__ == "__main__":
    print("Fetching live NBA games...")

    # Get list of currently active games
    active_games = get_live_games()

    if not active_games:
        print("❌ No live NBA games at the moment.")
    else:
        for game_id in active_games:
            print(f"🔄 Fetching play-by-play for Game ID: {game_id}")
            pbp_df = get_game_playbyplay(game_id)

            if not pbp_df.empty:
                # Save to CSV only if data exists
                filename = f"data/playbyplay_{game_id}.csv"
                pbp_df.to_csv(filename, index=False)
                print(f"✅ Play-by-play data saved: {filename}")
            else:
                print(f"❌ No play-by-play data available for Game ID: {game_id}")
