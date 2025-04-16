import os
import pandas as pd

INPUT_FOLDER = "data"
OUTPUT_FILE = "data/preprocessed_live_playbyplay.csv"

team_name_map = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets", "CHI": "Bulls",
    "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets", "DET": "Pistons", "GSW": "Warriors",
    "HOU": "Rockets", "IND": "Pacers", "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies",
    "MIA": "Heat", "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns", "POR": "Trail Blazers",
    "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors", "UTA": "Jazz", "WAS": "Wizards"
}

def load_and_merge_data():
    dfs = []
    for file in os.listdir(INPUT_FOLDER):
        if file.startswith("live_playbyplay_") and file.endswith(".csv"):
            game_id = file.replace("live_playbyplay_", "").replace(".csv", "")
            df = pd.read_csv(os.path.join(INPUT_FOLDER, file))
            df["gameId"] = game_id  # Extract from filename
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def preprocess_event(row):
    desc = str(row.get("description", "") or "")
    team_abbr = str(row.get("teamTricode", "") or "")
    full_team_name = team_name_map.get(team_abbr, team_abbr)

    if "TEAM" in desc.upper() or team_abbr in desc:
        desc = desc.replace("TEAM", full_team_name).replace(team_abbr, full_team_name)

    return f"{team_abbr}: {desc}" if team_abbr else desc

# Load and preprocess
df = load_and_merge_data()
df["time_remaining"] = df.get("clock", "")
df["structured_event"] = df.apply(preprocess_event, axis=1)

# Sort and reset index
df = df.sort_values(by=["gameId", "period", "clock"], ascending=[True, True, False])
df = df[["time_remaining", "structured_event"]]
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Preprocessed play-by-play data saved to {OUTPUT_FILE}")