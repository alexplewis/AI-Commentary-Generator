import pandas as pd

# Load preprocessed data
df = pd.read_csv("data/preprocessed_live_playbyplay.csv")

# Event keywords to extract
RARE_EVENTS = [
    "loose ball foul", "period start", "period end", "shot clock", "goaltending",
    "lane violation", "traveling", "double dribble", "backcourt", "technical",
    "take foul", "kicked ball", "discontinue", "delay of game", "flagrant", "alley oop"
]

# Extract rare or complex events
rare_df = df[df["event_description"].str.contains("|".join(RARE_EVENTS), case=False, na=False)]

# Extract team-based plays using teamId or teamTricode
team_plays_df = df[df["event_description"].str.contains("TEAM", case=False, na=False)]

# Also grab first 75 records from each original live file
sampled_dfs = []
file_paths = [
    "data/live_playbyplay_0022400966.csv",
    "data/live_playbyplay_0022400967.csv",
    "data/live_playbyplay_0022400965.csv",
    "data/live_playbyplay_0022400963.csv"
]

for path in file_paths:
    try:
        game_df = pd.read_csv(path)
        sampled_dfs.append(game_df.head(75))
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")

sampled_df = pd.concat(sampled_dfs, ignore_index=True)

# Combine everything and drop duplicates
final_df = pd.concat([rare_df, team_plays_df, sampled_df], ignore_index=True).drop_duplicates()

# Save to a new labeling file
final_df.to_csv("data/targeted_events_for_labeling_v2.csv", index=False)
print("✅ Saved extracted records for manual labeling to data/targeted_events_for_labeling_v2.csv")
