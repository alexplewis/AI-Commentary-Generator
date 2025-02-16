import pandas as pd

INPUT_FILE = "data/preprocessed_playbyplay.csv"
OUTPUT_FILE = "data/event_rewriting_data.csv"

# Load structured play-by-play data
df = pd.read_csv(INPUT_FILE)

# Keep only relevant columns
df = df[["time_remaining", "event_description"]]
df.rename(columns={"event_description": "structured_event"}, inplace=True)

# Add an empty "natural_description" column for AI-generated commentary
df["natural_description"] = ""

# Save the dataset
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Event rewriting dataset created: {OUTPUT_FILE}")
