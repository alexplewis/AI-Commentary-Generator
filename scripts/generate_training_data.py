import pandas as pd

INPUT_FILE = "data/preprocessed_playbyplay.csv"
OUTPUT_FILE = "data/training_data.csv"

# Load structured play-by-play data
df = pd.read_csv(INPUT_FILE)

# Simply store the structured event descriptions as training data
df.rename(columns={"event_description": "input_event"}, inplace=True)
df["ai_commentary"] = ""  # Leave blank for AI-generated output

# Save to training data file
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Training data saved to {OUTPUT_FILE}. AI model will learn to generate commentary from structured event descriptions.")
