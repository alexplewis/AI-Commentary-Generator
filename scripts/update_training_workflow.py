import pandas as pd

# File paths
ORIGINAL_FILE = "data/final_training_data.csv"
NEW_FILE = "data/targeted_events_for_labeling_v2.csv"

# Load both datasets
original_df = pd.read_csv(ORIGINAL_FILE)
new_df = pd.read_csv(NEW_FILE)

# Rename columns for consistency
if "event_description" in new_df.columns:
    new_df = new_df.rename(columns={"event_description": "structured_event"})
if "ai_commentary" in new_df.columns:
    new_df = new_df.rename(columns={"ai_commentary": "natural_description"})

# Drop any rows without manual commentary
new_df = new_df.dropna(subset=["natural_description"])

# Merge datasets
merged_df = pd.concat([original_df, new_df], ignore_index=True)
merged_df.to_csv("data/updated_training_data.csv", index=False)

print("✅ Saved merged dataset to data/updated_training_data.csv")