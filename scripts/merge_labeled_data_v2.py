import pandas as pd

original_file = "data/final_training_data.csv"
new_file = "data/targeted_events_for_labeling_v2.csv"
output_file = "data/final_training_data_v2.csv"

# Read original CSV safely
with open(original_file, "r", encoding="utf-8", errors="replace") as f:
    original_df = pd.read_csv(f)

# Read new labeled CSV safely
with open(new_file, "r", encoding="utf-8", errors="replace") as f:
    new_df = pd.read_csv(f)

# Combine and deduplicate
combined_df = pd.concat([original_df, new_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=["structured_event", "natural_description"])

# Save merged dataset
combined_df.to_csv(output_file, index=False)

print("✅ Merged training dataset saved to:", output_file)