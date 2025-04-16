import pandas as pd

INPUT_FILE = "data/preprocessed_live_playbyplay.csv"
OUTPUT_FILE = "data/targeted_events_for_labeling_v2.csv"

# Load the preprocessed live data
df = pd.read_csv(INPUT_FILE)

# Define keywords for rare/difficult events
keywords = [
    "goaltending", "jump ball", "delay of game", "shot clock",
    "discontinue", "double dribble", "backcourt", "technical",
    "kicked ball", "traveling", "alley oop", "take foul",
    "lane violation", "flagrant", "loose ball", "period start", "period end"
]

# Make sure column is string type
df["structured_event"] = df["structured_event"].fillna("").astype(str)

# Step 1: Filter for problematic events
problem_mask = df["structured_event"].str.lower().apply(
    lambda text: any(keyword in text for keyword in keywords)
)
problematic_sample = df[problem_mask].sample(n=min(200, problem_mask.sum()), random_state=42)
print(f"🔍 Found {len(problematic_sample)} problematic events.")

# Step 2: Determine how many more events are needed to reach 500
n_problematic = len(problematic_sample)
n_remaining = 500 - n_problematic

# Step 3: Sample the remaining from non-problematic events
remaining_df = df.drop(problematic_sample.index)
random_sample = remaining_df.sample(n=min(n_remaining, len(remaining_df)), random_state=42)

# Step 4: Combine both samples
final_sample = pd.concat([problematic_sample, random_sample])
final_sample = final_sample[["time_remaining", "structured_event"]].copy()
final_sample["natural_description"] = ""

# Save to CSV
final_sample.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(final_sample)} total records to {OUTPUT_FILE} for manual labeling.")