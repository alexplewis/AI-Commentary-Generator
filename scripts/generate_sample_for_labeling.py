import pandas as pd

INPUT_FILE = "data/preprocessed_playbyplay.csv"
OUTPUT_FILE = "data/manual_annotation_sample.csv"

# Load full dataset
df = pd.read_csv(INPUT_FILE)

# Randomly sample 5,500 records for manual labeling
sample_size = 5500  # Adjust this as needed
df_sample = df.sample(n=sample_size, random_state=42)

# Add an empty "natural_description" column for manual annotation
df_sample["natural_description"] = ""

# Save sample file
df_sample.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Random sample of {sample_size} records saved to {OUTPUT_FILE}. Ready for manual labeling!")
