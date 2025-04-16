import pandas as pd

# File paths
ORIGINAL_FILE = "data/labeled_training_data.csv"  # 2,500 labeled records
NEW_FILE = "data/targeted_events_for_labeling.csv"  # 500 newly labeled records
OUTPUT_FILE = "data/final_training_data.csv"

# Function to clean text encoding issues
def clean_text(text):
    if isinstance(text, str):
        text = text.encode("utf-8", "ignore").decode("utf-8")  # Ensure UTF-8 encoding
        text = text.replace("‚Äôs", "'s").replace("‚Äô", "'")   # Fix apostrophes
        text = text.replace("Äôs", "'s").replace("Äô", "'")     # Additional edge cases
        text = text.replace("„Ä¶", "...")                        # Handle ellipses
        text = text.replace("‚Äù", '"').replace("‚Äú", '"')     # Fix quotation marks
        text = text.replace("â€“", "-").replace("â€”", "—")     # Fix dashes
        text = text.replace("Ã©", "é").replace("Ã±", "ñ")       # Fix common accented characters
        text = text.replace("Â", "")                           # Remove extra encoding artifacts
        return text.strip()
    return text

# Load datasets with UTF-8 encoding
try:
    original_df = pd.read_csv(ORIGINAL_FILE, encoding="utf-8", encoding_errors="replace")
    new_df = pd.read_csv(NEW_FILE, encoding="utf-8", encoding_errors="replace")
except UnicodeDecodeError:
    print("⚠️ UnicodeDecodeError detected. Trying alternate encoding...")
    original_df = pd.read_csv(ORIGINAL_FILE, encoding="ISO-8859-1")
    new_df = pd.read_csv(NEW_FILE, encoding="ISO-8859-1")

# Apply text cleaning function
for column in ["input_event", "ai_commentary"]:
    if column in original_df.columns:
        original_df[column] = original_df[column].astype(str).apply(clean_text)
    if column in new_df.columns:
        new_df[column] = new_df[column].astype(str).apply(clean_text)

# Combine both datasets, ensuring no duplicates
merged_df = pd.concat([original_df, new_df]).drop_duplicates()

# Save merged dataset with proper encoding
merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"✅ Merged dataset saved as {OUTPUT_FILE} with {len(merged_df)} total records.")
