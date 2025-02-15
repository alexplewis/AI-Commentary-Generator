import os
import pandas as pd
import re

# Folder where play-by-play data is stored
DATA_DIR = "data/"
OUTPUT_FILE = "data/preprocessed_playbyplay.csv"

def clean_event_description(event):
    """
    Cleans and standardizes the play-by-play event description.
    :param event: Raw event description from the dataset.
    :return: Reformatted event description.
    """
    if pd.isna(event) or event.strip() == "":
        return None

    event = event.strip()

    # Standardize missed shots
    event = re.sub(r"MISS (\w+) (\d+)' (\w+)", r"\1 attempts a \2-foot \3 but misses.", event)

    # Standardize made shots
    event = re.sub(r"(\w+) (\d+)' (\w+)", r"\1 sinks a \2-foot \3.", event)

    # Standardize rebounds
    event = re.sub(r"(\w+) REBOUND", r"\1 grabs the rebound.", event)

    # Standardize steals & turnovers
    event = re.sub(r"(\w+) STEAL (\w+) Lost Ball Turnover", r"\1 steals the ball from \2.", event)

    # Standardize fouls
    event = re.sub(r"(\w+) OFF.Foul", r"\1 commits an offensive foul.", event)

    return event

def preprocess_pbp_data():
    """
    Reads all play-by-play files, processes them, and saves a structured dataset.
    """
    all_data = []

    # Loop through all play-by-play files
    for filename in os.listdir(DATA_DIR):
        if filename.startswith("playbyplay_") and filename.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, filename)
            print(f"🔄 Processing {filename}...")

            # Read CSV file
            df = pd.read_csv(file_path)

            # Keep only relevant columns
            df = df[['time_remaining', 'quarter', 'home_event', 'away_event']].dropna(how="all")

            # Convert timestamps into readable format (e.g., "Q3 - 10:45")
            df['time_remaining'] = df['quarter'].apply(lambda q: f"Q{q}") + " - " + df['time_remaining']

            # Merge home & away event descriptions
            df['event_description'] = df['home_event'].fillna('') + df['away_event'].fillna('')
            df['event_description'] = df['event_description'].apply(clean_event_description)

            # Remove empty event descriptions
            df = df.dropna(subset=['event_description'])

            # Keep only necessary columns
            df = df[['time_remaining', 'event_description']]
            
            all_data.append(df)

    # Concatenate all processed data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Preprocessed data saved to {OUTPUT_FILE}")
    else:
        print("❌ No valid play-by-play data found.")

if __name__ == "__main__":
    preprocess_pbp_data()
