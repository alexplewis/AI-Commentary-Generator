import pandas as pd
import random
import re

INPUT_FILE = "data/preprocessed_playbyplay.csv"
OUTPUT_FILE = "data/training_data.csv"

# Load preprocessed play-by-play data
df = pd.read_csv(INPUT_FILE)

# Expanded AI-generated responses
commentary_templates = {
    "3PT Jump Shot": [
        "[PLAYER] drills a deep three!", 
        "[PLAYER] connects from downtown!", 
        "[PLAYER] splashes a long-range shot!",
        "[PLAYER] drills a deep three!", 
        "[PLAYER] connects from way downtown!", 
        "[PLAYER] buries a long-range shot!", 
        "[PLAYER] lets it fly... and hits!"
    ],
    "Jump Shot": [
        "[PLAYER] pulls up and nails the jumper!", 
        "[PLAYER] sinks a mid-range shot!", 
        "[PLAYER] buries the jump shot!",
        "[PLAYER] pulls up and knocks it down!", 
        "[PLAYER] nails the mid-range jumper!", 
        "[PLAYER] rises up and buries it!"
    ],
    "Dunk": [
        "[PLAYER] slams it home!", 
        "[PLAYER] throws it down!", 
        "[PLAYER] hammers it in!"
    ],
    "Layup": [
        "[PLAYER] drives inside and lays it in!", 
        "[PLAYER] finishes strong at the rim!", 
        "[PLAYER] gets the bucket in traffic!",
        "[PLAYER] drives inside and lays it in!", 
        "[PLAYER] attacks the rim for two!", 
        "[PLAYER] glides in for the score!"
    ],
    "Rebound": [
        "[PLAYER] secures the board.", 
        "[PLAYER] snatches the ball off the glass.", 
        "[PLAYER] comes down with the rebound."
    ],
    "Steal": [
        "[PLAYER] steals the ball from [SECOND_PLAYER]!", 
        "[PLAYER] picks [SECOND_PLAYER]'s pocket!", 
        "[PLAYER] swipes it away from [SECOND_PLAYER]!"
    ],
    "Foul": [
        "[PLAYER] is charged with a foul.", 
        "[PLAYER] was assessed a foul."
    ],
    "Technical Foul": [
        "Technical foul assessed to [PLAYER].", 
        "[PLAYER] hit with a technical foul.", 
        "[PLAYER] T'd up by the ref!"
    ],
    "Turnover": [
        "[PLAYER] loses the ball.", 
        "[PLAYER] turns it over.", 
        "[PLAYER] coughs it up."
    ],
    "Block": [
        "[PLAYER] swats [SECOND_PLAYER]'s shot away!", 
        "[PLAYER] denies [SECOND_PLAYER] at the rim!", 
        "[PLAYER] sends [SECOND_PLAYER]'s shot packing!"
    ],
    "Substitution": [
        "[PLAYER] comes in for [SECOND_PLAYER].", 
        "[PLAYER] checks in for [SECOND_PLAYER].", 
        "[PLAYER] enters the game for [SECOND_PLAYER]."
    ],
        "Free Throw": [
        "Free throws for [PLAYER]."
    ],
    "Missed Free Throw": [
        "[PLAYER] misses the free throw after the technical foul."
    ],
    "Timeout": [
        "[PLAYER] calls a timeout.", 
        "Timeout [PLAYER]."
    ],
    "Ejection": [
        "[PLAYER] has been ejected from the game."
    ],
    "Jump Ball": [
        "There's a jump ball between [PLAYER] and [SECOND_PLAYER], and it's tipped to [THIRD_PLAYER]."
    ]
}

# Missed shot responses
missed_shot_templates = {
    "3PT Jump Shot": [
        "[PLAYER] fires from deep but can't connect.", 
        "[PLAYER] launches a three but misses."
    ],
    "Jump Shot": [
        "[PLAYER] takes a jumper but can't get it to fall.", 
        "[PLAYER] pulls up but it's off the mark."
    ],
    "Layup": [
        "[PLAYER] drives inside but the layup won’t fall.", 
        "[PLAYER] attacks the basket but can't finish."
    ]
}

# Extract player names from event
def extract_players(event):
    # Remove stat-related parentheses like "(1 STL)"
    event_cleaned = re.sub(r"\([\d\w\s]+\)", "", event)
    words = event_cleaned.split()
    players = []
    
    for i, word in enumerate(words):
        if word.istitle() and word not in ["Jr.", "Sr.", "III"]:
            players.append(word)
    
    if len(players) == 0:
        return "A player", None, None
    elif len(players) == 1:
        return players[0], None, None
    elif len(players) == 2:
        return players[0], players[1], None
    else:
        return players[0], players[1], players[2]

# Generate AI commentary
training_data = []
for _, row in df.iterrows():
    event = row["event_description"]

    # Determine if it's a missed shot
    is_missed_shot = "miss" in event.lower()
    
    # Extract player names
    player_name, second_player, third_player = extract_players(event)

    # Determine which template set to use
    templates = missed_shot_templates if is_missed_shot else commentary_templates

    # Find event type
    matched_key = None
    for key in templates.keys():
        if key in event:
            matched_key = key
            break

    # Handle special cases
    if "BLOCK" in event and second_player:
        commentary = random.choice(commentary_templates["Block"]).replace("[PLAYER]", player_name).replace("[SECOND_PLAYER]", second_player)
    elif "STEAL" in event and second_player:
        # Swap order to correctly credit the player who made the steal
        commentary = random.choice(commentary_templates["Steal"]).replace("[PLAYER]", player_name).replace("[SECOND_PLAYER]", second_player)
    elif "T.FOUL" in event:
        commentary = random.choice(commentary_templates["Technical Foul"]).replace("[PLAYER]", player_name)
    elif ".FOUL" in event:
        commentary = random.choice(commentary_templates["Foul"]).replace("[PLAYER]", player_name)
    elif "offensive foul" in event.lower():
        commentary = f"{player_name} commits an offensive foul."
    elif "Jump Ball" in event and "vs." in event and "Tip to" in event:
        # Extract player names correctly
        match = re.search(r"Jump Ball (\w+) vs. (\w+): Tip to (\w+)", event)
        if match:
            player_name, second_player, third_player = match.groups()
            commentary = random.choice(commentary_templates["Jump Ball"]).replace("[PLAYER]", player_name).replace("[SECOND_PLAYER]", second_player).replace("[THIRD_PLAYER]", third_player)
        else:
            commentary = "Jump ball in play."
    elif "Free Throw" in event:
        commentary = random.choice(commentary_templates["Free Throw"]).replace("[PLAYER]", player_name)
    elif "MISS" in event and "Free Technical" in event:
        commentary = f"{player_name} misses the free throw after the technical foul."
    elif "MAKE" in event and "Free Technical" in event:
        commentary = f"{player_name} makes the free throw after the technical foul."
    elif "Timeout" in event:
        commentary = random.choice(commentary_templates["Timeout"]).replace("[PLAYER]", player_name)
    elif "Ejection" in event:
        commentary = random.choice(commentary_templates["Ejection"]).replace("[PLAYER]", player_name)
    elif "SUB:" in event and "FOR" in event:
        match = re.search(r"SUB:\s*(\w+)\s*FOR\s*(\w+)", event)
        if match:
            player_name, second_player = match.groups()
            commentary = random.choice(commentary_templates["Substitution"]).replace("[PLAYER]", player_name).replace("[SECOND_PLAYER]", second_player)
        else:
            commentary = f"{player_name} checks into the game."
    elif matched_key:
        commentary = random.choice(templates[matched_key]).replace("[PLAYER]", player_name)
    elif any(shot_type in event.lower() for shot_type in ["shot", "fadeaway", "hook", "floating", "floater"]):
        commentary = random.choice(commentary_templates["Jump Shot"]).replace("[PLAYER]", player_name)
    else:
        # General event handling (fixes "makes a play" issue)
        if "rebound" in event.lower():
            commentary = f"{player_name} secures the rebound."
        elif "Turnover" in event:
            commentary = f"{player_name} turns the ball over."
        elif "Foul" in event:
            commentary = f"{player_name} is charged with a foul."
        else:
            # Preserve original event text without parentheses
            commentary = re.sub(r"\(.*?\)", "", event).strip()

    # Detect and add assist information
    assist_match = re.search(r"\(([^)]+) AST\)", event)
    if assist_match:
        assist_player = assist_match.group(1).split()[0]  
        commentary += f" {assist_player} assists."

    training_data.append([event, commentary])

# Convert to DataFrame and save
train_df = pd.DataFrame(training_data, columns=["input_event", "ai_commentary"])
train_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Training data saved to {OUTPUT_FILE}. AI model will learn correct phrasing.")