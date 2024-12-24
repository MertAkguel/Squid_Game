import pandas as pd
from datetime import datetime
import os
import cv2
import numpy as np


def save_to_csv(player_name, score):
    filename = "game_history.csv"
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame({
        "Player Name": [player_name],
        "Score": [score],
        "Date": [date]
    })

    # Append the new data or create a new file if it doesn't exist
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data

    updated_data.to_csv(filename, index=False)


def check_high_score(player_name, score):
    filename = "game_history.csv"

    if not os.path.exists(filename):
        return False  # No history exists yet

    # Load existing data
    data = pd.read_csv(filename)

    # Filter the data for the specific player
    player_data = data[data["Player Name"] == player_name]

    # Find the player's high score
    player_high_score = player_data["Score"].max() if not player_data.empty else 0

    # Return True if the current score exceeds the player's personal best
    return score > player_high_score


def calculate_motion(prev_frame, curr_frame, mask):
    # Apply the mask to both frames
    prev_roi = cv2.bitwise_and(prev_frame, prev_frame, mask=mask)
    curr_roi = cv2.bitwise_and(curr_frame, curr_frame, mask=mask)

    # Convert to grayscale for simplicity
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Threshold the difference to binarize
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Calculate the motion magnitude (sum of non-zero pixels)
    motion_magnitude = np.sum(diff_thresh > 0)
    return motion_magnitude
