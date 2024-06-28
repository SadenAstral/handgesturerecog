import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Define the path to your image folder
image_folder = "gesture_dataset"

# Initialize lists to store landmark data and labels
landmarks_list = []
labels_list = []

# Loop through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        # Read the image
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(img_rgb)

        # Extract landmarks
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    landmarks.extend([lm.x, lm.y])
            landmarks_list.append(landmarks)

            # Extract label from filename (e.g., "up", "down", "flip")
            label = filename.split('_')[0]
            labels_list.append(label)

# Create a DataFrame
columns = [f"Landmark_{i}" for i in range(1, 22) for _ in ["X", "Y"]]
columns.append("Label")
df = pd.DataFrame(np.concatenate([landmarks_list, np.array(labels_list).reshape(-1, 1)], axis=1), columns=columns)

# Convert columns to numeric

# Display the DataFrame
print(df)
csv_filename = "training_set.csv"
df.to_csv(csv_filename, index=False)