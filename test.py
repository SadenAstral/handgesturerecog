import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load the saved XGBoost model
model_filename = "xgb_model.pkl"
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Mapping dictionary
label_mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'land': 4, 'flip': 5, 'forward': 6, 'backward': 7}

# Function to get landmarks from an image
def get_landmarks(image_path):
    # Read the image
    img = cv2.imread(image_path)
    x, y, c = img.shape

    # Initialize Mediapipe hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(img_rgb)

    # Extract landmarks
    landmarks = []
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = lm.x
                lmy = lm.y
                landmarks.append([lmx, lmy])

    return landmarks

# Example image path
image_path = "gesture_dataset/backward_778.png"

# Get landmarks from the image
landmarks = get_landmarks(image_path)

# Convert landmarks to a flat list
flat_landmarks = [val for sublist in landmarks for val in sublist]

# Reshape the flat list to match the input format expected by the model
input_data = np.array(flat_landmarks).reshape(1, -1)

# Make prediction using the loaded XGBoost model
predicted_label_encoded = loaded_model.predict(input_data)[0]

# Convert the encoded label back to the original label using the mapping dictionary
predicted_label = [key for key, value in label_mapping.items() if value == predicted_label_encoded][0]

# Print the predicted label
print(f"Predicted Label: {predicted_label}")
