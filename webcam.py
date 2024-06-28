import cv2
import numpy as np
import mediapipe as mp
import pickle
import time

model_filename = "xgb_model.pkl"
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Mapping dictionary
label_mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'land': 4, 'flip': 5, 'forward': 6, 'backward': 7}

probability_threshold = 0.93
time_threshold = 0.5
gesture_duration = 0
current_gesture = None

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        print("Error: Could not read frame.")
        break

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post-process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = lm.x
                lmy = lm.y
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Convert landmarks to a flat list
        flat_landmarks = [val for sublist in landmarks for val in sublist]
        # Reshape the flat list to match the input format expected by the model
        input_data = np.array(flat_landmarks).reshape(1, -1)

        # Make prediction using the loaded XGBoost model
        predicted_probs = loaded_model.predict_proba(input_data)[0]
        predicted_label_encoded = np.argmax(predicted_probs)

        # Check if the maximum probability is above the threshold
        if predicted_probs[predicted_label_encoded] > probability_threshold:
            predicted_label = [key for key, value in label_mapping.items() if value == predicted_label_encoded][0]

            if predicted_label == current_gesture:
                gesture_duration += 1
            else:
                current_gesture = predicted_label
                gesture_duration = 1

            if gesture_duration >= int(time_threshold * cap.get(cv2.CAP_PROP_FPS)):
                cv2.putText(frame, f"Prediction: {predicted_label} ({predicted_probs[predicted_label_encoded]*100:.2f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    # Add a small delay to control the frame rate
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
