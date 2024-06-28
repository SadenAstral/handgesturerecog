import cv2
import os

# Create a directory to store the dataset
dataset_path = "gesture_dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

gesture_labels = {0: "up", 1: "down", 2: "left", 3: "right", 4: 'land', 5: 'flip', 6: 'forward', 7: 'backward'}

# Loop for data collection
while True:
    # Read each frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Display instructions for labeling
    cv2.putText(frame,
                "Press 'u' for 'Up', 'd' for 'Down', 'l' for 'Left', 'r' for 'Right', 's' for 'Land', 'f' for 'Flip', 'w' for 'Forward', 'b' for 'Backward'",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Capture Gestures", frame)

    # Collect keypress to label the gesture
    key = cv2.waitKey(1)

    if key & 0xFF == ord('u'):
        label = 0  # Up
    elif key & 0xFF == ord('d'):
        label = 1  # Down
    elif key & 0xFF == ord('l'):
        label = 2  # Left
    elif key & 0xFF == ord('r'):
        label = 3  # Right
    elif key & 0xFF == ord('s'):
        label = 4  # Land
    elif key & 0xFF == ord('f'):
        label = 5  # Flip
    elif key & 0xFF == ord('w'):
        label = 6  # Forward
    elif key & 0xFF == ord('b'):
        label = 7  # Backward
    else:
        continue

    # Save the labeled image
    image_name = f"{gesture_labels[label]}_{len(os.listdir(dataset_path)) + 1}.png"
    image_path = os.path.join(dataset_path, image_name)
    cv2.imwrite(image_path, frame)
    print(f"Image saved: {image_name}")

# Release resources when the loop is terminated
cap.release()
cv2.destroyAllWindows()
