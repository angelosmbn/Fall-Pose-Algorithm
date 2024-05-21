from ultralytics import YOLO
import numpy as np
import time
import cv2  # Import OpenCV
import os

# Load model
model = YOLO('yolov8s-pose.pt')

cnn_model = YOLO('sit_stand.pt')

# Initialize an empty array to store keypoints
keypoints_history = []

# Function to store latest keypoints
def store_keypoints(keypoints):
    global keypoints_history
    
    # Append the new keypoints to the history
    keypoints_history.append(keypoints)
    
    # Keep only the latest 30 entries
    if len(keypoints_history) > 30:
        keypoints_history = keypoints_history[-30:]


def predict_action(image_path):
    cnn_results = cnn_model(source=image_path, show=True, conf=0.80, stream=True)

    for cnn_result in cnn_results:
        bounding_boxes = cnn_result.boxes.cpu().numpy()

        class_name = bounding_boxes.cls
        confidence = bounding_boxes.conf
        if class_name.size > 0:
            if class_name == 0:
                print("Sitting - ", confidence)
                return "Sitting"
            elif class_name == 1:
                print("Standing - ", confidence)
                return "Standing"
        else:
            print("Unknown class")
            return "Unknown class"
        
# Function to generate a unique filename
def generate_filename(folder, prefix='fall_detected_', ext='.jpg'):
    existing_files = os.listdir(folder)
    existing_numbers = [int(f[len(prefix):-len(ext)]) for f in existing_files if f.startswith(prefix) and f.endswith(ext)]
    if existing_numbers:
        new_number = max(existing_numbers) + 1
    else:
        new_number = 0
    return os.path.join(folder, f'{prefix}{new_number}{ext}')

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')





prev_upper_body = None
prev_lower_body = None

velocity_threshold = .2
fall_counter = 0
start_time = None  # Initialize start_time to None

fall_detected = False
fall_countdown = None

# Open video capture (using webcam)
cap = cv2.VideoCapture(0)  # Change the source as needed



# Loop for continuous processing
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break
    
    # Inference
    results = model(source=frame, show=True, conf=0.5, stream=True)
    
    if fall_detected:
        if time.time() - fall_countdown >= 2:
            print("Fall detected! Saving image...")

            # Generate a unique filename
            filename = generate_filename('results')
            cv2.imwrite(filename, frame)
            print(f'Image saved as {filename}')

            # Predict action
            action = predict_action(filename)
            if action == "Sitting":
                print("FALL IS INVALID. Person is sitting.")
            elif action == "Standing":
                print("FALL IS INVALID. Person is standing.") 
            else:
                print("FALL IS VALID. CHECK YOUR EMAIL & SMS.")

                #print the keypoints stored.
                print(keypoints_history)

                #TERMINATE THE PROGRAM AND SEND EMAIL
                break

            fall_counter = 0
            fall_detected = False
            
    for result in results:
        keypoints = result.keypoints.cpu().numpy()

        xyn = keypoints.xyn
        #xy = keypoints.xy

        #store the latest 30 keypoints
        store_keypoints(xyn)

        # Check if any keypoints are detected
        if xyn.size == 0 or len(xyn) != 1:
            print("No keypoints detected.")
            prev_upper_body = None
            prev_lower_body = None
            continue

        for group in xyn:
            if len(group) < 15:  # Ensure there are enough keypoints in the group
                continue

            left_shoulder = group[5][1]  # Accessing the y-coordinate of index 5
            right_shoulder = group[6][1]

            left_knee = group[13][1]
            right_knee = group[14][1]

            left_hip = group[11][1]
            right_hip = group[12][1]

            upper_body = left_shoulder + right_shoulder + left_hip + right_hip
            lower_body = left_knee + right_knee + left_hip + right_hip

            if left_shoulder != 0 and right_shoulder != 0 and left_knee != 0 and right_knee != 0:
                if prev_upper_body is not None and prev_lower_body is not None:
                    upper_body_velocity = abs(upper_body - prev_upper_body)
                    lower_body_velocity = abs(lower_body - prev_lower_body)

                    ubk_lbk = abs(upper_body - lower_body)

                    if upper_body_velocity > velocity_threshold:
                        if fall_counter == 0:
                            start_time = time.time()  # Initialize timer at the beginning

                        fall_counter += 1
                        #print(fall_counter, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        print("FALL COUNTER: ", fall_counter)
                        if fall_counter >= 2:
                            if ubk_lbk < 0.35:
                                fall_detected = True
                                fall_countdown = time.time()  # Start countdown timer
                                print("Fall detected, wait for validation.")
                            #else:
                                #print("*********************************************************************failed ubk_lbk =", ubk_lbk)

                    # Reset fall_counter after 3 seconds if it's not zero
                    if fall_counter != 0 and time.time() - start_time >= 3:
                        fall_counter = 0
                        start_time = time.time()  # Reset timer
                        #print("Fall counter reset. ----------------------------------------------------------------------")

            prev_upper_body = upper_body
            prev_lower_body = lower_body

cap.release()  # Release the capture
cv2.destroyAllWindows()  # Close OpenCV windows

"""
# Keypoints
0 - Nose
1 - Left Eye
2 - Right Eye
3 - Left Ear
4 - Right Ear
5 - Left Shoulder
6 - Right Shoulder
7 - Left Elbow
8 - Right Elbow
9 - Left Wrist
10 - Right Wrist
11 - Left Hip
12 - Right Hip
13 - Left Knee
14 - Right Knee
15 - Left Ankle
16 - Right Ankle

I will provide you the arrays of the latest collected keypoints before the fall happened.
Analyze which body part was the first one to hit the floor.
Create an simple anaylsis of the fall. Just give me the analysis. Make it clear especially if the head also hit the floor.
"""