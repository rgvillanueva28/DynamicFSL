import time
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

# Stylized Landmark drawing


def draw_styled_landmarks(image, results):
    #     # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(18, 10, 80), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(3, 128, 252), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(158, 24, 129), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(135, 76, 122), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(38, 117, 16), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(83, 135, 69), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark[0:23]]).flatten(
    ) if results.pose_landmarks else np.zeros(23*2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose, lh, rh])


words = ["ako", "bakit", "hindi", "ikaw", "ito",
         "kamusta", "maganda", "magkano", "oo", "salamat"]

model = load_model("bilstm_final.h5")

cap = cv2.VideoCapture(5)

frame_count = 0
recording = False

predicted_word = ""
accuracy_text = ""
color = (21, 209, 0)
green_color = (21, 209, 0)
word_color = (255, 179, 0)

prev_frame_time = 0
new_frame_time = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == 32:  # Record pressing r
            recording = True

            frame_count = 0
            frames = []

            print("Start signing")
            color = green_color

            predicted_word = "Start signing"
            accuracy_text = ""

        elif pressedKey == ord("q"):  # Break pressing q
            break

        if frame_count >= 45:
            frame_count = 0
            recording = False

            res = model.predict(np.expand_dims(frames, axis=0))[0]
            print(words[np.argmax(res)], res[np.argmax(res)])
            color = word_color
            predicted_word = words[np.argmax(res)]
            accuracy = res[np.argmax(res)]
            accuracy_text = "{:.0%}".format(accuracy)
            # accuracy = "testing"
            frames = []

        if recording:
            frame_count += 1
            frames.append(keypoints)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps = str(fps)
        fps_text = "FPS: " + fps

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (225, 80), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 1.0)

        cv2.putText(image, fps_text, (5, 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, green_color)

        cv2.putText(image, predicted_word, (5, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, color)

        cv2.putText(image, accuracy_text, (5, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color)

        cv2.imshow('Dynamic FSL', image)

    cap.release()
    cv2.destroyAllWindows()
