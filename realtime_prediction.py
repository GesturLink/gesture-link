import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter
import time
import pygame
from utils.sound import bip

# Load model & label
model = tf.keras.models.load_model("model/best_model.h5")
label_classes = np.load("model/label_classes.npy")

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Variabel teks & prediksi buffer
current_text = ""
prediction_buffer = []
last_capture_time = time.time()

# Mulai webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera tidak terdeteksi.")
    exit()

print("[INFO] Sistem mulai. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    data = []

    if result.multi_hand_landmarks:
        handedness = result.multi_handedness
        landmarks = result.multi_hand_landmarks
        handed_landmarks = sorted(zip(handedness, landmarks), key=lambda x: x[0].classification[0].label)

        if len(handed_landmarks) > 2:
            handed_landmarks = handed_landmarks[:2]

        for _, lm in handed_landmarks:
            for point in lm.landmark:
                data.extend([point.x, point.y, point.z])

        while len(data) < 126:
            data.extend([0.0] * 3)

        if len(data) == 126:
            input_data = np.array(data).reshape(1, 126)
            prediction = model.predict(input_data, verbose=0)
            pred_index = np.argmax(prediction)
            pred_label = label_classes[pred_index]
            prediction_buffer.append(pred_label)

            # Draw landmarks
            for lm in landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    # Setiap 1 detik → ambil hasil prediksi terbaik dari buffer
    current_time = time.time()
    if current_time - last_capture_time >= 5.0:
        if prediction_buffer:
            most_common = Counter(prediction_buffer).most_common(1)[0][0]
            current_text += most_common
            print(f"[+] Ditambahkan: {most_common}")
            bip()
        prediction_buffer = []
        last_capture_time = current_time

    # Tampilkan teks di layar
    cv2.putText(frame, f"Teks: {current_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Tampilkan preview
    cv2.imshow("BISINDO Gesture to Text", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_text = ""
        print("[CLEAR] Teks dikosongkan.")

cap.release()
cv2.destroyAllWindows()
