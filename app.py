import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter
import time
import threading
from flask import Flask, Response, jsonify, render_template, request
import pygame
import os

# Inisialisasi pygame mixer sekali saat start app
pygame.mixer.init()

# Path suara bip, gunakan path absolut atau relatif
BEEP_PATH = r"E:\! Laskar Ai\CAPSTONE\gesture-link-main\utils\beep.mp3"

def bip():
    try:
        sound = pygame.mixer.Sound(BEEP_PATH)
        sound.play()
    except Exception as e:
        print(f"Error mainkan suara bip: {e}")

# Load model & label
model = tf.keras.models.load_model("model/best_model.h5")
label_classes = np.load("model/label_classes.npy")

# MediaPipe Hands init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

# Shared variables
current_text = ""
prediction_buffer = []
last_capture_time = time.time()
lock = threading.Lock()

def generate_frames():
    global current_text, prediction_buffer, last_capture_time, no_hand_detected_since
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak tersedia")

    no_hand_detected_since = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        data = []

        if result.multi_hand_landmarks:
            # Reset timer kalau ada tangan
            no_hand_detected_since = None

            handedness = result.multi_handedness
            landmarks = result.multi_hand_landmarks
            handed_landmarks = sorted(zip(handedness, landmarks),
                                      key=lambda x: x[0].classification[0].label)

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

                with lock:
                    prediction_buffer.append(pred_label)

                # Draw landmarks
                for lm in landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        else:
            # Kalau gak ada tangan, kosongkan buffer prediksi lama agar gak terpakai
            with lock:
                prediction_buffer.clear()

            # Mulai hitung waktu gak ada tangan
            if no_hand_detected_since is None:
                no_hand_detected_since = time.time()

        current_time = time.time()

        # Jika sudah 5 detik lebih sejak tangan terakhir terdeteksi
        if no_hand_detected_since is not None and (current_time - no_hand_detected_since >= 5.0):
            with lock:
                current_text += " "  # tambahkan spasi karena gak ada tangan
                bip()  # suara bip untuk tandai spasi
            no_hand_detected_since = None  # reset timer

            # reset last_capture_time juga supaya logika lain tetap jalan normal
            last_capture_time = current_time

        # Jika tangan terdeteksi dan sudah lewat 5 detik sejak capture terakhir
        elif current_time - last_capture_time >= 5.0 and prediction_buffer:
            with lock:
                most_common = Counter(prediction_buffer).most_common(1)[0][0]
                current_text += most_common
                prediction_buffer.clear()
                bip()
            last_capture_time = current_time

        # Encode frame untuk stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    with lock:
        text = current_text
    return jsonify({'prediction': text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global current_text
    with lock:
        current_text = ""
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True)
