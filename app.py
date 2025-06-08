import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter
import time
import random
import threading
from flask import Flask, Response, jsonify, render_template, request, session, redirect, url_for
import pygame
import os
from utils.sound import bip

model = tf.keras.models.load_model("model/best_model.h5")
label_classes = np.load("model/label_classes.npy")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)
app.secret_key = 'gesturelink-secret-key'

current_text = ""
prediction_buffer = []
last_capture_time = time.time()
lock = threading.Lock()

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_learn')
def video_learn():
    return Response(generate_frame_single(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global current_text, prediction_buffer, last_capture_time, lock
    no_hand_detected_since = None
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam tidak tersedia")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            data = []

            if result.multi_hand_landmarks:
                no_hand_detected_since = None
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

                    with lock:
                        prediction_buffer.append(pred_label)

                    for lm in landmarks:
                        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            else:
                with lock:
                    prediction_buffer.clear()

                if no_hand_detected_since is None:
                    no_hand_detected_since = time.time()

            current_time = time.time()

            if no_hand_detected_since is not None and (current_time - no_hand_detected_since >= 5.0):
                with lock:
                    current_text += " "
                    bip()
                no_hand_detected_since = None
                last_capture_time = current_time

            elif current_time - last_capture_time >= 5.0 and prediction_buffer:
                with lock:
                    most_common = Counter(prediction_buffer).most_common(1)[0][0]
                    current_text += most_common
                    prediction_buffer.clear()
                    bip()
                last_capture_time = current_time

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

def generate_frame_single():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak tersedia")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/get_prediction')
def get_prediction():
    with lock:
        text = current_text
    return jsonify({'prediction': text})

@app.route('/get_single_prediction')
def get_single_prediction():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'prediction': ''})

    success, frame = cap.read()
    cap.release()

    if not success:
        return jsonify({'prediction': ''})

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
            return jsonify({'prediction': pred_label})

    return jsonify({'prediction': ''})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global current_text
    with lock:
        current_text = ""
    return jsonify({'status': 'cleared'})

@app.route('/learn')
def learn():
    completed = session.get('completed', [])
    return render_template('learn.html', all_chars=getChars(), completed=completed)

@app.route('/learn/<label>')
def learn_letter(label):
    label = label.upper()
    if label not in getChars():
        return "Label tidak valid", 404
    return render_template('learn_single.html', label=label)

@app.route('/mark_complete/<label>', methods=['POST'])
def mark_complete(label):
    label = label.upper()
    valid_labels = getChars()
    if label not in valid_labels:
        return "Label tidak valid", 400

    completed = session.get('completed', [])
    if label not in completed:
        completed.append(label)
        session['completed'] = completed
    return "OK", 200

@app.route('/get_completed')
def get_completed():
    return jsonify(session.get('completed', []))

@app.route('/quiz')
def quiz():
    quiz_labels = random.sample(getChars(), 10)
    session['quiz_labels'] = quiz_labels
    session['quiz_index'] = 0
    session['quiz_score'] = 0
    return render_template('quiz.html')

@app.route('/quiz/next', methods=['POST'])
def quiz_next():
    prediction = request.json.get('prediction')
    index = session.get('quiz_index', 0)
    labels = session.get('quiz_labels', [])
    score = session.get('quiz_score', 0)

    if not labels or index >= len(labels):
        return jsonify({'done': True, 'score': score})

    if prediction is None:
        return jsonify({'done': False, 'label': labels[index], 'index': index + 1})

    prediction = prediction.upper()

    if prediction:
        if prediction == labels[index]:
            score += 1
        session['quiz_score'] = score

    bip()

    session['quiz_index'] = index + 1
    index += 1

    if index >= len(labels):
        return jsonify({'done': True, 'score': score})

    return jsonify({'done': False, 'label': labels[index], 'index': index + 1})

def getChars():
    labels_path = 'dataset/labels.txt'
    all_chars = []
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 1:
                    label = parts[0].strip().upper()
                    all_chars.append(label)
    return all_chars

if __name__ == '__main__':
    app.run(debug=True)
