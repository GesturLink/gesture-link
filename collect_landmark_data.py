import cv2
import os
import csv
import mediapipe as mp
import time
from datetime import datetime

# === Konfigurasi ===
SAVE_CSV_DIR = 'dataset/dataset_landmark_csv'
SAVE_IMG_DIR = 'dataset/dataset_images'
os.makedirs(SAVE_CSV_DIR, exist_ok=True)
os.makedirs(SAVE_IMG_DIR, exist_ok=True)

# === Input Label & Target Jumlah Data ===
label = input("Masukkan label gesture (misal: A): ").upper()
total_samples = int(input("Jumlah data yang ingin direkam: "))

csv_path = os.path.join(SAVE_CSV_DIR, f"{label}.csv")
img_label_dir = os.path.join(SAVE_IMG_DIR, label)
os.makedirs(img_label_dir, exist_ok=True)

# === Inisialisasi MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === Mulai Kamera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Kamera tidak terdeteksi.")
    exit()

print(f"[INFO] Tunggu deteksi tangan, lalu sistem akan mulai merekam dalam 10 detik...")

hand_detected = False
start_countdown = None
recording_started = False
count = 0

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)

    while count < total_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # === Deteksi awal tangan dan countdown ===
        if result.multi_hand_landmarks:
            if not hand_detected:
                hand_detected = True
                start_countdown = time.time()
            elif not recording_started:
                elapsed = int(time.time() - start_countdown)
                remaining = max(10 - elapsed, 0)
                if remaining == 0:
                    recording_started = True
                else:
                    cv2.putText(frame, f"Mulai dalam {remaining}s", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # === Perekaman data ===
        if recording_started and result.multi_hand_landmarks:
            data = []
            handedness = result.multi_handedness
            landmarks = result.multi_hand_landmarks

            handed_landmarks = sorted(zip(handedness, landmarks), key=lambda x: x[0].classification[0].label)
            if len(handed_landmarks) > 2:
                handed_landmarks = handed_landmarks[:2]

            for _, hand in handed_landmarks:
                for point in hand.landmark:
                    data.extend([point.x, point.y, point.z])

            while len(data) < 126:
                data.extend([0.0] * 3)

            data.append(label)
            writer.writerow(data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            img_filename = f"{label}_{count}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(img_label_dir, img_filename), frame)

            count += 1
            cv2.putText(frame, f"[{count}/{total_samples}] Tersimpan", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        elif not recording_started and hand_detected:
            elapsed = int(time.time() - start_countdown)
            remaining = max(10 - elapsed, 0)
            cv2.putText(frame, f"Mulai dalam {remaining}s", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        elif not hand_detected:
            cv2.putText(frame, "Tunggu deteksi tangan...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        # === Gambar landmark tangan jika ada ===
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # === Tampilkan kamera ===
        cv2.imshow("Perekaman Gesture BISINDO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"[SELESAI] {count} data gesture '{label}' berhasil disimpan.")
