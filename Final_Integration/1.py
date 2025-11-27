"""
Real-time Face Emotion Detection with Bounding Box
- Matches training preprocessing (IMG_SIZE and normalization)
- Uses OpenCV Haar Cascade for face detection
- Loads Keras model emotion_model.h5 (same shape: (IMG_SIZE, IMG_SIZE, 3))
"""

import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque

# ---------------------------
# Configuration (match training)
# ---------------------------
IMG_SIZE = 48               # MUST match training IMG_SIZE
MODEL_PATH = "emotion_model.h5"
SCORE_SMOOTHING = 5        # number of frames to smooth predictions over (use 0 to disable)
CONFIDENCE_THRESHOLD = 0.0  # whether to show only above threshold (0.0 = show all)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CLASS_NAMES = ['Negative', 'Neutral', 'Positive']  # must match training class order

# ---------------------------
# Load model
# ---------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Loaded model from {MODEL_PATH}")
except Exception as e:
    print("ERROR loading model:", e)
    raise

# Ensure model input shape info
in_shape = model.input_shape
print("[INFO] Model input shape:", in_shape)

# ---------------------------
# Face detector (Haar Cascade)
# ---------------------------
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade xml. Check cv2 installation.")

# ---------------------------
# Helper: preprocess face for model
# ---------------------------
def preprocess_face(face_bgr):
    # face_bgr: cropped face from cv2 (BGR)
    # convert BGR -> RGB because model trained on RGB images
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    face_arr = face_resized.astype("float32") / 255.0   # same normalization as training
    face_arr = np.expand_dims(face_arr, axis=0)        # shape (1, H, W, 3)
    return face_arr

# ---------------------------
# Smoothing buffer per face (we'll use single global smoothing because multi-face tracking not implemented)
# ---------------------------
smooth_buffer = deque(maxlen=SCORE_SMOOTHING) if SCORE_SMOOTHING > 0 else None

# ---------------------------
# Start video capture (0 = default webcam). Replace 0 with filename for video file.
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. If running in headless server, use a video file or enable webcam.")

# Optional: set camera resolution (may help speed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] No frame received. Exiting...")
        break

    # convert to gray for face detection (faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMultiScale params tuned for frontal faces; adjust scaleFactor/minNeighbors for your data
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If multiple faces, we will process each; smoothing uses aggregated per-face smoothing is not implemented
    for (x, y, w, h) in faces:
        # Expand bbox slightly to include more context (eyes/mouth)
        pad = int(0.2 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_crop = frame[y1:y2, x1:x2].copy()
        if face_crop.size == 0:
            continue

        # Preprocess and predict
        inp = preprocess_face(face_crop)
        preds = model.predict(inp, verbose=0)[0]   # shape (3,)
        
        # smoothing
        if smooth_buffer is not None:
            smooth_buffer.append(preds)
            avg_preds = np.mean(np.stack(smooth_buffer, axis=0), axis=0)
        else:
            avg_preds = preds

        label_idx = int(np.argmax(avg_preds))
        label_conf = float(avg_preds[label_idx])

        # Optionally skip drawing if confidence below threshold
        if label_conf < CONFIDENCE_THRESHOLD:
            continue

        label_text = f"{CLASS_NAMES[label_idx]}: {label_conf:.2f}"

        # Draw bounding box and label
        box_color = (0, 255, 0) if label_idx == 2 else (0, 165, 255) if label_idx == 1 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Text background
        (text_w, text_h), _ = cv2.getTextSize(label_text, FONT, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), box_color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 5), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Show FPS
    # Simple FPS calculation
    # (for an improved FPS display you can track frame times outside loop)
    cv2.imshow("Emotion Detection (q to quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Exited cleanly.")
