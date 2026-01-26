"""
Real-time Face Emotion Detection
✔ Matches high_performance_model
✔ IMG_SIZE = 64
✔ RGB input
✔ Normalization = /255.0
✔ Haar Cascade face detection
✔ Shows 'No one is present' if no face
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ==========================
# CONFIG (MATCH TRAINING)
# ==========================
IMG_SIZE = 64
MODEL_PATH = "emotion_model_7.h5"

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

SMOOTHING_FRAMES = 5     # set 0 to disable smoothing
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully")
print("[INFO] Model input shape:", model.input_shape)

# ==========================
# LOAD HAAR CASCADE
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Haar Cascade failed to load")

# ==========================
# PREPROCESS FACE (SAME AS TRAINING)
# ==========================
def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_rgb = face_rgb.astype("float32") / 255.0
    face_rgb = np.expand_dims(face_rgb, axis=0)  # (1, 64, 64, 3)
    return face_rgb

# ==========================
# SMOOTHING BUFFER
# ==========================
smooth_buffer = deque(maxlen=SMOOTHING_FRAMES) if SMOOTHING_FRAMES > 0 else None

# ==========================
# START CAMERA
# ==========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("[INFO] Press 'q' to quit")

# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        cv2.putText(
            frame,
            "No one is present",
            (30, 50),
            FONT,
            1.0,
            (0, 0, 255),
            2
        )
    else:
        for (x, y, w, h) in faces:
            pad = int(0.15 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            inp = preprocess_face(face_crop)
            preds = model.predict(inp, verbose=0)[0]

            if smooth_buffer is not None:
                smooth_buffer.append(preds)
                preds = np.mean(np.stack(smooth_buffer), axis=0)

            idx = int(np.argmax(preds))
            conf = float(preds[idx])

            label = f"{CLASS_NAMES[idx]} ({conf*100:.2f}%)"

            # Color by emotion type
            color = (0, 255, 0) if CLASS_NAMES[idx] == "happy" else (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 5),
                FONT,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
print("[INFO] Program exited cleanly")
