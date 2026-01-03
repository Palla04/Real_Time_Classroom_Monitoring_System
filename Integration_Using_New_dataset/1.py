import cv2
import numpy as np
import tensorflow as tf

# ==============================
# CONFIG (MATCHES TRAINING)
# ==============================
IMG_SIZE = 64
MODEL_PATH = "emotion_model.h5"

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names from model
class_names = model.class_names if hasattr(model, "class_names") else [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
]

print("Loaded classes:", class_names)

# ==============================
# FACE DETECTOR
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# PREPROCESS FUNCTION (SAME AS TRAINING)
# ==============================
def preprocess_face(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0      # 🔥 SAME normalization
    face = np.expand_dims(face, axis=0)
    return face

# ==============================
# WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ==============================
    # NO FACE CASE
    # ==============================
    if len(faces) == 0:
        cv2.putText(
            frame,
            "No one is present",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ==============================
    # FACE FOUND
    # ==============================
    for (x, y, w, h) in faces:
        face_rgb = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)

        face_input = preprocess_face(face_rgb)

        preds = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(preds)
        emotion = class_names[emotion_idx]
        confidence = np.max(preds) * 100

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Label
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
