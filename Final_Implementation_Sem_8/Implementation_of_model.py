import cv2
import numpy as np
import tensorflow as tf

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion classes (must match training order)
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

IMG_SIZE = 96

# -----------------------------
# Load face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Crop face (same preprocessing)
# -----------------------------
def crop_face(frame, face):

    x, y, w, h = face
    h_img, w_img, _ = frame.shape

    top_crop = int(0.18 * h)
    bottom_expand = int(0.08 * h)
    side_expand = int(0.05 * w)

    x1 = max(0, x - side_expand)
    y1 = max(0, y + top_crop)
    x2 = min(w_img, x + w + side_expand)
    y2 = min(h_img, y + h + bottom_expand)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)  # ADD THIS
    face_crop = face_crop / 255.0

    face_crop = np.expand_dims(face_crop, axis=0)

    return face_crop


# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120,120)
    )

    # If no face
    if len(faces) == 0:

        cv2.putText(
            frame,
            "No one is present",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    for (x,y,w,h) in faces:

        face_input = crop_face(frame, (x,y,w,h))

        if face_input is None:
            continue

        prediction = model.predict(face_input, verbose=0)

        emotion_index = np.argmax(prediction)

        emotion = CLASSES[emotion_index]

        confidence = np.max(prediction)

        label = f"{emotion} ({confidence*100:.1f}%)"

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        # Show emotion label
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Real-Time Emotion Detection", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()