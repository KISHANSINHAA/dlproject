import sys
import os

sys.path.append(os.path.abspath("src"))

import cv2
import numpy as np
import tensorflow as tf

from config import IMG_SIZE, CLASSES

MODEL_PATH = "saved_model.keras"

def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = list(CLASSES.keys())

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        bbox, cls = model.predict(img[None, ...], verbose=0)

        box = bbox[0]
        cls_id = np.argmax(cls[0])
        label = class_names[cls_id]

        y1, x1, y2, x2 = (box * [h, w, h, w]).astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
