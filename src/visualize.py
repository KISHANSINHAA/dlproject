import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from config import IMG_SIZE, CLASSES
from dataset import get_dataset

MODEL_PATH = "saved_model.keras"

def main():
    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = get_dataset("test.txt", batch_size=1)
    class_names = list(CLASSES.keys())

    count = 0

    for image, target in test_ds:
        img = image[0].numpy()
        h, w, _ = img.shape

        pred_box, pred_class = model.predict(image, verbose=0)
        box = pred_box[0]
        cls_id = np.argmax(pred_class[0])
        label = class_names[cls_id]

        y1, x1, y2, x2 = (box * [h, w, h, w]).astype(int)

        vis = (img * 255).astype("uint8")
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        plt.figure(figsize=(4, 4))
        plt.imshow(vis)
        plt.axis("off")
        plt.title(f"Prediction: {label}")
        plt.show()

        count += 1
        if count == 10:
            break

if __name__ == "__main__":
    main()
