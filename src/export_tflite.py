import os
import tensorflow as tf

from config import MODEL_DIR, TFLITE_DIR

def main():
    os.makedirs(TFLITE_DIR, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    tflite_path = os.path.join(TFLITE_DIR, "face_mask_detector.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("TFLite model saved at:", tflite_path)

if __name__ == "__main__":
    main()
