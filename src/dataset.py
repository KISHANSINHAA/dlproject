import os
import cv2
import random
import numpy as np
import tensorflow as tf

from config import IMG_DIR, ANN_DIR, IMG_SIZE, NUM_CLASSES, SPLIT_DIR
from xml_parser import parse_xml

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.astype(np.float32)

def create_splits(train_ratio=0.7, val_ratio=0.2):
    os.makedirs(SPLIT_DIR, exist_ok=True)

    images = [
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]

    random.shuffle(images)

    n = len(images)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    splits = {
        "train.txt": images[:train_end],
        "val.txt": images[train_end:val_end],
        "test.txt": images[val_end:]
    }

    for name, files in splits.items():
        with open(os.path.join(SPLIT_DIR, name), "w") as f:
            for file in files:
                f.write(file + "\n")

def data_generator(split_file):
    with open(split_file) as f:
        image_names = f.read().splitlines()

    for img_name in image_names:
        img_path = os.path.join(IMG_DIR, img_name)
        xml_path = os.path.join(
            ANN_DIR,
            img_name.replace(".jpg", ".xml").replace(".png", ".xml")
        )

        if not os.path.exists(xml_path):
            continue

        image = load_image(img_path)
        boxes, class_ids = parse_xml(xml_path)

        yield (
            image,
            {
                "bbox": np.array(boxes[0], dtype=np.float32),
                "class": tf.keras.utils.to_categorical(
                    class_ids[0], NUM_CLASSES
                )
            }
        )

def get_dataset(split_name, batch_size):
    split_file = os.path.join(SPLIT_DIR, split_name)

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(split_file),
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            {
                "bbox": tf.TensorSpec(shape=(4,), dtype=tf.float32),
                "class": tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
            }
        )
    )

    dataset = dataset.shuffle(256)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
