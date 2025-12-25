import sys
import os

sys.path.append(os.path.abspath("src"))

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from config import IMG_SIZE, CLASSES

MODEL_PATH = "saved_model.keras"

st.set_page_config(
    page_title="Face Mask Detection",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = list(CLASSES.keys())

st.title("Face Mask Detection System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    h, w, _ = image_np.shape

    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    bbox, cls = model.predict(img[None, ...], verbose=0)

    box = bbox[0]
    cls_id = np.argmax(cls[0])
    label = class_names[cls_id]

    y1, x1, y2, x2 = (box * [h, w, h, w]).astype(int)

    vis = image_np.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        vis,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    st.image(
        vis,
        caption=f"Prediction: {label}",
        use_column_width=True
    )
