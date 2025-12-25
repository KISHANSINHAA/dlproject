import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    bbox_output = Dense(4, activation="sigmoid", name="bbox")(x)
    class_output = Dense(NUM_CLASSES, activation="softmax", name="class")(x)

    model = Model(
        inputs=base_model.input,
        outputs=[bbox_output, class_output]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "bbox": "mse",
            "class": "categorical_crossentropy"
        },
        metrics={
            "class": "accuracy"
        }
    )

    return model
