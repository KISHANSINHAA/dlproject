import os
import tensorflow as tf

from config import BATCH_SIZE, EPOCHS
from dataset import create_splits, get_dataset
from model import build_model

def main():
    os.makedirs("saved_model", exist_ok=True)

    if not os.path.exists("data/splits/train.txt"):
        create_splits()

    train_ds = get_dataset("train.txt", BATCH_SIZE)
    val_ds = get_dataset("val.txt", BATCH_SIZE)

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.3
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save("saved_model.keras")
    model.export("saved_model")

    print("\nModel saved successfully:")
    print("✔ saved_model.keras (for evaluation & visualization)")
    print("✔ saved_model/       (for deployment & TFLite)")

if __name__ == "__main__":
    main()
