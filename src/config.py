IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

CLASSES = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

NUM_CLASSES = len(CLASSES)

DATA_DIR = "data/raw"
IMG_DIR = f"{DATA_DIR}/images"
ANN_DIR = f"{DATA_DIR}/annotations"

SPLIT_DIR = "data/splits"

MODEL_DIR = "saved_model"
TFLITE_DIR = "tflite"
