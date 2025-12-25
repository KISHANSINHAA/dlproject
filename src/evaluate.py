import numpy as np
import tensorflow as tf

from config import BATCH_SIZE, MODEL_DIR, CLASSES
from dataset import get_dataset

def compute_iou(box1, box2):
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])

    inter_area = max(0, y2 - y1) * max(0, x2 - x1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def main():
    model = tf.keras.models.load_model(f"{MODEL_DIR}.keras")
    test_ds = get_dataset("test.txt", BATCH_SIZE)

    class_names = list(CLASSES.keys())
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    for images, targets in test_ds:
        pred_boxes, pred_classes = model.predict(images, verbose=0)

        true_boxes = targets["bbox"].numpy()
        true_classes = np.argmax(targets["class"].numpy(), axis=1)

        pred_cls_ids = np.argmax(pred_classes, axis=1)

        for i in range(len(images)):
            iou = compute_iou(true_boxes[i], pred_boxes[i])

            true_cls = class_names[true_classes[i]]
            class_total[true_cls] += 1

            if pred_cls_ids[i] == true_classes[i] and iou >= 0.5:
                class_correct[true_cls] += 1

    print("Evaluation Results (IoU ≥ 0.5):\n")
    for cls in class_names:
        acc = class_correct[cls] / max(1, class_total[cls])
        print(f"{cls}: {acc:.4f}")

if __name__ == "__main__":
    main()
