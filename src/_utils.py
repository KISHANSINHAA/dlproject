import cv2
import numpy as np
import tensorflow as tf
from config import IMG_SIZE


def preprocess_image(image_path):
    """Preprocess image for model input"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    
    # Resize image to model input size
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    return image_normalized, (original_h, original_w)


def letterbox_resize(image, target_size):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size, target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Resize image
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h))
    
    # Create letterboxed image
    letterboxed_img = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # Place resized image in center
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    letterboxed_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
    
    return letterboxed_img, (start_x, start_y, scale)


def draw_bbox(image, bbox, label, confidence, color=(0, 255, 0)):
    """Draw bounding box on image"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label_text = f"{label}: {confidence:.2f}"
    cv2.putText(image, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image


def compute_precision_recall(tp, fp, fn):
    """Compute precision and recall from true positives, false positives, false negatives"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


def create_class_color_map():
    """Create color mapping for different classes"""
    return {
        'with_mask': (0, 255, 0),      # Green
        'without_mask': (0, 0, 255),   # Red
        'mask_weared_incorrect': (0, 255, 255)  # Yellow
    }