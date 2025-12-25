import tensorflow as tf
import numpy as np
import cv2
from config import IMG_SIZE, GRID_SIZE, NUM_ANCHORS, CLASSES, MODEL_DIR
import os


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    # box format: [x_min, y_min, x_max, y_max]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def decode_predictions(predictions, confidence_threshold=0.5):
    """Decode YOLO predictions to bounding boxes"""
    boxes = []
    scores = []
    class_ids = []
    
    # predictions shape: [GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 4 + 1 + NUM_CLASSES]
    for grid_y in range(GRID_SIZE):
        for grid_x in range(GRID_SIZE):
            for anchor_idx in range(NUM_ANCHORS):
                # Extract prediction for this grid cell and anchor
                pred = predictions[grid_y, grid_x, anchor_idx]
                
                # Extract bounding box coordinates (relative to grid cell)
                x_center = (pred[0] + grid_x) / GRID_SIZE
                y_center = (pred[1] + grid_y) / GRID_SIZE
                width = pred[2] / GRID_SIZE
                height = pred[3] / GRID_SIZE
                
                # Extract confidence and class probabilities
                confidence = pred[4]
                class_probs = pred[5:]
                
                # Only consider if confidence is above threshold
                if confidence > confidence_threshold:
                    class_id = np.argmax(class_probs)
                    score = confidence * class_probs[class_id]
                    
                    if score > confidence_threshold:
                        boxes.append([
                            max(0, x_center - width/2), 
                            max(0, y_center - height/2), 
                            min(1, x_center + width/2), 
                            min(1, y_center + height/2)
                        ])  # Convert to [xmin, ymin, xmax, ymax] in normalized coordinates
                        scores.append(score)
                        class_ids.append(class_id)
    
    return np.array(boxes), np.array(scores), np.array(class_ids)


def non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5):
    """Apply non-maximum suppression to filter overlapping boxes"""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    # Sort by scores in descending order
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Keep the box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        remaining = indices[1:]
        ious = []
        
        for idx in remaining:
            iou = calculate_iou(boxes[current], boxes[idx])
            ious.append(iou)
        
        # Keep boxes with IoU less than threshold
        ious = np.array(ious)
        indices = remaining[ious < iou_threshold]
    
    return keep


def detect_faces(model, image):
    """Detect faces in an image using the trained model"""
    h, w = image.shape[:2]
    
    # Preprocess image
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Make prediction
    predictions = model.predict(img_batch, verbose=0)
    
    # Decode predictions
    boxes, scores, class_ids = decode_predictions(predictions[0])
    
    # Apply non-maximum suppression
    keep_indices = non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5)
    
    # Convert normalized coordinates back to image coordinates
    detected_faces = []
    class_names = list(CLASSES.keys())
    
    for idx in keep_indices:
        box = boxes[idx]
        score = scores[idx]
        class_id = class_ids[idx]
        
        # Convert to image coordinates
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        
        label = class_names[class_id]
        
        detected_faces.append({
            'bbox': [x1, y1, x2, y2],
            'class': label,
            'confidence': float(score),
            'class_id': int(class_id)
        })
    
    return detected_faces


def visualize_detections(image, detections, save_path=None):
    """Visualize detections on image"""
    vis_image = image.copy()
    class_colors = {
        'with_mask': (0, 255, 0),      # Green
        'without_mask': (0, 0, 255),   # Red
        'mask_weared_incorrect': (0, 255, 255)  # Yellow
    }
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = detection['class']
        confidence = detection['confidence']
        color = class_colors.get(label, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(vis_image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image


def load_model_for_inference():
    """Load the trained model for inference"""
    model_path = os.path.join(MODEL_DIR, "final_model")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "best_model.keras")
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print("Trained model not found. Please train the model first using: python src/train.py")
        return None


if __name__ == "__main__":
    # Example usage
    model = load_model_for_inference()
    if model is not None:
        # Test with a sample image
        # Replace with your actual image path
        test_image_path = "data/raw/images/sample.jpg"  # Update this path
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            if image is not None:
                detections = detect_faces(model, image)
                print(f"Detected {len(detections)} faces:")
                for i, det in enumerate(detections):
                    print(f"  Face {i+1}: {det['class']} with confidence {det['confidence']:.2f}")
                
                # Visualize and save results
                vis_image = visualize_detections(image, detections, "detection_result.jpg")
                print("Detection result saved as detection_result.jpg")
            else:
                print("Could not load test image")
        else:
            print("Test image not found. Please provide a valid image path.")
    else:
        print("Could not load model for inference.")