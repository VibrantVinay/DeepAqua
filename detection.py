from ultralytics import YOLO
import cv2
import logging

class ThreatDetector:
    def __init__(self, model_path: str):
        """Initializes the YOLOv11 model for aquatic threat detection."""
        try:
            self.model = YOLO(model_path)
            logging.info(f"Threat detection model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load detection model: {e}")
            raise

    def predict(self, frame):
        """Runs inference on the enhanced frame."""
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        detections = []
        annotated_frame = results[0].plot() # YOLO's built-in bounding box plotting
        
        # Parse results for the alerting system
        for box in results[0].boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': self.model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist() # [x1, y1, x2, y2]
            }
            detections.append(detection)
            
        return detections, annotated_frame
