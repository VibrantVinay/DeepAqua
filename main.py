import cv2
import logging
from models.enhancement import ImageEnhancer
from models.detection import ThreatDetector
from utils.alert_system import AlertBroadcaster

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Initializing DeepAqua Vision Pipeline...")
    
    # Initialize models and utilities
    # Note: Ensure weight files are placed in models/weights/
    enhancer = ImageEnhancer(model_path='models/weights/funie_gan.pth')
    detector = ThreatDetector(model_path='models/weights/yolov11_custom.pt')
    alerter = AlertBroadcaster(endpoint_url="http://localhost:8080/alert")
    
    # Initialize video stream (0 for default laptop webcam, or path to video file)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logging.error("Failed to open video stream.")
        return

    logging.info("Pipeline active. Press 'q' to terminate.")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            logging.warning("Dropped frame or end of stream.")
            break

        # Stage 1: Image Enhancement (FUnIE-GAN/Water-Net)
        enhanced_frame = enhancer.process(raw_frame)

        # Stage 2: Threat Detection & Ecological Monitoring (YOLOv11)
        detections, annotated_frame = detector.predict(enhanced_frame)

        # Stage 3: Decision & Alerting logic
        for detection in detections:
            # If a high-confidence threat is detected (e.g., class 'mine' or 'unauthorized_diver')
            if detection['class_name'] in ['mine', 'diver'] and detection['confidence'] > 0.85:
                alerter.trigger_alert(
                    threat_type=detection['class_name'],
                    confidence=detection['confidence'],
                    frame_data=annotated_frame
                )

        # Stage 4: Visualization
        # Display both raw and enhanced/annotated frames side-by-side for demo purposes
        combined_view = cv2.hconcat([cv2.resize(raw_frame, (640, 480)), cv2.resize(annotated_frame, (640, 480))])
        cv2.imshow("DeepAqua: Raw vs Processed", combined_view)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Pipeline terminated safely.")

if __name__ == "__main__":
    main()
