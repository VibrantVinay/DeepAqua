import cv2
import json
import logging
from models.enhancement import ImageEnhancer
from models.detection import ThreatDetector
from utils.alert_system import AlertBroadcaster
from utils.stream_handler import BufferlessVideoCapture
from utils.knowledge_engine import EdgeKnowledgeEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Initializing DeepAqua Vision Pipeline...")
    
    # Initialize all modules
    # Make sure to point the ThreatDetector to the "best.pt" output from your training script
    enhancer = ImageEnhancer(model_path='models/weights/funie_gan.pth')
    detector = ThreatDetector(model_path='models/weights/deepaqua_biology_run/weights/best.pt')
    alerter = AlertBroadcaster(endpoint_url="http://localhost:8080/alert")
    knowledge_engine = EdgeKnowledgeEngine(model_name="llama3") # Requires local Ollama running
    
    # Initialize bufferless video stream (0 for webcam, or path to test video)
    cap = BufferlessVideoCapture(source=0)
    
    if not cap.isOpened():
        logging.error("Failed to open video stream.")
        return

    logging.info("Pipeline active. Press 'q' to terminate.")

    while True:
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            continue

        # Stage 1: Image Enhancement (FUnIE-GAN)
        enhanced_frame = enhancer.process(raw_frame)

        # Stage 2: Object & Threat Detection (YOLOv11)
        detections, annotated_frame = detector.predict(enhanced_frame)

        # Stage 3: Autonomous Knowledge Generation & Alerting
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Only trigger deep-dive for confident detections
            if confidence > 0.75:
                # Retrieve local LLM Dossier
                dossier = knowledge_engine.get_entity_dossier(class_name)
                
                # Log to console
                print(f"\n--- DEEPAQUA INTELLIGENCE: {class_name.upper()} ---")
                print(json.dumps(dossier, indent=2))
                
                # Broadcast the alert payload (Visual Evidence + LLM Data)
                alerter.trigger_alert(
                    threat_type=class_name,
                    confidence=confidence,
                    frame_data=annotated_frame,
                    # Note: You may need to update alert_system.py to accept this 'deep_data' kwarg
                )

        # Stage 4: Visualization
        # Display raw vs processed frames side-by-side
        raw_resized = cv2.resize(raw_frame, (640, 480))
        proc_resized = cv2.resize(annotated_frame, (640, 480))
        combined_view = cv2.hconcat([raw_resized, proc_resized])
        
        cv2.imshow("DeepAqua: Raw (Left) vs Processed (Right)", combined_view)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Pipeline terminated safely.")

if __name__ == "__main__":
    main()
