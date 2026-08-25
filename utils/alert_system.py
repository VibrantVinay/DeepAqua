import requests
import cv2
import base64
import json
import logging
from datetime import datetime

class AlertBroadcaster:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        # Cooldown mechanism to prevent network flooding from the same obje
        self.last_alert_time = None
        self.cooldown_seconds = 5

    def trigger_alert(self, threat_type: str, confidence: float, frame_data):
        """Packages visual evidence and telemetry, then transmits it to the monitoring station."""
        now = datetime.now()
        
        # Check cooldown to prevent spamming APIs
        if self.last_alert_time and (now - self.last_alert_time).total_seconds() < self.cooldown_seconds:
            return
            
        self.last_alert_time = now
        
        # Encode frame to base64 for JSON transmission
        _, buffer = cv2.imencode('.jpg', frame_data)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "timestamp": now.isoformat(),
            "threat_type": threat_type,
            "confidence_score": round(confidence, 3),
            "telemetry": {
                "node_id": "EDGE_NODE_01",
                "status": "CRITICAL"
            },
            "image_evidence": encoded_image
        }

        try:
            # Fire-and-forget network request
            response = requests.post(self.endpoint_url, json=payload, timeout=2)
            if response.status_code == 200:
                logging.info(f"Alert broadcast successful: {threat_type} detected.")
            else:
                logging.warning(f"Alert rejected by server. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Network failure during alert broadcast: {e}")
