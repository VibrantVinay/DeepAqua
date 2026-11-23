import time
import json
import cv2
from pathlib import Path


class AlertSystem:
    def __init__(self, alert_dir="runs/alerts"):
        self.alert_dir = Path(alert_dir)
        self.alert_dir.mkdir(parents=True, exist_ok=True)

    def trigger(self, threats, frame):
        for threat in threats:
            timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}"
            alert_id = f"ALERT_{threat['severity']}_{timestamp}"
            print(
                f"\n[!] DEFENSE ALERT: {threat['label']} | "
                f"Severity: {threat['severity']} | Confidence: {threat['confidence']}"
            )
            evidence_path = self.alert_dir / f"{alert_id}.jpg"
            cv2.imwrite(str(evidence_path), frame)
            log_entry = {
                "alert_id": alert_id,
                "threat": threat["label"],
                "severity": threat["severity"],
                "confidence": threat["confidence"],
                "bounding_box": threat["bbox"],
                "evidence_path": str(evidence_path),
            }
            with open(self.alert_dir / "alert_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
