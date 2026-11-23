import cv2
import base64
import os
from groq import Groq


class KnowledgeEngine:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.client = None
        self.vlm_active = False
        if self.groq_api_key:
            try:
                self.client = Groq(api_key=self.groq_api_key)
                self.vlm_active = True
            except Exception:
                self.client = None

        self.priority_threats = {
            0: {"label": "Hostile Submarine", "severity": "CRITICAL"},
            1: {"label": "Hostile AUV/UUV", "severity": "CRITICAL"},
            2: {"label": "Naval Mine", "severity": "HIGH"},
            3: {"label": "Combat Diver", "severity": "HIGH"},
        }

        self.monitored_entities = {
            4: "Civilian ROV", 5: "Fish", 6: "Shark/Ray",
            7: "Marine Mammal", 8: "Coral Reef", 9: "Aquatic Plant",
            10: "Shipwreck", 11: "Marine Debris"
        }

    def evaluate_detections(self, boxes, current_threshold):
        threats_detected = []
        ecology_detected = []
        if boxes is None:
            return threats_detected, ecology_detected

        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            if cls_id in self.priority_threats and confidence >= current_threshold:
                threat_info = self.priority_threats[cls_id].copy()
                threat_info["confidence"] = round(confidence, 3)
                threat_info["bbox"] = xyxy
                threat_info["source"] = "YOLO"
                threats_detected.append(threat_info)
            elif cls_id in self.monitored_entities and confidence >= current_threshold:
                ecology_detected.append({
                    "label": self.monitored_entities[cls_id],
                    "confidence": round(confidence, 3),
                    "bbox": xyxy,
                    "source": "YOLO",
                })
        return threats_detected, ecology_detected

    def run_deep_scan(self, frame, bbox, base_label):
        if self.client is None:
            return "VLM offline: configure GROQ_API_KEY in the server environment."
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return f"Crop failed ({base_label})"
        ok, buffer = cv2.imencode(".jpg", crop)
        if not ok:
            return f"Image encoding failed ({base_label})"
        img_b64 = base64.b64encode(buffer).decode("ascii")
        prompt = (
            f"Identify the most specific supportable class or species for this {base_label}. "
            "If the image is unclear, say uncertain. Keep it to four words."
        )
        try:
            completion = self.client.chat.completions.create(
                model=os.getenv("DEEPAQUA_VLM_MODEL", "qwen/qwen3.8-27b"),
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}],
                temperature=0.1,
                max_tokens=24,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            return f"VLM request failed ({type(exc).__name__})"
