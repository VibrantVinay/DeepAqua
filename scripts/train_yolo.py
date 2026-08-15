import os
import torch
from ultralytics import YOLO

def main():
    print("=== DeepAqua YOLOv11 Training Pipeline ===")
    
    # 1. Select the base model
    # For edge devices, we start with the 'nano' (n) or 'small' (s) architecture
    # to keep parameter counts low for fast real-time execution.
    model_version = "yolo11n.pt" 
    
    print(f"[*] Loading base model: {model_version}")
    model = YOLO(model_version)

    # 2. Define Training Parameters
    # Adjust batch size depending on your laptop/GPU VRAM (e.g., 8, 16, or 32)
    config = {
        "data": "datasets/marine_biology/data.yaml", # Path to your generated yaml
        "epochs": 100,             # Number of training loops
        "imgsz": 640,              # Standardize input image size
        "batch": 16,               # Batch size
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 4,              # Dataloader workers
        "optimizer": "auto",
        "patience": 20,            # Early stopping if no improvement
        "project": "models/weights",
        "name": "deepaqua_biology_run"
    }

    print(f"[*] Starting training on device: {config['device']}")
    
    # 3. Execute Training
    results = model.train(**config)
    
    print("\n[+] Training Complete!")
    print(f"[+] Best weights saved to: models/weights/{config['name']}/weights/best.pt")

if __name__ == "__main__":
    main()
