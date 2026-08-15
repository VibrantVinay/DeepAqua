import os
import cv2
import yaml
import shutil
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# ==============================================================================
# CONFIGURATION & TAXONOMY DEFINITION
# ==============================================================================
CLASS_MAPPING = {
    0: "flora_seagrass",
    1: "flora_kelp_giant",
    2: "coral_brain",
    3: "coral_staghorn",
    4: "fauna_crustacean_crab",
    5: "fauna_fish_lionfish",
    6: "fauna_fish_clownfish",
    7: "fauna_mammal_dolphin"
}

BASE_DATASET_DIR = Path("datasets/marine_biology")
RAW_DATA_DIR = Path("data/raw")


def setup_directories(base_dir: Path):
    """Creates the standard YOLO dataset folder hierarchy."""
    for split in ["train", "val", "test"]:
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[+] Directory tree initialized at: {base_dir}")


def bbox_to_yolo(box: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Converts bounding box from [xmin, ymin, xmax, ymax] to normalized YOLO format:
    [x_center, y_center, width, height] in range [0.0, 1.0].
    """
    xmin, ymin, xmax, ymax = box
    
    # Calculate dimensions
    width = xmax - xmin
    height = ymax - ymin
    
    # Calculate centers
    x_center = xmin + (width / 2.0)
    y_center = ymin + (height / 2.0)
    
    # Normalize by image dimensions
    x_center /= img_w
    y_center /= img_h
    width /= img_w
    height /= img_h
    
    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)


def mask_to_yolo_bboxes(mask_path: str, class_id: int) -> List[str]:
    """
    Extracts bounding boxes from binary or semantic segmentation masks (e.g. SUIM dataset)
    and converts them to YOLO format label lines.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    img_h, img_w = mask.shape
    
    # Threshold mask to binary (foreground vs background)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_lines = []
    for contour in contours:
        # Filter out minor noise artifacts
        if cv2.contourArea(contour) < 100:
            continue
            
        xmin, ymin, w, h = cv2.boundingRect(contour)
        xmax = xmin + w
        ymax = ymin + h
        
        xc, yc, nw, nh = bbox_to_yolo((xmin, ymin, xmax, ymax), img_w, img_h)
        yolo_lines.append(f"{class_id} {xc} {yc} {nw} {nh}\n")
        
    return yolo_lines


def write_label_file(label_path: Path, lines: List[str]):
    """Writes label lines to a .txt file."""
    with open(label_path, "w") as f:
        f.writelines(lines)


def generate_yaml(base_dir: Path, class_mapping: Dict[int, str]):
    """Generates the data.yaml file required by YOLO."""
    yaml_path = base_dir / "data.yaml"
    
    data = {
        "path": str(base_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_mapping),
        "names": class_mapping
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
    print(f"[+] YOLO data configuration written to: {yaml_path}")


def split_and_distribute_dataset(
    image_files: List[Path],
    label_files: List[Path],
    base_dir: Path,
    train_ratio=0.8,
    val_ratio=0.15,
    seed=42
):
    """
    Randomly splits paired images and label files into Train, Validation, and Test sets.
    """
    random.seed(seed)
    combined = list(zip(image_files, label_files))
    random.shuffle(combined)
    
    total = len(combined)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    splits = {
        "train": combined[:train_end],
        "val": combined[train_end:val_end],
        "test": combined[val_end:]
    }
    
    for split_name, pairs in splits.items():
        for img_p, lbl_p in pairs:
            # Destination paths
            dest_img = base_dir / "images" / split_name / img_p.name
            dest_lbl = base_dir / "labels" / split_name / lbl_p.name
            
            shutil.copy2(img_p, dest_img)
            shutil.copy2(lbl_p, dest_lbl)
            
        print(f"[+] Allocated {len(pairs)} samples to '{split_name}' split.")


if __name__ == "__main__":
    print("=== Marine Biology Dataset Ingestion & Preprocessing ===")
    setup_directories(BASE_DATASET_DIR)
    generate_yaml(BASE_DATASET_DIR, CLASS_MAPPING)
    print("\nDataset pipeline ready. Place raw images and masks into your staging area.")
