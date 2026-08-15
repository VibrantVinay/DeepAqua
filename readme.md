<div align="center">

# 🌊 DeepAqua: Real-Time Underwater Image Enhancement & Threat Detection

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A high-throughput, edge-optimized deep learning pipeline for real-time optical restoration, semantic segmentation, and automated anomaly classification in degraded marine environments.**

---

</div>

## 📌 Overview

Raw optical data captured in underwater environments suffers from severe physical degradation, including:
* **Wavelength-dependent light attenuation** causing heavy blue-green color casts.
* **Forward and backward optical scattering** leading to low contrast and blurred imagery.
* **Floating marine particulate matter ("marine snow")** obscuring targets.

These factors cause conventional open-air computer vision models to fail. **DeepAqua** resolves this bottleneck by implementing a unified, purely software-defined, edge-capable execution pipeline that restores visual fidelity via conditional Generative Adversarial Networks (cGANs) before executing high-speed object detection and pixel-level semantic segmentation.

---

## 🏗️ System Architecture

```text
                                [ Raw Underwater Video Stream ]
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │    Bufferless Stream Handler    │
                              │  (Thread-Safe Frame Dropping)   │
                              └────────────────┬────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │   Stage 1: Image Enhancement    │
                              │      FUnIE-GAN / Water-Net      │
                              │   (Color, SSIM & Detail Loss)   │
                              └────────────────┬────────────────┘
                                               │ Enhanced Frame
                                               ▼
                              ┌─────────────────────────────────┐
                              │   Stage 2: Vision & Detection   │
                              │  ├── YOLOv11 (Threat Detection) │
                              │  └── SUIM-Net (Segmentation)    │
                              └────────────────┬────────────────┘
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
         ┌─────────────────────────────┐               ┌─────────────────────────────┐
         │     Telemetry & Alerts      │               │     Live Visualisation      │
         │ (JSON Payload over REST/WS) │               │   (Side-by-Side CV Output)  │
         └─────────────────────────────┘               └─────────────────────────────┘
 Key Features⚡
Zero-Latency Frame Capture: Thread-safe bufferless stream handler prevents frame backlog during compute-intensive deep learning inference.
🎨 Perceptual Enhancement (FUnIE-GAN): Restores global contrast, local styles, and true color profiles using an optimized U-Net generator backbone.
🎯 Precision Threat & Object Detection: Detects high-priority underwater targets (e.g., naval mines, unauthorized divers, marine fauna) with custom-trained YOLOv11 architectures.
🗺️ Semantic Scene Parsing: Multi-class pixel-level segmentation via SUIM-Net for background topology mapping (reefs, wrecks, aquatic plants).
🚨 Automated Network Broadcasting: Generates asynchronous telemetry payloads with base64-encoded visual evidence and transmits them to central command dashboards.📂 Repository StructurePlaintextdeepaqua-vision/
├── models/
│   ├── __init__.py
│   ├── enhancement.py          # FUnIE-GAN Generator network & preprocessing
│   ├── detection.py            # YOLOv11 threat detector wrapper
│   └── weights/                # Directory for model checkpoints (.pt, .pth)
│       └── .gitkeep
├── utils/
│   ├── __init__.py
│   ├── alert_system.py         # Automated telemetry broadcaster & rate limiter
│   └── stream_handler.py       # Threaded, bufferless OpenCV video capture
├── sample_data/                # Sample test videos and images
│   └── degraded_sample.mp4
├── main.py                     # Central inference pipeline orchestrator
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git tracking exclusion rules
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
🚀 Getting StartedPrerequisitesPython 3.9+ installedNVIDIA GPU with CUDA 11.8+ support (Recommended for real-time performance)1. Clone the RepositoryBashgit clone [https://github.com/your-username/deepaqua-vision.git](https://github.com/your-username/deepaqua-vision.git)
cd deepaqua-vision
2. Create and Activate a Virtual EnvironmentBash# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
3. Install DependenciesBashpip install --upgrade pip
pip install -r requirements.txt
4. Setup Model WeightsPlace the pre-trained model weights in the models/weights/ folder:ModelCheckpoint FilenameExpected LocationFUnIE-GANfunie_gan.pthmodels/weights/funie_gan.pthYOLOv11yolov11_custom.ptmodels/weights/yolov11_custom.pt💻 Execution & ConfigurationRunning the Live PipelineBashpython main.py
Custom ConfigurationsYou can adjust input feeds, confidence thresholds, and broadcast endpoints directly in main.py or through command-line parameters:Python# Video Source: Camera index, RTSP link, or video file path
cap = BufferlessVideoCapture(source="sample_data/underwater_feed.mp4")

# Alert Receiver URL
alerter = AlertBroadcaster(endpoint_url="[http://192.168.1.100:8000/api/v1/alerts](http://192.168.1.100:8000/api/v1/alerts)")
📊 Benchmark Datasets & PerformanceThe pipeline is trained and evaluated using standard underwater benchmarks:UIEB Dataset: 950 real-world underwater images evaluated for SSIM, PSNR, and UIQM improvements.SUIM Dataset: 1,500+ pixel-annotated images across 8 marine entity categories.Brackish Dataset: Turbid underwater bounding-box annotations for object tracking.📖 ReferencesCode snippet@article{islam2020fast,
  title={Fast Underwater Image Enhancement for Improved Visual Perception},
  author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={3227--3234},
  year={2020}
}

@article{li2019underwater,
  title={An Underwater Image Enhancement Benchmark Dataset and Beyond},
  author={Li, Chongyi and Guo, Chunle and Ren, Wenqi and Cong, Runmin and Hou, Junhui and Kwong, Sam and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4376--4389},
  year={2019}
}

@inproceedings{islam2020semantic,
  title={Semantic Segmentation of Underwater Imagery: Dataset and Benchmark},
  author={Islam, Md Jahidul and Edge, Chelsey and Xiao, Yuyang and Luo, Peter and Mehtaz, Munshi and Morse, Christopher and Enan, Sadman Sakib and Sattar, Junaed},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1769--1776},
  year={2020}
}
