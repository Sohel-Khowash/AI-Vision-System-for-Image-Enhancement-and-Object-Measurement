# Underwater Object Detection and Size Estimation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Jetson%20Orin%20Nano-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

An end-to-end computer vision and sensor integration framework designed for Autonomous Underwater Vehicles (AUVs) and marine robotics. This project combines a single-beam Sonar sensor, camera visual feed, and deep learning (YOLOv8n) with underwater physical refraction modeling to achieve real-time object detection, distance estimation, and true physical size measurement[cite: 1].

To maintain reliable performance in murky or turbid water, this repository integrates multiple state-of-the-art enhancement pipelines including **Hybrid Color Correction & Dehazing**, **Polarization-Based Dehazing**, **FUnIE-GAN**, and **CLAHE**[cite: 1].

---

## 📋 Table of Contents
- [Team & Acknowledgments](#-team--acknowledgments)
- [Mathematical Model](#-mathematical-model)
- [Directory Structure](#-directory-structure)
- [Hardware Setup](#-hardware-setup)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Image & Video Enhancement Methods](#-image--video-enhancement-methods)
- [Experimental Results](#-experimental-results)
- [References](#-references)

---

## 👥 Team & Acknowledgments

* **Institution:** National Institute of Technology (NIT) Silchar[cite: 1]
* **Supervisor:** Prof. Binoy Krishna Roy[cite: 1]
* **Authors:**
  * Sohel Khowash (Roll: 2213071)[cite: 1]
  * Debasish Das (Roll: 2213076)[cite: 1]
  * Zohab Faiz (Roll: 2213142)[cite: 1]
* **Presentation Date:** May 13, 2026[cite: 1]

---

## 📐 Mathematical Model

### 1. Refractive Pinhole Model
Standard camera pinhole geometry is adjusted for underwater light refraction using the refractive index of water ($n \approx 1.33$)[cite: 1]:

$$d_{\text{underwater}} \approx 1.33 \cdot d_{\text{air}}$$

Where $d$ represents the calibrated focal length[cite: 1].

### 2. Physical Size & Distance Calculations
Using similar triangles from the pinhole model[cite: 1]:

$$\frac{W}{D} = \frac{w}{d_{\text{underwater}}}$$

* **Distance Estimation ($D$):** When real-world object dimension ($W$) is known[cite: 1]:
  $$D = \frac{d_{\text{underwater}} \cdot W}{w}$$

* **Physical Size Estimation ($W$):** When distance ($D$) is provided by Sonar[cite: 1]:
  $$W = \frac{w \cdot D}{d_{\text{underwater}}}$$

---

## 📂 Directory Structure

```bash
.
├── config/
│   ├── camera_calibration.yaml   # Camera intrinsic matrices & distortion coeffs
│   └── sonar_config.json         # Baud rate and port configuration for Sonar
├── models/
│   ├── yolov8n_underwater.pt     # Trained YOLOv8n object detection weights
│   └── funie_gan_generator.pth   # Pre-trained FUnIE-GAN weights for video enhancement
├── src/
│   ├── enhancement/
│   │   ├── clahe.py              # CLAHE processing pipeline
│   │   ├── hybrid_dehaze.py      # Hybrid Color Correction & Dehazing
│   │   ├── polarization.py       # Polarization-based scattering restoration
│   │   └── funie_gan.py          # GAN-based real-time enhancement
│   ├── sensors/
│   │   ├── camera_stream.py      # OpenCV feed processor for Explore HD
│   │   └── sonar_ping.py         # Serial interface for Ping Sonar
│   ├── detection.py              # YOLOv8 inference script
│   └── estimation.py             # Size and distance calculation engine
├── main.py                       # Main execution script for hardware
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
