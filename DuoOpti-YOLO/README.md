# DuoOpti-YOLO: Dual-layer Channel Optimized YOLO for Multi-scale Erosion Detection

This repository contains the implementation of DuoOpti-YOLO, a dual-stream object detection framework that enhances YOLOv8 using BiFPN-HMC and SA-C2f modules for robust erosion detection in composite materials.

## Features
- Multi-scale feature fusion with BiFPN-HMC
- Dual attention strategy: spatial + channel via SA-C2f
- Fully compatible with Ultralytics YOLOv8
- Improved mAP and real-time performance on GFRP erosion datasets

## Quick Start
```bash
pip install -r requirements.txt
python train.py --config configs/config.yaml
```
