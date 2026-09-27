# UAV Aerial Detection YOLO Pipeline

Production-grade YOLOv8 pipeline optimized for aerial platforms (DJI Matrice 4TD / NPU) - VisDrone2019-DET aerial vehicle/person detection with NPU-ready export and MQTT @3fps.

**Real training:** GTX 1650 4GB, 80 epochs 8.4h, 6,471 train / 548 val, mAP50 0.495 Vehicle 0.641

## 📸 Demo - 100% REAL (No Fake)

### Real Inference from best.pt 5.6MB - 18-58 detections

| Real 01 - 18 det Vehicle 0.97 Person 0.82 | Real 02 - 13 det |
|---|---|
| ![Real 01](demo/real_01_0000026_02500_d_0000029_conf0.25.jpg) | ![Real 02](demo/real_02_0000276_02601_d_0000520_conf0.25.jpg) |

| Real 03 - 58 det parking | Real 04 - 28 det |
|---|---|
| ![Real 03](demo/real_03_0000280_01801_d_0000621_conf0.25.jpg) | ![Real 04](demo/real_04_0000242_02762_d_0000010_conf0.25.jpg) |

| Real 05 - 24 det |
|---|
| ![Real 05](demo/real_05_0000242_00627_d_0000003_conf0.25.jpg) |

### Real Training Plots from runs/detect/train/ (YOLO generated)

| Confusion Matrix Real | Results Real 80ep |
|---|---|
| ![Confusion Real](demo/confusion_matrix_real.png) | ![Results Real](demo/results_real.png) |

| F1 Curve Real | PR Curve Real |
|---|---|
| ![F1 Real](demo/F1_curve_real.png) | ![PR Real](demo/PR_curve_real.png) |

**Verified:** mAP50 0.495 Vehicle 0.641 Person 0.349 - 548 val 37k instances
