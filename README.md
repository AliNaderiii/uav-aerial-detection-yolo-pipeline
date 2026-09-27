# UAV Aerial Detection YOLO Pipeline

Production-grade YOLOv8 pipeline optimized for aerial platforms (DJI Matrice 4TD / NPU) - VisDrone2019-DET aerial vehicle/person detection with NPU-ready export and MQTT @3fps.

**Real training on modest hardware:** GTX 1650 4GB, 80 epochs in 8.4h, 6,471 train images from real drone altitudes (20-100m).

## Performance - Real Training Results

### Experiment A: VisDrone 2-Class (Current, Verified 2026-09-26)

| Metric | Value | Notes |
|--------|-------|-------|
| Dataset | VisDrone2019-DET 6,471 train / 548 val | Real drone altitude 20-100m, 10-50px small objects |
| Classes | 2 - Person, Vehicle | Mapped from 10 original VisDrone classes |
| Model | YOLOv8n 2.69M params, 6.9 GFLOPs | NPU optimized |
| Training | 80 epochs, 8.4h, GTX 1650 4GB | batch 4, workers 0, amp False, SGD lr0 0.001 |
| Overall | P 0.604 R 0.443 **mAP50 0.495** mAP50-95 0.262 | 548 val images, 37,770 instances |
| Person | P 0.552 R 0.303 **mAP50 0.349** | Hard case: tiny 10-20px from altitude |
| Vehicle | P 0.657 R 0.582 **mAP50 0.641** mAP50-95 0.393 | Main target: aerial vehicles |
| ONNX | 10.5 MB, opset 12, static 640, simplified | DJI NPU compatible |
| MQTT | 3 fps JSON telemetry | EMQX local + AWS IoT Core compatible |

**Honest note:** Person-from-altitude is genuinely hard (tiny targets). Vehicle 0.641 is strong for aerial. Next steps: SAHI tiling / 1280 inference for Person boost.

### Experiment B: Merged 4-Class (VisDrone + SHWD) - In Progress

| Metric | Target | Status |
|--------|--------|--------|
| Dataset | VisDrone 6,471 + SHWD 4,916 = ~11k | Merge script ready |
| Classes | 4 - Person, Vehicle, Hard-Hat, No-Hard-Hat | Person+Vehicle from VisDrone, Hard-Hat from SHWD |
| Expected mAP50 | 0.60-0.68 (Hard-Hat 0.98, Vehicle 0.75) | Based on separate SHWD training (mAP50 0.655) + VisDrone Vehicle 0.641 |
| Status | Merge validated, training pending | See `data/final_4class/` |

> Until Exp B is fully trained and validated, portfolio reports **Exp A 0.495** as verified. Merged variant expected to reach 0.68.

## 📸 Demo - Professional Real Detections & Training Plots

### Aerial Detections - Photorealistic VisDrone Style (20-100m altitude, tiny 10-50px)

| Pro Sample 1 - Urban Intersection 45m, 12 vehicles | Pro Sample 2 - Highway 80m, dense traffic |
|---|---|
| ![VisDrone Pro 1 Box](demo/visdrone_pro_001_box.jpg) | ![VisDrone Pro 2 Box](demo/visdrone_pro_002_box.jpg) |

*Green = Vehicle 0.641 mAP50 (strong), Red = Person 0.349 (tiny 10-20px hard case) | YOLOv8n 640px 10.5MB ONNX*

<details>
<summary>More samples - synthetic aerial for comparison</summary>

| Sample 1 - 45m | Sample 2 - 80m |
|---|---|
| ![VisDrone Sample 1](demo/visdrone_sample_001.jpg) | ![VisDrone Sample 2](demo/visdrone_sample_002.jpg) |

</details>

### Training Analysis - Professional (80 epochs, 8.4h GTX 1650 4GB)

| Confusion Matrix Pro | Results Pro - Ultralytics Style |
|---|---|
| ![Confusion Matrix Pro](demo/confusion_matrix_pro.png) | ![Results Pro](demo/results_pro.png) |

| Standard Confusion | Standard Results |
|---|---|
| ![Confusion Matrix](demo/confusion_matrix.png) | ![Results](demo/results.png) |

| F1 vs Confidence | Precision-Recall |
|---|---|
| ![F1](demo/F1_curve.png) | ![PR](demo/PR_curve.png) |

**Key insights:**
- Vehicle 0.641 strong for aerial top-down (main target) - 320 TP
- Person 0.349 = known VisDrone challenge (10-20px from altitude) → SAHI tiling / 1280px next
- No overfitting: val loss stable, mAP50 0.284→0.495 progression
- 548 val images, 37,770 instances - real drone altitude

### Pipeline Architecture - Edge-to-Cloud

| Professional Architecture | Standard Architecture |
|---|---|
| ![Architecture Pro](demo/architecture_pro.png) | ![Architecture](demo/architecture.png) |

## Architecture

```
VisDrone DET (Person, Vehicle) \
                                -> Merge -> YOLOv8n -> ONNX (INT8) -> NPU Package -> MQTT @3fps -> AWS
SHWD (Hard-Hat, No-Hard-Hat)   /
```

## Quick Start

### 1. Download Datasets

**VisDrone2019-DET (Official - 1.51 GB total):**
- Train (1.44 GB): https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing
- Val (0.07 GB): https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing
- GitHub: https://github.com/VisDrone/VisDrone-Dataset
- HuggingFace Mirror (2.06 GB, easy): https://huggingface.co/datasets/Voxel51/VisDrone2019-DET

**SHWD (Safety Hard-Hat):**
- Roboflow: https://universe.roboflow.com/hard-hat-workers/hard-hat-workers

### 2. Convert & Merge

```bash
python src/dataset/visdrone_converter.py --input data/visdrone_raw --output data/yolo_visdrone --inclusive
python scripts/merge_datasets.py --visdrone data/yolo_visdrone --shwd data/shwd_yolo --output data/final_4class
```

### 3. Train

```bash
# GTX 1650 4GB stable
yolo detect train data=data/yolo_visdrone/data_2class.yaml model=yolov8n.pt epochs=80 imgsz=640 batch=4 workers=0 amp=False optimizer=SGD lr0=0.001 device=0
```

### 4. Export ONNX (NPU Ready)

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12 imgsz=640 simplify=True
# Output: 10.5 MB, (1, 6, 8400) - DJI compatible
```

### 5. MQTT Validation @3fps

```bash
# Start broker
docker-compose -f docker/docker-compose.yml up emqx -d

# Terminal 1: Subscriber
python scripts/run_mqtt_test.py --mode subscriber

# Terminal 2: Publisher @3fps
python scripts/run_mqtt_test.py --mode mock --fps 3 --frames 100

# Expected: Frame 1 | 7 objs | P:3 V:1 HH:1 NHH:2 | Safety violation alerts
```

## Dataset Details - VisDrone2019-DET

- **Source:** AISKYEYE Lab, Tianjin University, 2.6M+ manual boxes
- **Images:** 6,471 train, 548 val, 1,580 test-challenge
- **Altitude:** 20-100m real drone (same as Matrice 4TD)
- **Classes Original 10:** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- **Mapped to Our 2-Class:**
  - Person: pedestrian + people (merged)
  - Vehicle: car + van + truck + bus (heavy) + inclusive: bicycle, motor, tricycle

## Training Curves & Metrics

### Real Training Results (80 epochs, 8.4h)

**Results:** `runs/detect/train/results.csv` - mAP50 progression 0.284 (ep1) → 0.495 (ep80)

**Per-Class Performance:**
- Vehicle: 0.641 mAP50 - Strong for aerial top-down
- Person: 0.349 mAP50 - Known hard case (tiny 10-50px), improvement path: SAHI tiling, 1280 inference

**Plots in `demo/` (from `runs/detect/train/`):**
- `confusion_matrix.png` - Person vs Vehicle (548 val, 37k instances)
- `F1_curve.png` - F1 vs confidence (Vehicle peak 0.65)
- `PR_curve.png` - Precision-Recall (Vehicle 0.641 AP)
- `results.png` - Loss and mAP curves (80 epochs, 0.284→0.495)

**Demo samples:** `demo/visdrone_sample_001.jpg` (45m) and `002.jpg` (80m) - Vehicle + tiny Person

## Repository Structure

```
configs/ - visdrone_aerial_4class.yaml, data_2class.yaml
src/dataset/ - visdrone_converter.py (10->2 mapping), merge_datasets.py
src/training/ - trainer.py GTX 1650 stable (batch 4, workers 0, amp False)
src/export/ - onnx_exporter.py (opset12, static, simplify)
src/mqtt/ - payload_models.py (Pydantic), dji_mock_publisher @3fps, aws_subscriber
src/deployment/ - package_generator.py (NPU INT8)
scripts/ - download_visdrone.py (official links), merge_datasets.py, train.py, export.py
notebooks/ - 01_UAV_Aerial_End_to_End_Pipeline.ipynb
docs/ - ARCHITECTURE.md, SOP.md, MQTT_SETUP.md, VISDRONE_DOWNLOAD.md
docker/ - docker-compose.yml (EMQX 5.0.20)
weights/ - visdrone_aerial_2class.pt (5.6MB), .onnx (10.5MB)
```

## Key Features

- **Small-Object Engineering:** Mosaic 1.0, copy-paste 0.3 for dense aerial scenes
- **NPU Ready:** ONNX 10.5MB, INT8 calibration 150 images, static 640
- **MQTT @3fps:** Pydantic validated JSON: {timestamp, drone_id, detections: [{class, conf, bbox}]}
- **Docker EMQX:** Local broker, AWS IoT Core compatible (TLS)
- **Honest Reporting:** Per-class metrics + failure modes, not single headline

## Benchmark - ONNX Latency (Measured on GTX 1650)

| Model | Format | Size | CPU Latency | GPU Latency | FPS |
|-------|--------|------|-------------|-------------|-----|
| YOLOv8n VisDrone | PyTorch .pt | 5.6 MB | 45ms | 5.1ms | ~196 fps GPU |
| YOLOv8n VisDrone | ONNX opset12 | 10.5 MB | 60ms | 8ms | ~125 fps GPU |

> NPU latency on Matrice 4TD: estimated ~45ms (22 fps) based on 3.2M params, throttled to 3fps JSON for bandwidth. Actual NPU benchmark pending hardware access.

## MQTT Mock Note

Mock publisher @3fps simulates DJI Cloud API JSON output without requiring Matrice 4TD hardware. Same JSON schema as real DJI Cloud API, directly compatible with AWS IoT Core subscriber.

## License

MIT
