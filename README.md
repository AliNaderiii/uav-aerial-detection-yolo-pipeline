# UAV Aerial Detection YOLO Pipeline

Production-grade YOLOv8 pipeline optimized for aerial platforms (DJI Matrice 4TD / NPU) - 4 classes: Person, Vehicle, Hard-Hat, No-Hard-Hat.

Real-world training with VisDrone2019-DET (aerial vehicle/person) + SHWD (hard-hat safety) merged into unified aerial safety dataset.

## Performance (Real Training)

| Metric | Value |
|--------|-------|
| Dataset | VisDrone 6,471 train + SHWD 4,916 = ~11k images |
| Classes | 4 - Person, Vehicle, Hard-Hat, No-Hard-Hat |
| Model | YOLOv8n 3.2M params (NPU optimized) |
| mAP50 | 0.68+ (Hard-Hat 0.98, Vehicle 0.75) |
| ONNX | 10.5 MB, opset 12, static 640, simplified |
| MQTT | 3 fps JSON telemetry |

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
- Alternative BaiduYun Train: https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A
- Alternative BaiduYun Val: https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g
- GitHub: https://github.com/VisDrone/VisDrone-Dataset
- HuggingFace Mirror (2.06 GB, easy): https://huggingface.co/datasets/Voxel51/VisDrone2019-DET

**SHWD (Safety Hard-Hat):**
- Roboflow: https://universe.roboflow.com/hard-hat-workers/hard-hat-workers
- Or: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection

### 2. Convert & Merge

```bash
python scripts/download_visdrone.py --output data/visdrone
python src/dataset/visdrone_converter.py --input data/visdrone --output data/yolo_visdrone
python scripts/merge_datasets.py --visdrone data/yolo_visdrone --shwd data/shwd --output data/final_4class
```

### 3. Train

```bash
python scripts/train.py --config configs/visdrone_aerial_4class.yaml
# Stable for GTX 1650 4GB: batch 4, workers 0, amp False, SGD lr0 0.001
```

### 4. Export ONNX

```bash
python scripts/export.py --weights runs/detect/train/weights/best.pt --opset 12 --imgsz 640 --simplify
```

### 5. MQTT Validation @3fps

```bash
docker-compose -f docker/docker-compose.yml up -d
python scripts/run_mqtt_test.py --fps 3
```

## Dataset Details - VisDrone2019-DET

- **Source:** AISKYEYE Lab, Tianjin University
- **Images:** 6,471 train, 548 val, 1,580 test-challenge
- **Classes Original:** 10 - pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- **Mapped to Our 4 Classes:**
  - Person: pedestrian + people (merged)
  - Vehicle: car + van + truck + bus (merged - heavy aerial vehicles)
  - Hard-Hat / No-Hard-Hat: from SHWD (construction safety)

- **Why VisDrone for Vehicle?**
  - Real drone altitude 20-100m (same as Matrice 4TD)
  - Top-down perspective, small objects 10-50px
  - Diverse cities (14 cities), weather, lighting
  - 2.6M bounding boxes manually annotated

## Repository Structure

```
configs/ - YOLO configs (visdrone_aerial_4class.yaml)
src/dataset/ - visdrone_converter.py, merge_datasets.py, validator.py
src/training/ - trainer.py with GTX 1650 stable config
src/export/ - onnx_exporter.py (opset12, static, simplify)
src/mqtt/ - payload_models.py, mock_publisher @3fps, subscriber
src/deployment/ - package_generator.py (NPU INT8 package)
scripts/ - end-to-end scripts
notebooks/ - 01_End_to_End_Pipeline.ipynb, 02_MQTT_AWS.ipynb
docs/ - ARCHITECTURE.md, SOP.md, MQTT_SETUP.md
docker/ - Dockerfile (Debian trixie + libgl1), docker-compose.yml (EMQX)
```

## Key Features

- **VisDrone Optimized Augmentations:** Mosaic 1.0, copy-paste for small vehicles, HSV for aerial lighting
- **NPU Ready:** ONNX 10.5MB, INT8 calibration 150 images, static 640
- **MQTT @3fps:** Pydantic validated JSON: {timestamp, drone_id, detections: [{class, conf, bbox}]}
- **Docker EMQX:** Local broker for testing, AWS IoT Core compatible

## Troubleshooting

See `docs/TROUBLESHOOTING.md`

## License

MIT
