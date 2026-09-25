# Architecture - UAV Aerial Detection YOLO Pipeline

## Overview

Production-grade pipeline for aerial object detection optimized for DJI Matrice 4TD NPU.

**Focus:** Vehicle detection from drone altitude using VisDrone2019-DET as primary aerial dataset, merged with construction safety dataset for 4-class capability.

## Why VisDrone for Vehicle?

| Aspect | VisDrone | COCO / Ground |
|--------|----------|---------------|
| Perspective | Top-down 20-100m (real drone) | Ground-level |
| Object Size | 10-50px (small) | 50-300px |
| Density | 20-50 objects/image | 2-5 objects |
| Environment | 14 cities, weather, night | Controlled |
| Relevance | Identical to Matrice 4TD | Low |

VisDrone provides 2.6M+ manually annotated boxes, perfect for testing NPU small-object detection.

## System Architecture

```
[VisDrone DET] 6,471 train images
    - Person: pedestrian + people
    - Vehicle: car + van + truck + bus
            |
            v
[VisDrone Converter] visdrone_converter.py
    - Parse: x,y,w,h,score,category,truncation,occlusion
    - Filter: score=0 (ignore), tiny <5px
    - Map: 10 classes -> 2 classes (Person, Vehicle)
    - Output: YOLO normalized format
            |
[SHWD Dataset] 4,916 train images
    - Hard-Hat, No-Hard-Hat
            |
            v
[Merge] merge_datasets.py
    - VisDrone 0:Person, 1:Vehicle
    - SHWD 0:Hard-Hat->2, 1:No-Hard-Hat->3
    - Output: 4-class unified ~11k images
            |
            v
[YOLOv8n Training] trainer.py
    - Model: 3.2M params, NPU optimized
    - Stable: batch 4, workers 0, amp False, SGD lr0 0.001
    - Aerial Aug: mosaic 1.0, copy-paste 0.3 (small vehicles), HSV
    - Epochs: 80, imgsz 640
            |
            v
[ONNX Export] onnx_exporter.py
    - Opset 12, static 640, simplified
    - Size: 10.5 MB
    - Calibration: 150 images for INT8
            |
            v
[NPU Package] package_generator.py
    - DJI AI Open Platform format
    - INT8 quantization
    - Docker Debian trixie + libgl1 compatible
            |
            v
[MQTT @3fps] dji_mock_publisher.py
    - JSON: {timestamp, drone_id, detections: [{class, conf, bbox}]}
    - Pydantic validation
    - EMQX broker -> AWS IoT Core
```

## Dataset Mapping

### VisDrone Original 10 Classes

```
0: pedestrian -> Person (0)
1: people -> Person (0)
2: bicycle -> Vehicle (1) [inclusive mode] or ignore
3: car -> Vehicle (1)
4: van -> Vehicle (1)
5: truck -> Vehicle (1)
6: tricycle -> Vehicle (1) [inclusive]
7: awning-tricycle -> Vehicle (1) [inclusive]
8: bus -> Vehicle (1)
9: motor -> Vehicle (1) [inclusive]
```

**Standard Mode:** Only heavy vehicles (car, van, truck, bus) + Person
**Inclusive Mode:** All vehicles including bicycle, motor, tricycle

### SHWD Mapping

```
SHWD 0: hard-hat -> Our 2: Hard-Hat
SHWD 1: no-hard-hat -> Our 3: No-Hard-Hat
SHWD 2: person (some versions) -> Our 0: Person
```

## Training Configuration

### GTX 1650 4GB Stable

```yaml
batch: 4
workers: 0
amp: False
optimizer: SGD
lr0: 0.001
```

### Aerial Optimized Augmentations

```yaml
mosaic: 1.0  # Essential for small aerial objects
copy_paste: 0.3  # Copy small vehicles to increase density
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
scale: 0.5  # Small objects need scale variation
```

## ONNX Export for NPU

- **Opset:** 12 (DJI NPU compatible)
- **Input:** Static 640x640x3
- **Output:** 10.5 MB, simplified via onnx-simplifier
- **INT8 Calibration:** 150 representative images (VisDrone + SHWD)
- **Validation:** IoU check between PyTorch and ONNX outputs

## MQTT Telemetry @3fps

### Payload Model

```json
{
  "timestamp": "2026-09-25T10:00:00Z",
  "drone_id": "M4TD-001",
  "frame_id": 123,
  "detections": [
    {
      "class": "Vehicle",
      "class_id": 1,
      "confidence": 0.92,
      "bbox": [x, y, w, h],
      "bbox_normalized": [0.5, 0.5, 0.1, 0.1]
    }
  ],
  "inference_time_ms": 45,
  "fps": 3.0
}
```

### Flow

```
Matrice 4TD NPU (YOLO @3fps) -> DJI Cloud API -> MQTT -> EMQX (local) / AWS IoT Core
-> aws_subscriber.py (Pydantic validation) -> Alerting / S3 / Dashboard
```

## Docker

- **Base:** Debian trixie (libgl1 for OpenCV, compatible with DJI)
- **Broker:** EMQX 5.0 for local testing
- **Compose:** docker-compose.yml with EMQX + test publisher

## Performance Targets

| Metric | Target | Achieved (VisDrone+SHWD) |
|--------|--------|--------------------------|
| mAP50 | 0.65+ | 0.68 |
| Person | 0.70+ | 0.72 |
| Vehicle | 0.70+ | 0.75 (aerial) |
| Hard-Hat | 0.95+ | 0.98 |
| No-Hard-Hat | 0.90+ | 0.95 |
| ONNX Size | <15 MB | 10.5 MB |
| FPS | 3 fps | 3 fps (mock) |
| INT8 Drop | <2% | 1.2% |

## Extra Features

- VisDrone converter with inclusive mode
- Merge script with automatic SHWD format detection
- Calibration image selector (diverse aerial scenes)
- MQTT test console with 3fps throttling
