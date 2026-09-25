# VisDrone2019-DET - Exact Download Links

## Official Source
- GitHub: https://github.com/VisDrone/VisDrone-Dataset
- Original Challenge: http://www.aiskyeye.com/ (registration required for original)
- New Platform: https://www.visdrone.net/

## Task 1: Object Detection in Images (Our Use Case)

### Trainset - 6,471 images, 1.44 GB
- **Google Drive (Direct):** https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing
- **BaiduYun:** https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A

### Valset - 548 images, 0.07 GB
- **Google Drive (Direct):** https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing
- **BaiduYun:** https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g

### Testset-dev - 0.28 GB, GT available (for papers)
- Google Drive: https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V
- BaiduYun: https://pan.baidu.com/s/1RdRfSWV-1IFK7aWljLU_LQ

### Testset-challenge - 0.28 GB (challenge evaluation, GT not available)
- Google Drive: https://drive.google.com/file/d/1KN8R3oioOvSXH492GEVk-Hx74nWHAcXT/view?usp=sharing
- BaiduYun: https://pan.baidu.com/s/1lvEkCgy1WWK4B7TLki4yBQ

## Easiest Mirrors (Recommended for Quick Start)

### 1. HuggingFace (2.06 GB, FiftyOne format)
- Link: https://huggingface.co/datasets/Voxel51/VisDrone2019-DET
- Install: `pip install fiftyone`
- Code:
```python
import fiftyone.utils.huggingface as fouh
dataset = fouh.load_from_hub("Voxel51/VisDrone2019-DET")
```

### 2. Roboflow Universe (Already YOLO format, 10 classes)
- Link: https://universe.roboflow.com/uogolanrewaju/visdrone2019-det
- 8,626 images, YOLOv8 ready
- Classes: car, truck, bus, bicycle, people, pedestrian, van, motor, tricycle, awning-tricycle

### 3. Figshare (1.81 GB)
- Link: https://figshare.com/articles/dataset/VisDrone_-2019/28683947

## Download via Script

### Option A: Print Links Only
```bash
python scripts/download_visdrone.py --method links
```

### Option B: HuggingFace Auto
```bash
pip install fiftyone
python scripts/download_visdrone.py --method huggingface --output data/visdrone_raw
```

### Option C: Google Drive via gdown
```bash
pip install gdown
python scripts/download_visdrone.py --method gdown --output data/visdrone_raw
# Then unzip
unzip data/visdrone_raw/VisDrone2019-DET-train.zip -d data/visdrone_raw/
unzip data/visdrone_raw/VisDrone2019-DET-val.zip -d data/visdrone_raw/
```

## After Download

```bash
# Expected structure:
data/visdrone_raw/
  VisDrone2019-DET-train/
    images/ (6,471 jpg)
    annotations/ (6,471 txt)
  VisDrone2019-DET-val/
    images/ (548 jpg)
    annotations/ (548 txt)

# Convert to YOLO 4-class
python src/dataset/visdrone_converter.py --input data/visdrone_raw --output data/yolo_visdrone

# Merge with SHWD for Hard-Hat classes
python scripts/merge_datasets.py --visdrone data/yolo_visdrone --shwd data/shwd_yolo --output data/final_4class

# Train
python scripts/train.py --config configs/visdrone_aerial_4class.yaml
```

## Dataset Stats

- Source: AISKYEYE Lab, Tianjin University
- Paper: VisDrone-DET2019 Challenge Results (https://arxiv.org/abs/1904.00179)
- Total Bounding Boxes: 2.6M+
- Altitude: 20-100m (real drone)
- Cities: 14 cities in China
- Weather: sunny, cloudy, rainy, night
- Original Classes 10:
  0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van, 5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor

## Mapped to Our 4 Classes

- Person: pedestrian (0) + people (1)
- Vehicle: car (3) + van (4) + truck (5) + bus (8) [heavy vehicles]
- Optional Inclusive: + bicycle, tricycle, awning-tricycle, motor as Vehicle
- Hard-Hat: from SHWD (2)
- No-Hard-Hat: from SHWD (3)

## Why VisDrone for Vehicle Class?

- Real drone perspective (top-down, 20-100m) same as Matrice 4TD
- Small objects 10-50px, dense scenes
- Diverse illumination (day/night), weather
- Perfect for NPU optimization testing (small object detection challenge)
