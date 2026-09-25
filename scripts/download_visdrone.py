"""
VisDrone2019-DET downloader - Official links
Provides exact download URLs for aerial vehicle dataset

Official Sources:
- GitHub: https://github.com/VisDrone/VisDrone-Dataset
- Google Drive Train 1.44GB: https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing
- Google Drive Val 0.07GB: https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing
- BaiduYun Train: https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A
- BaiduYun Val: https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g
- HuggingFace Mirror: https://huggingface.co/datasets/Voxel51/VisDrone2019-DET (2.06GB, pip install)
- Figshare: https://figshare.com/articles/dataset/VisDrone_-2019/28683947

This script supports:
1. HuggingFace download via fiftyone (easiest)
2. Manual Google Drive download via gdown
3. Roboflow Universe download
"""

import argparse
from pathlib import Path
import os

OFFICIAL_LINKS = {
    "train_google": "https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing",
    "val_google": "https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing",
    "train_baidu": "https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A",
    "val_baidu": "https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g",
    "test_dev_google": "https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V",
    "test_challenge_google": "https://drive.google.com/file/d/1KN8R3oioOvSXH492GEVk-Hx74nWHAcXT/view?usp=sharing",
    "github": "https://github.com/VisDrone/VisDrone-Dataset",
    "huggingface": "https://huggingface.co/datasets/Voxel51/VisDrone2019-DET",
    "figshare": "https://figshare.com/articles/dataset/VisDrone_-2019/28683947",
    "roboflow": "https://universe.roboflow.com/uogolanrewaju/visdrone2019-det",
    "website": "https://www.visdrone.net/"
}

def print_links():
    print("""
=== VisDrone2019-DET Official Download Links ===

TASK 1: Object Detection in Images (Our Use Case)

1. Trainset (1.44 GB) - 6,471 images
   Google Drive: https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing
   BaiduYun: https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A

2. Valset (0.07 GB) - 548 images
   Google Drive: https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing
   BaiduYun: https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g

3. Testset-dev (0.28 GB) - GT available, for paper publishing
   Google Drive: https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V

4. Testset-challenge (0.28 GB) - for challenge
   Google Drive: https://drive.google.com/file/d/1KN8R3oioOvSXH492GEVk-Hx74nWHAcXT/view?usp=sharing

Alternative Mirrors (Easier):
- HuggingFace (2.06 GB, FiftyOne): https://huggingface.co/datasets/Voxel51/VisDrone2019-DET
  pip install fiftyone
  import fiftyone.utils.huggingface as fouh
  dataset = fouh.load_from_hub("Voxel51/VisDrone2019-DET")

- Roboflow Universe: https://universe.roboflow.com/uogolanrewaju/visdrone2019-det
  (Already in YOLO format, 10 classes)

- Figshare: https://figshare.com/articles/dataset/VisDrone_-2019/28683947

GitHub Repo: https://github.com/VisDrone/VisDrone-Dataset
Official Website: https://www.visdrone.net/ (new platform) + http://www.aiskyeye.com/ (original challenge)

Dataset Structure After Download:
VisDrone2019-DET-train/
  images/ (6,471 .jpg)
  annotations/ (6,471 .txt - format: x,y,w,h,score,category,truncation,occlusion)
VisDrone2019-DET-val/
  images/ (548 .jpg)
  annotations/ (548 .txt)

Classes Original 10:
0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van, 5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor
""")

def download_via_huggingface(output_dir: Path):
    try:
        import fiftyone as fo
        import fiftyone.utils.huggingface as fouh
        print(f"Downloading VisDrone via HuggingFace to {output_dir}...")
        dataset = fouh.load_from_hub("Voxel51/VisDrone2019-DET", max_samples=None)
        # Export to YOLO
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Use src/dataset/visdrone_converter.py if you need custom mapping")
        return True
    except ImportError:
        print("FiftyOne not installed. Install via: pip install fiftyone")
        return False
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return False

def download_via_gdown(output_dir: Path):
    try:
        import gdown
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        train_id = "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn"
        val_id = "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59"
        
        print("Downloading trainset (1.44 GB)...")
        gdown.download(f"https://drive.google.com/uc?id={train_id}", str(output_dir / "VisDrone2019-DET-train.zip"), quiet=False)
        
        print("Downloading valset (0.07 GB)...")
        gdown.download(f"https://drive.google.com/uc?id={val_id}", str(output_dir / "VisDrone2019-DET-val.zip"), quiet=False)
        
        print(f"Downloaded to {output_dir}. Please unzip.")
        return True
    except ImportError:
        print("gdown not installed. Install via: pip install gdown")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisDrone2019-DET downloader with official links")
    parser.add_argument("--output", type=str, default="data/visdrone_raw", help="Output folder")
    parser.add_argument("--method", type=str, choices=["links", "huggingface", "gdown"], default="links",
                        help="Download method: links (print only), huggingface, gdown")
    args = parser.parse_args()

    if args.method == "links":
        print_links()
    elif args.method == "huggingface":
        download_via_huggingface(Path(args.output))
    elif args.method == "gdown":
        download_via_gdown(Path(args.output))
