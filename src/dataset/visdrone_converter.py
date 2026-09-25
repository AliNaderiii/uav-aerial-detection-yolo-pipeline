"""
VisDrone2019-DET to YOLO converter - Production grade
Handles 10 classes -> 4 classes mapping for aerial NPU deployment

Original VisDrone classes (0-9):
0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van, 5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor

Mapped to 4-class aerial safety:
0: Person (pedestrian + people)
1: Vehicle (car + van + truck + bus) - heavy aerial vehicles
2: Hard-Hat (from SHWD, not in VisDrone)
3: No-Hard-Hat (from SHWD, not in VisDrone)

For this converter, we produce Person + Vehicle only from VisDrone.
Hard-Hat classes are added later via merge_datasets.py
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Tuple
import cv2
from tqdm import tqdm
import argparse

# VisDrone -> Our 4-class mapping (only Person and Vehicle from VisDrone)
VISDRONE_TO_4CLASS = {
    0: 0,  # pedestrian -> Person
    1: 0,  # people -> Person
    2: None,  # bicycle -> ignore (or map to Vehicle if needed)
    3: 1,  # car -> Vehicle
    4: 1,  # van -> Vehicle
    5: 1,  # truck -> Vehicle
    6: None,  # tricycle -> ignore
    7: None,  # awning-tricycle -> ignore
    8: 1,  # bus -> Vehicle
    9: None,  # motor -> ignore (optional: map to Vehicle)
}

# Optional: include lightweight vehicles as Vehicle
VISDRONE_TO_4CLASS_INCLUSIVE = {
    0: 0,  # pedestrian -> Person
    1: 0,  # people -> Person
    2: 1,  # bicycle -> Vehicle (light)
    3: 1,  # car -> Vehicle
    4: 1,  # van -> Vehicle
    5: 1,  # truck -> Vehicle
    6: 1,  # tricycle -> Vehicle
    7: 1,  # awning-tricycle -> Vehicle
    8: 1,  # bus -> Vehicle
    9: 1,  # motor -> Vehicle
}

CLASS_NAMES = ["Person", "Vehicle", "Hard-Hat", "No-Hard-Hat"]


def parse_visdrone_annotation(ann_path: Path, img_width: int, img_height: int, inclusive: bool = False) -> list:
    """
    Parse VisDrone annotation file
    Format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    """
    mapping = VISDRONE_TO_4CLASS_INCLUSIVE if inclusive else VISDRONE_TO_4CLASS
    yolo_lines = []

    if not ann_path.exists():
        return yolo_lines

    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = map(int, parts[:4])
                score = int(parts[4]) if parts[4] != '' else 1
                category = int(parts[5]) - 1  # VisDrone category 1-10 -> 0-9

                if score == 0:  # ignore regions
                    continue

                if category not in mapping or mapping[category] is None:
                    continue

                new_class = mapping[category]

                # Convert to YOLO normalized format: class x_center y_center width height
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                # Filter tiny boxes (< 5px) and invalid
                if w < 5 or h < 5:
                    continue
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    continue

                yolo_lines.append(f"{new_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            except Exception as e:
                continue

    return yolo_lines


def convert_visdrone_to_yolo(
    visdrone_root: Path,
    output_root: Path,
    inclusive_vehicles: bool = False,
    copy_images: bool = True
):
    """
    Convert VisDrone2019-DET structure to YOLO format

    Expected input structure (from official download):
    VisDrone2019-DET-train/
        images/
        annotations/
    VisDrone2019-DET-val/
        images/
        annotations/

    Output YOLO structure:
    output_root/
        images/
            train/
            val/
        labels/
            train/
            val/
        train.txt
        val.txt
        data.yaml
    """
    visdrone_root = Path(visdrone_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Find train/val folders (support multiple naming)
    train_img_dir = None
    train_ann_dir = None
    val_img_dir = None
    val_ann_dir = None

    candidates = list(visdrone_root.rglob("images"))
    for img_dir in candidates:
        parent = img_dir.parent
        ann_dir = parent / "annotations"
        if "train" in str(parent).lower() or "train" in str(img_dir).lower():
            if (parent / "images").exists():
                train_img_dir = parent / "images"
                train_ann_dir = parent / "annotations"
        if "val" in str(parent).lower():
            val_img_dir = parent / "images"
            val_ann_dir = parent / "annotations"

    # Fallback: check direct structure
    if train_img_dir is None:
        for p in visdrone_root.iterdir():
            if p.is_dir() and "train" in p.name.lower():
                if (p / "images").exists():
                    train_img_dir = p / "images"
                    train_ann_dir = p / "annotations"
                elif (visdrone_root / "VisDrone2019-DET-train" / "images").exists():
                    train_img_dir = visdrone_root / "VisDrone2019-DET-train" / "images"
                    train_ann_dir = visdrone_root / "VisDrone2019-DET-train" / "annotations"

    if val_img_dir is None:
        if (visdrone_root / "VisDrone2019-DET-val" / "images").exists():
            val_img_dir = visdrone_root / "VisDrone2019-DET-val" / "images"
            val_ann_dir = visdrone_root / "VisDrone2019-DET-val" / "annotations"

    if train_img_dir is None or not train_img_dir.exists():
        # Try flat structure: visdrone_root/images + annotations
        if (visdrone_root / "images").exists() and (visdrone_root / "annotations").exists():
            train_img_dir = visdrone_root / "images"
            train_ann_dir = visdrone_root / "annotations"
            print(f"Using flat structure: {train_img_dir}")
        else:
            raise FileNotFoundError(f"Could not find VisDrone train images in {visdrone_root}. "
                                    f"Expected VisDrone2019-DET-train/images/")

    print(f"Train images: {train_img_dir}")
    print(f"Train annotations: {train_ann_dir}")
    print(f"Val images: {val_img_dir}")
    print(f"Val annotations: {val_ann_dir}")

    def process_split(img_dir: Path, ann_dir: Path, split: str):
        if not img_dir or not img_dir.exists():
            print(f"Skipping {split}: not found")
            return 0

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"Processing {split}: {len(images)} images")

        count = 0
        for img_path in tqdm(images, desc=f"Converting {split}"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                ann_path = ann_dir / f"{img_path.stem}.txt"
                yolo_lines = parse_visdrone_annotation(ann_path, w, h, inclusive=inclusive_vehicles)

                # Copy image
                dst_img = output_root / "images" / split / img_path.name
                if copy_images:
                    shutil.copy2(img_path, dst_img)

                # Write label (even if empty - YOLO needs file for negative handling)
                dst_label = output_root / "labels" / split / f"{img_path.stem}.txt"
                with open(dst_label, 'w') as f:
                    f.write("\n".join(yolo_lines))

                count += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Write split file list
        split_file = output_root / f"{split}.txt"
        with open(split_file, 'w') as f:
            for img_path in (output_root / "images" / split).glob("*.jpg"):
                f.write(str(img_path.resolve()) + "\n")

        return count

    train_count = process_split(train_img_dir, train_ann_dir, "train")
    val_count = process_split(val_img_dir, val_ann_dir, "val") if val_img_dir else 0

    # Create data.yaml for YOLO training
    yaml_content = f"""# VisDrone + SHWD 4-class aerial safety dataset
path: {output_root.resolve()}
train: {output_root.resolve()}/train.txt
val: {output_root.resolve()}/val.txt
test:

nc: 4
names: {CLASS_NAMES}

# Notes:
# - VisDrone provides Person and Vehicle only
# - Hard-Hat classes added via merge_datasets.py with SHWD
# - Train: {train_count} images
# - Val: {val_count} images
"""

    with open(output_root / "data.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"\nConversion complete:")
    print(f"  Train: {train_count} images -> {output_root}/images/train")
    print(f"  Val: {val_count} images -> {output_root}/images/val")
    print(f"  YAML: {output_root}/data.yaml")

    return train_count, val_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VisDrone2019-DET to YOLO 4-class")
    parser.add_argument("--input", type=str, required=True, help="VisDrone root folder (contains VisDrone2019-DET-train/)")
    parser.add_argument("--output", type=str, required=True, help="Output YOLO folder")
    parser.add_argument("--inclusive", action="store_true", help="Include bicycle/motor/tricycle as Vehicle")
    parser.add_argument("--no-copy", action="store_true", help="Don't copy images, only labels")
    args = parser.parse_args()

    convert_visdrone_to_yolo(
        visdrone_root=Path(args.input),
        output_root=Path(args.output),
        inclusive_vehicles=args.inclusive,
        copy_images=not args.no_copy
    )
