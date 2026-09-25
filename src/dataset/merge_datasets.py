"""
Merge VisDrone (Person, Vehicle) + SHWD (Hard-Hat, No-Hard-Hat) into unified 4-class aerial safety dataset

Production approach:
- VisDrone: aerial perspective, small objects, real drone altitude 20-100m
- SHWD: ground-level construction safety
- Merged: simulates Matrice 4TD construction site monitoring from air

Output: YOLO format ready for YOLOv8 training
"""

import shutil
import random
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse

CLASS_NAMES = ["Person", "Vehicle", "Hard-Hat", "No-Hard-Hat"]


def merge_yolo_datasets(
    visdrone_yolo: Path,
    shwd_yolo: Path,
    output: Path,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Merge two YOLO datasets into 4-class

    VisDrone YOLO: classes 0=Person, 1=Vehicle (from visdrone_converter.py)
    SHWD YOLO: classes 0=Hard-Hat, 1=No-Hard-Hat -> remap to 2,3

    SHWD mapping:
    SHWD original 0 -> Our 2 (Hard-Hat)
    SHWD original 1 -> Our 3 (No-Hard-Hat) OR SHWD may have Person class too
    """
    random.seed(seed)
    visdrone_yolo = Path(visdrone_yolo)
    shwd_yolo = Path(shwd_yolo)
    output = Path(output)

    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    def collect_images_labels(yolo_root: Path, split: str):
        img_dir = yolo_root / "images" / split
        label_dir = yolo_root / "labels" / split
        if not img_dir.exists():
            # Try flat structure
            img_dir = yolo_root / "images"
            label_dir = yolo_root / "labels"
            if not img_dir.exists():
                return []

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        pairs = []
        for img in images:
            label = label_dir / f"{img.stem}.txt"
            if label.exists():
                pairs.append((img, label))
        return pairs

    # Collect VisDrone
    vis_train = collect_images_labels(visdrone_yolo, "train")
    vis_val = collect_images_labels(visdrone_yolo, "val")
    print(f"VisDrone - Train: {len(vis_train)}, Val: {len(vis_val)}")

    # Collect SHWD
    shwd_train = collect_images_labels(shwd_yolo, "train")
    shwd_val = collect_images_labels(shwd_yolo, "val")
    # If SHWD is not split, collect all and split manually
    if len(shwd_train) == 0 and len(shwd_val) == 0:
        all_shwd = []
        for img_dir in [shwd_yolo / "images", shwd_yolo]:
            if img_dir.exists():
                for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
                    label = shwd_yolo / "labels" / f"{img.stem}.txt"
                    if not label.exists():
                        label = img_dir.parent / "labels" / f"{img.stem}.txt"
                    if label.exists():
                        all_shwd.append((img, label))
        random.shuffle(all_shwd)
        split_idx = int(len(all_shwd) * (1 - val_ratio))
        shwd_train = all_shwd[:split_idx]
        shwd_val = all_shwd[split_idx:]
    print(f"SHWD - Train: {len(shwd_train)}, Val: {len(shwd_val)}")

    def process_and_copy(pairs, src_type: str, split: str):
        """Copy and remap labels"""
        count = 0
        for img_path, label_path in tqdm(pairs, desc=f"Merging {src_type} {split}"):
            try:
                # Copy image
                dst_img = output / "images" / split / f"{src_type}_{img_path.name}"
                shutil.copy2(img_path, dst_img)

                # Read and remap labels
                dst_label = output / "labels" / split / f"{src_type}_{img_path.stem}.txt"
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    rest = " ".join(parts[1:])

                    if src_type == "visdrone":
                        # VisDrone already 0=Person, 1=Vehicle - keep
                        if cls in [0, 1]:
                            new_lines.append(f"{cls} {rest}")
                    elif src_type == "shwd":
                        # SHWD: need to handle multiple formats
                        # Common: 0=Hard-Hat, 1=No-Hard-Hat, sometimes 2=Person
                        # We map:
                        # SHWD 0 (hard-hat) -> 2
                        # SHWD 1 (no hard-hat) -> 3
                        # SHWD 2 (person) -> 0
                        if cls == 0:
                            new_lines.append(f"2 {rest}")
                        elif cls == 1:
                            new_lines.append(f"3 {rest}")
                        elif cls == 2:
                            new_lines.append(f"0 {rest}")
                        else:
                            # Unknown, keep if in range
                            if cls in [0, 1, 2, 3]:
                                new_lines.append(f"{cls} {rest}")

                with open(dst_label, 'w') as f:
                    f.write("\n".join(new_lines))

                count += 1
            except Exception as e:
                print(f"Error {img_path}: {e}")
                continue
        return count

    # Merge
    vis_train_c = process_and_copy(vis_train, "visdrone", "train")
    vis_val_c = process_and_copy(vis_val, "visdrone", "val")
    shwd_train_c = process_and_copy(shwd_train, "shwd", "train")
    shwd_val_c = process_and_copy(shwd_val, "shwd", "val")

    # Create data.yaml
    train_total = vis_train_c + shwd_train_c
    val_total = vis_val_c + shwd_val_c

    # Create train.txt and val.txt
    for split in ["train", "val"]:
        split_file = output / f"{split}.txt"
        with open(split_file, 'w') as f:
            for img in (output / "images" / split).glob("*.*"):
                if img.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    f.write(str(img.resolve()) + "\n")

    yaml_content = f"""# Merged Aerial Safety Dataset - VisDrone + SHWD
# VisDrone: Person, Vehicle (aerial perspective)
# SHWD: Hard-Hat, No-Hard-Hat (safety)
path: {output.resolve()}
train: {output.resolve()}/train.txt
val: {output.resolve()}/val.txt
test:

nc: 4
names: {CLASS_NAMES}

# Stats
# VisDrone Train: {vis_train_c}, Val: {vis_val_c}
# SHWD Train: {shwd_train_c}, Val: {shwd_val_c}
# Total Train: {train_total}, Val: {val_total}
# Aerial optimized for Matrice 4TD NPU @ 640
"""

    with open(output / "data.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"\nMerge complete:")
    print(f"  Train: {train_total} images (VisDrone {vis_train_c} + SHWD {shwd_train_c})")
    print(f"  Val: {val_total} images (VisDrone {vis_val_c} + SHWD {shwd_val_c})")
    print(f"  Output: {output}")
    print(f"  YAML: {output}/data.yaml")

    return train_total, val_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge VisDrone + SHWD into 4-class")
    parser.add_argument("--visdrone", type=str, required=True, help="VisDrone YOLO folder (from visdrone_converter.py)")
    parser.add_argument("--shwd", type=str, required=True, help="SHWD YOLO folder")
    parser.add_argument("--output", type=str, required=True, help="Output merged YOLO folder")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    merge_yolo_datasets(
        visdrone_yolo=Path(args.visdrone),
        shwd_yolo=Path(args.shwd),
        output=Path(args.output),
        val_ratio=args.val_ratio
    )
