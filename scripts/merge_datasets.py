"""
Wrapper for merging VisDrone + SHWD into 4-class aerial dataset
"""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset.merge_datasets import merge_yolo_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge VisDrone + SHWD -> 4-class")
    parser.add_argument("--visdrone", type=str, required=True, help="VisDrone YOLO folder")
    parser.add_argument("--shwd", type=str, required=True, help="SHWD YOLO folder")
    parser.add_argument("--output", type=str, required=True, help="Output merged folder")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    merge_yolo_datasets(
        visdrone_yolo=Path(args.visdrone),
        shwd_yolo=Path(args.shwd),
        output=Path(args.output),
        val_ratio=args.val_ratio
    )
