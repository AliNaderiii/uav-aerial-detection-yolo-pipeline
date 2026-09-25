#!/bin/bash
# Setup environment for UAV aerial detection pipeline

echo "=== UAV Aerial Detection YOLO Pipeline - Environment Setup ==="

# Check Python
python3 --version

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "=== Setup Complete ==="
echo "To activate: source .venv/bin/activate"
echo "Next steps:"
echo "1. Download VisDrone: python scripts/download_visdrone.py --method links"
echo "2. Convert: python src/dataset/visdrone_converter.py --input data/visdrone_raw --output data/yolo_visdrone"
echo "3. Merge: python scripts/merge_datasets.py --visdrone data/yolo_visdrone --shwd data/shwd --output data/final_4class"
echo "4. Train: python scripts/train.py --config configs/visdrone_aerial_4class.yaml"
