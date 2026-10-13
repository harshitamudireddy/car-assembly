"""
Automotive Sealant Bead Classifier Training using Ultralytics YOLOv8.
Automatically detects GPU (CUDA), unpacks dataset if needed,
trains the model, and exports 'best.pt' directly to the project root.
"""

import os
import shutil
import zipfile
import torch
from ultralytics import YOLO


def train_bead_classifier(base_dir=None, epochs=25, imgsz=416, batch=16):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(base_dir, 'dataset')
    zip_path = os.path.join(base_dir, 'dataset.zip')

    # Unzip dataset if needed
    if not os.path.exists(dataset_path) and os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found at {dataset_path}. Run generate_dataset.py first.")

    # Detect hardware
    device = 0 if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n========================================================")
    print(f"Hardware Detected: {device_name} (device={device})")
    print(f"Dataset: {dataset_path}")
    print(f"Target: 4-class Bead Quality Classifier (PASS / MISSING / GAP / MISPLACED)")
    print(f"========================================================\n")

    model = YOLO('yolov8n-cls.pt')

    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=os.path.join(base_dir, 'runs'),
        name='car_bead_classifier',
        exist_ok=True,
        verbose=True,
        workers=2 if torch.cuda.is_available() else 0
    )

    # Copy best.pt to root directory for instant plug-and-play
    runs_best = os.path.join(base_dir, 'runs', 'car_bead_classifier', 'weights', 'best.pt')
    root_best = os.path.join(base_dir, 'best.pt')

    if os.path.exists(runs_best):
        shutil.copy2(runs_best, root_best)
        print(f"\n" + "=" * 60)
        print(f"[SUCCESS] Model trained successfully!")
        print(f"Saved to: {root_best}")
        print(f"Send 'best.pt' back to Harsh to place directly in the car-assembly folder.")
        print(f"=" * 60 + "\n")
    else:
        print(f"\nTraining completed. Weights saved in runs/car_bead_classifier.")

    return results


if __name__ == '__main__':
    train_bead_classifier()
