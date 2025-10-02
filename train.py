"""
train.py - Multi-QR Code Detection Training Script

This script prepares the dataset (train/val split), 
creates the YOLO data.yaml config, 
and trains a YOLOv8-OBB model for QR code detection.

Hackathon Requirement:
- Must be runnable end-to-end.
- Produces trained model under outputs/qr_detection_obb/.
"""

import os
import shutil
import random
import argparse
import yaml
from ultralytics import YOLO


def prepare_dataset(dataset_dir: str, split_ratio: float = 0.1):
    """
    Prepares dataset by splitting into train/valid sets.
    Args:
        dataset_dir (str): Path to dataset root folder (expects train/images and train/labels).
        split_ratio (float): Fraction of data for validation.
    Returns:
        str: Path to generated data.yaml
    """
    train_dir = os.path.join(dataset_dir, "train")
    train_images = os.path.join(train_dir, "images")
    train_labels = os.path.join(train_dir, "labels")

    valid_dir = os.path.join(dataset_dir, "valid")
    valid_images = os.path.join(valid_dir, "images")
    valid_labels = os.path.join(valid_dir, "labels")

    os.makedirs(valid_images, exist_ok=True)
    os.makedirs(valid_labels, exist_ok=True)

    # Collect images
    images = [f for f in os.listdir(train_images) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)

    num_val = int(len(images) * split_ratio)
    val_images = images[:num_val]

    for img_name in val_images:
        # Move image
        shutil.move(os.path.join(train_images, img_name), os.path.join(valid_images, img_name))

        # Move corresponding label
        label_name = os.path.splitext(img_name)[0] + ".txt"
        shutil.move(os.path.join(train_labels, label_name), os.path.join(valid_labels, label_name))

    # Create data.yaml for YOLO
    data_config = {
        'train': os.path.join(dataset_dir, 'train/images'),
        'val': os.path.join(dataset_dir, 'valid/images'),
        'nc': 1,  # number of classes
        'names': ['QR_code']
    }

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)

    print(f"Dataset prepared. data.yaml saved at {yaml_path}")
    return yaml_path


def train_model(data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 16, device: int = 0):
    """
    Trains YOLOv8-OBB model.
    Args:
        data_yaml (str): Path to YOLO data.yaml
        epochs (int): Training epochs
        imgsz (int): Image size
        batch (int): Batch size
        device (int): Device ID (0=GPU, -1=CPU)
    """
    # Load YOLOv8-OBB nano model (lightweight, fast)
    model = YOLO("yolov8n-obb.pt")

    print("Starting training:")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="outputs",
        name="qr_detection_obb"
    )

    print("Training completed. Model saved in outputs/qr_detection_obb/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-OBB for QR Code Detection")
    parser.add_argument("--dataset", type=str, default="data/dataset", help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="Device (0=GPU, -1=CPU)")
    parser.add_argument("--split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    # Step 1: Prepare dataset
    data_yaml = prepare_dataset(args.dataset, args.split)

    # Step 2: Train model
    train_model(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
