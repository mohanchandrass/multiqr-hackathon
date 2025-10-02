"""
infer.py - Multi-QR Code Detection Inference Script

Runs inference on a folder of images using the trained YOLOv8-OBB model
and generates submission_detection_1.json in the required format:

[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]
"""

import os
import json
import argparse
import cv2
from ultralytics import YOLO


def run_inference(model_path: str, input_dir: str, output_json: str, imgsz: int = 640, conf: float = 0.25, padding: int = 10):
    """
    Run inference on images and save bounding boxes in JSON format.
    Args:
        model_path (str): Path to trained YOLOv8-OBB weights
        input_dir (str): Directory containing test/demo images
        output_json (str): Output JSON file path
        imgsz (int): Inference image size
        conf (float): Confidence threshold
        padding (int): Extra pixels to expand each bounding box
    """
    # Load trained model
    model = YOLO(model_path)

    # Collect image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    results_json = []

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {img_name}, cannot read image.")
            continue

        height, width = img.shape[:2]

        # Run YOLO inference
        results = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)

        qrs = []
        if results and results[0].obb is not None:
            # Each detected polygon → convert to rectangular bbox with padding
            for poly in results[0].obb.xyxy:  
                coords = poly.tolist()  # polygon [x1, y1, x2, y2, ..., x4, y4]

                xs = coords[0::2]
                ys = coords[1::2]
                xmin, xmax = int(min(xs)), int(max(xs))
                ymin, ymax = int(min(ys)), int(max(ys))

                # Apply padding while keeping inside image boundaries
                xmin = max(0, xmin - padding)
                ymin = max(0, ymin - padding)
                xmax = min(width - 1, xmax + padding)
                ymax = min(height - 1, ymax + padding)

                qrs.append({"bbox": [xmin, ymin, xmax, ymax]})

        results_json.append({
            "image_id": os.path.splitext(img_name)[0],
            "qrs": qrs
        })

    # Save JSON file
    with open(output_json, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Inference complete. Results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-OBB Inference for QR Detection")
    parser.add_argument("--model", type=str, default="src/models/qr_detection_obb/weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--input", type=str, default="E:/summer internship/multiqr-hackathon/src/datasets/QR_Dataset/images/test", help="Folder with input images")
    parser.add_argument("--output", type=str, default="submission_detection_1.json", help="Path to save JSON output")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--padding", type=int, default=10, help="Padding for bounding boxes")
    args = parser.parse_args()

    run_inference(args.model, args.input, args.output, args.imgsz, args.conf, args.padding)
