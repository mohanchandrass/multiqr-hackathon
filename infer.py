"""
infer.py - Multi-QR Code Detection + Decoding Inference Script

Generates:
- submission_detection_1.json : bounding boxes only
- submission_decoding_2.json : bounding boxes + decoded QR values
"""

import os
import json
import argparse
import cv2
from ultralytics import YOLO


def decode_qr_codes(image, boxes, padding=20):
    """
    Given an image and bounding boxes, decode QR codes using OpenCV QRCodeDetector.
    Handles tilted/multiple QR codes robustly.
    Returns list of dicts with bbox and decoded value.
    """
    results = []
    height, width = image.shape[:2]
    detector = cv2.QRCodeDetector()

    for box in boxes:
        xmin, ymin, xmax, ymax = box

        # Apply extra padding for robustness
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(width - 1, xmax + padding)
        ymax = min(height - 1, ymax + padding)

        crop = image[ymin:ymax, xmin:xmax]

        if crop.size == 0:
            results.append({"bbox": box, "value": ""})
            continue

        value, points, _ = detector.detectAndDecode(crop)

        if value:
            results.append({"bbox": [xmin, ymin, xmax, ymax], "value": value})
        else:
            results.append({"bbox": [xmin, ymin, xmax, ymax], "value": ""})

    return results


def run_inference(model_path: str, input_dir: str, output_json1: str, output_json2: str,
                  imgsz: int = 640, conf: float = 0.25, padding: int = 20):
    """
    Run YOLOv8-OBB detection and OpenCV decoding.
    Saves only submission_detection_1.json and submission_decoding_2.json.
    """
    model = YOLO(model_path)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    detection_results = []
    decoding_results = []

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {img_name}, cannot read image.")
            continue

        height, width = img.shape[:2]

        # YOLO Detection
        results = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)

        boxes = []
        if results and results[0].obb is not None:
            for poly in results[0].obb.xyxy:
                coords = poly.tolist()
                xs = coords[0::2]
                ys = coords[1::2]
                xmin, xmax = int(min(xs)), int(max(xs))
                ymin, ymax = int(min(ys)), int(max(ys))

                # Apply padding
                xmin = max(0, xmin - padding)
                ymin = max(0, ymin - padding)
                xmax = min(width - 1, xmax + padding)
                ymax = min(height - 1, ymax + padding)

                boxes.append([xmin, ymin, xmax, ymax])

        # Stage 1: Detection-only results
        detection_results.append({
            "image_id": os.path.splitext(img_name)[0],
            "qrs": [{"bbox": box} for box in boxes]
        })

        # Stage 2: Decoding
        decoded_qrs = decode_qr_codes(img, boxes, padding=padding)
        decoding_results.append({
            "image_id": os.path.splitext(img_name)[0],
            "qrs": decoded_qrs
        })

    # Save JSON results
    with open(output_json1, "w") as f:
        json.dump(detection_results, f, indent=2)

    with open(output_json2, "w") as f:
        json.dump(decoding_results, f, indent=2)

    print(f"Inference complete.\nDetection saved: {output_json1}\nDecoding saved: {output_json2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-OBB + OpenCV QRCodeDetector Inference")
    parser.add_argument("--model", type=str, default="src/models/qr_detection_obb/weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--input", type=str, default="data/demo_images", help="Folder with input images")
    parser.add_argument("--output1", type=str, default="outputs/submission_detection_1.json", help="Output JSON for detection")
    parser.add_argument("--output2", type=str, default="outputs/submission_decoding_2.json", help="Output JSON for decoding")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--padding", type=int, default=20, help="Bounding box padding")
    args = parser.parse_args()

    run_inference(args.model, args.input, args.output1, args.output2, args.imgsz, args.conf, args.padding)
