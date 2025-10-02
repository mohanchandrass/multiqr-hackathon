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


def decode_qr_codes(image, boxes):
    """
    Given an image and bounding boxes, run OpenCV QRCodeDetector to decode QR codes.
    Returns list of dicts with bbox and decoded value.
    """
    detector = cv2.QRCodeDetector()
    results = []

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        crop = image[ymin:ymax, xmin:xmax]

        if crop.size == 0:
            continue

        value, points, _ = detector.detectAndDecode(crop)
        if value:
            results.append({"bbox": box, "value": value})
        else:
            results.append({"bbox": box, "value": ""})  # Could not decode

    return results


def run_inference(model_path: str, input_dir: str, output_json1: str, output_json2: str,
                  imgsz: int = 640, conf: float = 0.25, padding: int = 10):
    """
    Run YOLOv8-OBB detection and OpenCV decoding.
    Generates two submissions as per hackathon guidelines.
    """
    # Load YOLOv8-OBB model
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

        detection_results.append({
            "image_id": os.path.splitext(img_name)[0],
            "qrs": [{"bbox": box} for box in boxes]
        })

        decoded_qrs = decode_qr_codes(img, boxes)
        decoding_results.append({
            "image_id": os.path.splitext(img_name)[0],
            "qrs": decoded_qrs
        })

    # Save Stage 1 detection results
    with open(output_json1, "w") as f:
        json.dump(detection_results, f, indent=2)

    # Save Stage 2 decoding results
    with open(output_json2, "w") as f:
        json.dump(decoding_results, f, indent=2)

    print(f"Inference complete.\nDetection saved: {output_json1}\nDecoding saved: {output_json2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-OBB + OpenCV QRCodeDetector Inference")
    parser.add_argument("--model", type=str, default="src/models/qr_detection_obb/weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--input", type=str, default="e:/summer internship/QR_Dataset/images/test", help="Folder with input images")
    parser.add_argument("--output1", type=str, default="submission_detection_1.json", help="Output JSON for detection")
    parser.add_argument("--output2", type=str, default="submission_decoding_2.json", help="Output JSON for decoding")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--padding", type=int, default=10, help="Bounding box padding")
    args = parser.parse_args()

    run_inference(args.model, args.input, args.output1, args.output2, args.imgsz, args.conf, args.padding)
