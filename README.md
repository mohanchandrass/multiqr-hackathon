
# multiqr-hackathon

## Setup & Usage Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/mohanchandrass/multiqr-hackathon.git
cd multiqr-hackathon
```

### 2. Create Python Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Training
My approach for Multi-QR Code Detection involved three main steps:

#### 4.1 Dataset Annotation
- I first wrote a custom annotation script to generate bounding box labels for the QR codes in my dataset.
- These annotations were then imported into Roboflow for manual verification. Missing QR codes and incorrect labels were corrected by hand.
- I then split the dataset into training and validation sets.

#### 4.2 Model Choice
- I trained a **YOLOv8-OBB** model (Oriented Bounding Box) instead of a traditional bounding box model.
- **Reason:** QR codes in real-world scenarios often appear rotated, tilted, or partially obscured. YOLOv8-OBB detects objects in arbitrary orientations, making it ideal for this task.
- YOLOv8-OBB offers state-of-the-art speed and efficiency, making it suitable for real-time deployment.

#### 4.3 Training Process
- The training script (`train.py`) automatically splits the dataset into training and validation sets, generates the required `data.yaml` configuration file, and trains the YOLOv8-OBB model.
- **Training settings:**
  - Model: `yolov8n-obb.pt` (nano model, lightweight and fast)
  - Image size: `640 × 640`
  - Batch size: `16`
  - Epochs: `100`
  - Validation split: `10%`
- The trained model weights are saved under `outputs/qr_detection_obb/`.

**Command to train:**
```bash
python train.py --dataset data/dataset --epochs 100 --imgsz 640 --batch 16 --device 0 --split 0.1
```

### 5. Inference
```bash
python infer.py --input data/demo_images/ --output outputs/submission_detection_1.json
```
- Runs QR detection on images in the given folder.
- Saves results in JSON format.

**Example JSON output:**
```json

<p align="center">
  <img src="https://github.com/mohanchandrass/multiqr-hackathon/blob/main/outputs/annotated/img210.jpg" alt="Detection Demo" width="400"/>
</p


[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]
```

### 6. (Optional) Decoding & Classification
```bash
python decode.py --input data/demo_images/ --output outputs/submission_decoding_2.json
```
- Outputs bounding boxes with decoded QR values and classification types.

**Example JSON output:**
```json
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max], "value": "B12345"},
      {"bbox": [x_min, y_min, x_max, y_max], "value": "MFR56789"}
    ]
  }
]
```
