# main.py
import os
import cv2
import numpy as np
import torch
import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.configs import model_configs
from PIL import Image
from io import BytesIO
import base64

# === Setup ===
app = Flask(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
DRIVE_FILE_ID = "1wcL4ynZ4-2MYe-udV2VolKKAtZVmSh0L"  # Google Drive ID

# === Ensure checkpoint exists ===
os.makedirs("checkpoints", exist_ok=True)
if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading DepthAnything checkpoint...")
    URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
    r = requests.get(URL)
    with open(CHECKPOINT_PATH, 'wb') as f:
        f.write(r.content)
    print("Checkpoint downloaded.")

# === Load DepthAnythingV2 ===
depth_model = DepthAnythingV2(model_configs["vits"])
depth_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
depth_model.to(DEVICE).eval()

# === Ensure YOLO model exists ===
YOLO_MODEL_PATH = "yolov8s.pt"
if not os.path.exists(YOLO_MODEL_PATH):
    print("Downloading YOLOv8s model...")
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    r = requests.get(yolo_url)
    with open(YOLO_MODEL_PATH, 'wb') as f:
        f.write(r.content)
    print("YOLOv8s model downloaded.")

# === Load YOLOv8 ===
yolo_model = YOLO(YOLO_MODEL_PATH)

# === Inference Route ===
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Image file not found"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    frame = np.array(image)

    # Depth estimation preprocessing
    resized = cv2.resize(frame, (518, 518))
    rgb_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    with torch.no_grad():
        depth_map = depth_model(rgb_tensor)[0][0].cpu().numpy()
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # YOLOv8 object detection
    results = yolo_model(frame)[0]
    boxes = results.boxes.data.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    names = results.names

    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls_id = map(int, box[:4]) + [box[4], int(class_ids[i])]
        label = names[cls_id]
        depth_crop = depth_map[y1:y2, x1:x2]
        if depth_crop.size == 0:
            continue
        avg_depth = float(np.mean(depth_crop))
        detections.append({
            "label": label,
            "confidence": round(float(conf), 2),
            "distance": round(avg_depth, 2)
        })

    return jsonify({"detections": detections})

# === Start Flask app ===
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

