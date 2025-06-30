# main.py
import os
import cv2
import numpy as np
import torch
import requests
import gdown
from flask import Flask, request, jsonify
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.configs import model_configs
from PIL import Image

app = Flask(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_tensor_type(torch.FloatTensor)  # Avoid CUDA allocation if not using GPU

# === Paths ===
CHECKPOINT_PATH = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
YOLO_MODEL_PATH = "yolov8n.pt"  # Switched from yolov8s.pt to yolov8n.pt
DRIVE_FILE_ID = "1wcL4ynZ4-2MYe-udV2VolKKAtZVmSh0L"

# === Ensure checkpoint directory ===
os.makedirs("checkpoints", exist_ok=True)

# === Download checkpoint if missing ===
if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading DepthAnything checkpoint...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, CHECKPOINT_PATH, quiet=False)

# === Download YOLOv8n if missing ===
if not os.path.exists(YOLO_MODEL_PATH):
    print("Downloading YOLOv8n model...")
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
    import requests
    r = requests.get(yolo_url)
    with open(YOLO_MODEL_PATH, "wb") as f:
        f.write(r.content)

# === Lazy Load Models ===
yolo_model = None
depth_model = None


def load_models():
    global yolo_model, depth_model
    if yolo_model is None:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    if depth_model is None:
        depth_model = DepthAnythingV2(**model_configs["vits"])
        depth_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        depth_model.to(DEVICE).eval()


@app.route('/walkalong', methods=['POST'])
def walkalong():
    try:
        load_models()

        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = Image.open(request.files['image']).convert("RGB")
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Object detection
        results = yolo_model(image_cv2)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()

        # Depth estimation
        transform = depth_model.get_transform(image_cv2)
        input_tensor = transform(image_cv2).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            depth = depth_model(input_tensor)[0].cpu().numpy()

        response = []
        for box, cls in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            distance = round(float(depth[center_y, center_x]), 2)
            response.append({
                "label": yolo_model.names[int(cls)],
                "distance": distance
            })

        return jsonify({"obstacles": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "🦯 WalkAlong API is up."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
