import os
import cv2
import numpy as np
import torch
import gdown
from flask import Flask, request, jsonify
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.configs import model_configs
from depth_anything_v2.transforms import Resize, NormalizeImage, PrepareForNet
from PIL import Image
import torchvision.transforms as T
import traceback

app = Flask(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_tensor_type(torch.FloatTensor)  # Safe default

# === Paths ===
CHECKPOINT_PATH = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
YOLO_MODEL_PATH = "yolov8s.pt"
# DRIVE_FILE_ID = "1dMaBF9osgUJ30tIEEwiB_cqhlqm0v_J6" #base
DRIVE_FILE_ID = "1wcL4ynZ4-2MYe-udV2VolKKAtZVmSh0L" #small

os.makedirs("checkpoints", exist_ok=True)

# === Download model files if missing ===
if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading DepthAnything checkpoint...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", CHECKPOINT_PATH, quiet=False)

if not os.path.exists(YOLO_MODEL_PATH):
    print("Downloading YOLOv8s model...")
    import requests
    r = requests.get("https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt")
    with open(YOLO_MODEL_PATH, "wb") as f:
        f.write(r.content)

# === Load models ===
print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
depth_model = DepthAnythingV2(**model_configs["vits"])
depth_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
depth_model.to(DEVICE).eval()
print("Models loaded successfully.")

# === Define transform manually ===
depth_transform = T.Compose([
    Resize(width=560, height=448),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet()
])

@app.route('/walkalong', methods=['POST'])
def walkalong():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # === Prepare image ===
        image = Image.open(request.files['image']).convert("RGB")
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # === Run YOLO ===
        results = yolo_model(image_cv2)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()

        # === Run DepthAnything ===
        image_np = np.array(image_cv2).astype(np.float32) / 255.0
        transformed = depth_transform({"image": image_np})
        input_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_model(input_tensor)[0].cpu().numpy()

        # === Map detections to depth ===
        height, width = depth.shape
        response = []

        for box, cls in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))
            distance = float(depth[cy, cx])
            response.append({
                "label": yolo_model.names[int(cls)],
                "distance": round(distance, 2)
            })

        return jsonify({"obstacles": response})

    except Exception as e:
        print("Error in /walkalong:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "🦯 WalkAlong API is up."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
