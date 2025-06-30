import cv2
import numpy as np
import torch
import pyttsx3
import time
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Initialize Depth Anything V2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}
encoder = 'vits'
depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

# Load YOLOv8 model
yolo_model = YOLO("yolov8s.pt")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = {}
cooldown_sec = 3  # seconds between alerts for same object class

# Open webcam
cap = cv2.VideoCapture(3)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to calculate object distances
def get_object_distance(detections, depth_map):
    results = []
    for detection in detections:
        class_name, conf, x1, y1, x2, y2 = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        object_depth = depth_map[y1:y2, x1:x2]
        if object_depth.size == 0:
            continue
        avg_depth = np.mean(object_depth)
        results.append((class_name, conf, avg_depth, (x1, y1, x2, y2)))
    return results

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize frame for depth model
    input_size = 518
    resized_frame = cv2.resize(frame, (input_size, input_size))

    # Get depth map
    depth = depth_anything.infer_image(resized_frame, input_size)
    depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    # Object detection
    yolo_results = yolo_model(frame)
    detections = yolo_results[0].boxes.data.cpu().numpy()

    formatted_detections = [
        (yolo_model.names[int(d[5])], d[4], d[0], d[1], d[2], d[3])
        for d in detections
    ]

    object_distances = get_object_distance(formatted_detections, depth_resized)

    # Annotate and speak alerts
    for obj in object_distances:
        class_name, conf, distance, (x1, y1, x2, y2) = obj

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {distance:.2f} m"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Speak if object is close
        if distance < 1.5:
            now = time.time()
            if class_name not in last_spoken or now - last_spoken[class_name] > cooldown_sec:
                speak = f"{class_name} ahead at {distance:.1f} meters"
                print("SPEAK:", speak)
                engine.say(speak)
                engine.runAndWait()
                last_spoken[class_name] = now

    # Show only annotated camera frame
    cv2.imshow("Walk Along Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()