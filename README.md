# Navigation and Object Detection API (YOLOv8 + DepthAnythingV2)

This Flask-based API detects objects and estimates their distance using YOLOv8 and DepthAnythingV2. It is designed for real-time assistive navigation applications, such as helping visually impaired users avoid obstacles.

---

## 🔧 Features

- Object detection using pretrained YOLOv8
- Monocular depth estimation using DepthAnythingV2
- ROI-based filtering to focus on obstacles in front
- Auto-download of depth checkpoint from Google Drive
- JSON response suitable for mobile app integration
- Processes one frame per second for mobile performance

---

## 🛠️ Tech Stack

- Python 3.10+
- Flask
- YOLOv8 (`ultralytics`)
- DepthAnythingV2
- Torch, OpenCV, NumPy

---

## 📦 Setup Instructions

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server locally:
   ```bash
   python main.py
   ```

---

## ☁️ Deployment

This app can be deployed to [Render](https://render.com) using the included `render.yaml`. The checkpoint is auto-downloaded from Google Drive if not found.

---

## 📱 Android Integration

The API endpoint `/detect` accepts a POST request with an image and returns JSON:

```json
[
  {
    "label": "person",
    "distance": 1.23,
    "obstacle": true
  },
  ...
]
```

---

## 🔗 References

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)
