import os
import cv2
import numpy as np
from ultralytics import YOLO

# -------- PATH SETUP (AUTO) --------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

FRAME_DIR = os.path.join(BASE_DIR, "dataset", "frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "yolo_detections")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- LOAD YOLOv8 PRETRAINED --------
model = YOLO("yolov8n.pt")   # lightweight & fast

# Classes we care about (COCO IDs)
VALID_CLASSES = [0, 1, 2, 3, 5, 7]
# person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7

def run_yolo_on_class(class_name):
    input_dir = os.path.join(FRAME_DIR, class_name)
    output_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(input_dir):
        if not img_file.endswith(".jpg"):
            continue
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        results = model(img, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VALID_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                detections.append([
                    cls_id, conf, x1, y1, x2, y2
                ])
                # draw box (for visualization)
                cv2.rectangle(img, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 2)
        # save detection data
        npy_name = img_file.replace(".jpg", ".npy")
        np.save(os.path.join(output_dir, npy_name), np.array(detections))
        # save visualization image
        cv2.imwrite(os.path.join(output_dir, img_file), img)
    print(f"✅ YOLO detection done for class: {class_name}")

if __name__ == "__main__":
    print("🚀 Running YOLOv8 detection...")
    run_yolo_on_class("accident")
    run_yolo_on_class("normal")
    print("✅ YOLOv8 detection completed!")
