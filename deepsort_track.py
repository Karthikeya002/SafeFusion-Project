import os
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------- PATH SETUP --------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

DETECTION_DIR = os.path.join(BASE_DIR, "outputs", "yolo_detections")
FRAME_DIR = os.path.join(BASE_DIR, "dataset", "frames")
TRACK_DIR = os.path.join(BASE_DIR, "outputs", "tracks")

os.makedirs(TRACK_DIR, exist_ok=True)

# -------- INIT DEEPSORT --------
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
)

def run_tracking(class_name):
    det_dir = os.path.join(DETECTION_DIR, class_name)
    frame_dir = os.path.join(FRAME_DIR, class_name)
    output_dir = os.path.join(TRACK_DIR, class_name)
    os.makedirs(output_dir, exist_ok=True)
    for file in sorted(os.listdir(det_dir)):
        if not file.endswith(".npy"):
            continue
        det_path = os.path.join(det_dir, file)
        img_name = file.replace(".npy", ".jpg")
        img_path = os.path.join(det_dir, img_name)
        # fallback: load original frame if visualization image not found
        if not os.path.exists(img_path):
            img_path = os.path.join(frame_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        detections = np.load(det_path)
        formatted_dets = []
        for det in detections:
            cls_id, conf, x1, y1, x2, y2 = det
            w = x2 - x1
            h = y2 - y1
            formatted_dets.append(([x1, y1, w, h], conf, cls_id))
        tracks = tracker.update_tracks(formatted_dets, frame=frame)
        track_data = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            track_data.append([track_id, l, t, r, b])
        save_name = file.replace(".npy", "_tracks.npy")
        np.save(os.path.join(output_dir, save_name), np.array(track_data))
    print(f"✅ DeepSORT tracking completed for class: {class_name}")

if __name__ == "__main__":
    print("🚀 Running DeepSORT tracking...")
    run_tracking("accident")
    run_tracking("normal")
    print("✅ DeepSORT tracking finished!")
