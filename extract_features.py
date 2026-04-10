import os
import numpy as np
from collections import defaultdict
import math

# -------- PATH SETUP --------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

TRACK_DIR = os.path.join(BASE_DIR, "outputs", "tracks")
FEATURE_DIR = os.path.join(BASE_DIR, "outputs", "features")

os.makedirs(FEATURE_DIR, exist_ok=True)

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extract_features(class_name):
    input_dir = os.path.join(TRACK_DIR, class_name)
    output_dir = os.path.join(FEATURE_DIR, class_name)
    os.makedirs(output_dir, exist_ok=True)
    prev_positions = defaultdict(list)
    for file in sorted(os.listdir(input_dir)):
        if not file.endswith("_tracks.npy"):
            continue
        tracks = np.load(os.path.join(input_dir, file), allow_pickle=True)
        num_objects = len(tracks)
        velocities = []
        centers = []
        for t in tracks:
            track_id = int(t[0])
            x1 = float(t[1])
            y1 = float(t[2])
            x2 = float(t[3])
            y2 = float(t[4])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append((cx, cy))
            if len(prev_positions[track_id]) > 0:
                px, py = prev_positions[track_id][-1]
                velocity = euclidean((cx, cy), (px, py))
                velocities.append(velocity)
            prev_positions[track_id].append((cx, cy))
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = np.max(velocities) if velocities else 0
        # minimum distance between objects (collision risk)
        min_distance = 9999
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = euclidean(centers[i], centers[j])
                min_distance = min(min_distance, d)

        if min_distance == 9999:
            min_distance = 0

        feature_vector = np.array([
            num_objects,
            avg_velocity,
            max_velocity,
            min_distance
        ])

        save_name = file.replace("_tracks.npy", "_features.npy")
        np.save(os.path.join(output_dir, save_name), feature_vector)

    print(f"✅ Feature extraction completed for class: {class_name}")

if __name__ == "__main__":
    print("🚀 Extracting motion features...")
    extract_features("accident")
    extract_features("normal")
    print("✅ Feature extraction finished!")
