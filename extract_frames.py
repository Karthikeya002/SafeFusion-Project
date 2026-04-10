import cv2
import os
# 🔥 Automatically detect SAFE folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

VIDEO_DIR = os.path.join(BASE_DIR, "dataset", "videos")
FRAME_DIR = os.path.join(BASE_DIR, "dataset", "frames")
IMG_SIZE = (640, 640)


def extract_frames_from_video(video_path, save_dir, label):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame_filename = f"{video_name}_{label}_{frame_count:05d}.jpg"
        frame_path = os.path.join(save_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames from {video_name}")

def process_class(class_name):
    video_class_dir = os.path.join(VIDEO_DIR, class_name)
    frame_class_dir = os.path.join(FRAME_DIR, class_name)

    if not os.path.exists(video_class_dir):
        print(f"❌ Folder not found: {video_class_dir}")
        return

    for video_file in os.listdir(video_class_dir):
        if video_file.lower().endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(video_class_dir, video_file)
            extract_frames_from_video(video_path, frame_class_dir, class_name)

if __name__ == "__main__":
    print("🚀 Starting frame extraction...")
    process_class("accident")
    process_class("normal")
    print("✅ Frame extraction completed successfully!")
