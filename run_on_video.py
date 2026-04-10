import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

# ---------------- PATH ----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(CURRENT_DIR, "normal.mp4")

print("==============================================")
print("🚀 SAFE FUSION SYSTEM INITIALIZING...")
print("==============================================")

# ---------------- LOAD YOLO ----------------
print("🔹 YOLOv8 Model Started...")
yolo = YOLO("yolov8n.pt")
print("✅ YOLOv8 Loaded Successfully")

print("🔹 DeepSORT Tracker Started...")
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)
print("✅ DeepSORT Tracker Ready")

print("🔹 Hybrid Collision + Near-Miss Module Started...")
print("==============================================")
print("🎥 Processing Video Stream...\n")

# ---------------- PARAMETERS ----------------
YOLO_CONF = 0.4
IOU_THRESHOLD = 0.15
DIST_THRESHOLD = 40
NEAR_MISS_THRESHOLD = 80
CONFIRM_FRAMES = 2

accident_counter = 0
near_miss_counter = 0

# Store previous centers for motion analysis
previous_centers = {}

# ---------------- HELPER FUNCTIONS ----------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def center_distance(boxA, boxB):
    cx1 = (boxA[0] + boxA[2]) / 2
    cy1 = (boxA[1] + boxA[3]) / 2
    cx2 = (boxB[0] + boxB[2]) / 2
    cy2 = (boxB[1] + boxB[3]) / 2
    return math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 640))
    results = yolo(frame_resized, conf=YOLO_CONF, verbose=False)[0]

    detections = []

    # ---------------- YOLO Detection ----------------
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in [0, 1, 2, 3, 5, 7]:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

            cv2.rectangle(frame_resized,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

    # ---------------- DeepSORT Tracking ----------------
    tracks = tracker.update_tracks(detections, frame=frame_resized)
    boxes = []
    centers = {}

    for t in tracks:
        if not t.is_confirmed():
            continue

        l, t1, r, b = t.to_ltrb()
        boxes.append([l, t1, r, b])

        cx = int((l + r) / 2)
        cy = int((t1 + b) / 2)
        centers[t.track_id] = (cx, cy)

        cv2.putText(frame_resized,
                    f"ID {t.track_id}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1)

    # ---------------- Hybrid Detection Logic ----------------
    collision_detected = False
    near_miss_detected = False

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):

            iou = compute_iou(boxes[i], boxes[j])
            dist = center_distance(boxes[i], boxes[j])

            # COLLISION
            if iou > IOU_THRESHOLD or dist < DIST_THRESHOLD:
                collision_detected = True

            # NEAR MISS (close but not overlapping)
            elif dist < NEAR_MISS_THRESHOLD:
                near_miss_detected = True

    label = "NORMAL"
    color = (0, 255, 0)

    # Confirm Collision
    if collision_detected:
        accident_counter += 1
    else:
        accident_counter = 0

    if accident_counter >= CONFIRM_FRAMES:
        label = "⚠ ACCIDENT DETECTED"
        color = (0, 0, 255)

    # Confirm Near Miss
    if near_miss_detected and not collision_detected:
        near_miss_counter += 1
    else:
        near_miss_counter = 0

    if near_miss_counter >= CONFIRM_FRAMES:
        label = "⚠ NEAR MISS DETECTED"
        color = (0, 165, 255)  # Orange

    # ---------------- Display ----------------
    cv2.putText(frame_resized,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    cv2.imshow("SafeFusion - Hybrid Detection", frame_resized)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n==============================================")
print("✅ SafeFusion Execution Completed")
print("==============================================")