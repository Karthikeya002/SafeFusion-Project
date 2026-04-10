🚦 SafeFusion: Intelligent Accident Detection System
📌 Overview

SafeFusion is a next-generation AI-powered traffic surveillance system that combines YOLOv8, DeepSORT, and Transformer models to detect, track, and predict road accidents in real time.

The system not only identifies accidents but also predicts near-miss events using spatiotemporal analysis, enabling early warning and faster emergency response.

🎯 Key Features
🚗 Real-Time Object Detection using YOLOv8
🔄 Multi-Object Tracking using DeepSORT
🧠 Temporal Analysis & Prediction using Transformer
⚠️ Accident & Near-Miss Detection
📡 Automated Alert System for authorities
⚡ Low Latency Processing (~23.8 ms/frame)
📊 High Accuracy (~92% mAP)
🏗️ System Architecture
Video Input → Preprocessing → YOLOv8 Detection → DeepSORT Tracking → 
Transformer Analysis → Accident Prediction → Alert Generation
🔍 How It Works
1. Data Acquisition & Preprocessing
Captures video from CCTV / dashcams
Applies:
Noise reduction
Normalization
Frame resizing
Data augmentation
2. Object Detection (YOLOv8)
Detects:
Vehicles 🚗
Pedestrians 🚶
Cyclists 🚴
Outputs bounding boxes + confidence scores
3. Object Tracking (DeepSORT)
Assigns unique IDs to objects
Tracks:
Speed (velocity)
Trajectory
Lane behavior
4. Temporal Analysis (Transformer)
Learns interactions over time
Detects:
Sudden stops
Collisions
Abnormal movement patterns
5. Alert Generation
Sends real-time alerts with:
Timestamp
Location
Object details
Snapshot
📊 Performance
Metric	Value
Accuracy	92%
Precision	91.5%
Recall	92.3%
Latency	23.8 ms/frame
FPS	~42 FPS
Comparison
Method	Accuracy	Latency
YOLO Only	75%	10 ms
CNN-LSTM	80%	180 ms
SafeFusion	92%	23.8 ms
🛠️ Tech Stack
Programming Language: Python
Frameworks: PyTorch, OpenCV
Models Used:
YOLOv8
DeepSORT
Transformer (Self-Attention)
Hardware: NVIDIA GPU (recommended)
📁 Project Structure
SafeFusion/
│── dataset/
│── models/
│── outputs/
│── src/
│   ├── detection/
│   ├── tracking/
│   ├── transformer/
│   ├── utils/
│── main.py
│── requirements.txt
│── README.md
▶️ Installation & Setup
1. Clone the Repository
git clone [https://github.com/your-username/SafeFusion.git](https://github.com/Karthikeya002/SafeFusion-Project)
cd SafeFusion
2. Install Dependencies
pip install -r requirements.txt
3. Run the Project
python main.py
📸 Output
Real-time detection with bounding boxes
Tracking IDs for each object
Accident alerts with highlighted regions
Logs for analysis
🌍 Applications
🚦 Smart Cities
🚓 Traffic Monitoring Systems
🚑 Emergency Response Systems
🛣️ Highway Surveillance
📊 Urban Planning & Safety Analysis
⚠️ Limitations
Performance drops in:
Heavy rain / fog 🌧️
Dense traffic
Requires high computational power
Limited multimodal sensor integration
🔮 Future Improvements
Integration with LiDAR & Radar
Better performance in low visibility conditions
Edge deployment optimization
Explainable AI for decision transparency
👨‍💻 Authors
K. Karthikeya
T. Kalaichelvi
Derangula Alekhya
Team SafeFusion

Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology

📄 License

This project is for academic and research purposes.

⭐ Acknowledgement

Inspired by advancements in Computer Vision, YOLO, and Transformer architectures for intelligent transportation systems
