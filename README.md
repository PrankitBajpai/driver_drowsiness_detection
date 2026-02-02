#🚗 Driver Drowsiness Detection System

A real-time Driver Drowsiness Detection System built using Computer Vision and MediaPipe Face Mesh.
The system monitors eye closure and yawning patterns through a webcam to detect driver fatigue and triggers an alarm alert to prevent accidents.

#📌 Features

👁️ Eye Aspect Ratio (EAR) based eye-closure detection

😮 Mouth Aspect Ratio (MAR) based yawn detection

📊 Sleepiness Score (0–100%) with active vs sleepy percentage

🔔 Audio Alarm Alert when drowsiness is detected

📈 Moving Average Smoothing for stable predictions

⚡ Real-time performance using webcam feed

#🧠 How It Works

Captures live video feed using OpenCV

Detects facial landmarks using MediaPipe Face Mesh

Computes:

EAR (Eye Aspect Ratio) → detects closed eyes

MAR (Mouth Aspect Ratio) → detects yawning

Maintains frame-based counters to avoid false positives

Calculates a sleepiness score

Triggers an alarm if drowsiness persists beyond a threshold

🛠️ Tech Stack

Python

OpenCV

MediaPipe

NumPy

playsound

Threading

📂 Project Structure
Driver-Drowsiness-Detection/
│
├── main.py                 # Main application file
├── alarm.wav               # Alarm sound file
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

⚙️ Installation & Setup
1️⃣ Clone the Repository
https://github.com/PrankitBajpai/driver_drowsiness_detection.git

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
python main.py


Press ESC to exit the application.

🔊 Alarm Configuration

Place an alarm.wav file in the project root directory.

Alarm triggers only after sustained drowsiness (cooldown enabled).

📈 Thresholds Used
Parameter	Value
EAR Threshold	0.21
MAR Threshold	0.75
Eye Closed Frames	20
Yawn Frames	15
Alarm Cooldown	5 seconds

(These can be tuned based on lighting and camera quality.)

#🧪 Use Cases

🚘 Driver safety systems

🚌 Commercial vehicle monitoring

🧑‍💻 Fatigue detection for long working hours

🎓 Computer vision & AI learning projects

🚀 Future Improvements

Face orientation & head pose detection

Mobile / embedded deployment (Raspberry Pi)

Deep learning–based fatigue classification

Cloud dashboard & alert analytics

👨‍💻 Author

Prankit Bajpai
B.Tech CSE | AI / ML Enthusiast
