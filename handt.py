import cv2
import mediapipe as mp
from playsound import playsound
import threading
import time
from math import dist
import numpy as np

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Indices for eyes and mouth from MediaPipe
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Constants
EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold (adjusted for better accuracy)
MAR_THRESHOLD = 0.75  # Mouth Aspect Ratio threshold (increased to reduce false yawns)
CLOSED_FRAMES = 20  # Number of frames for drowsiness detection
YAWN_FRAMES = 15  # Number of frames for yawning detection (increased)
ALARM_COOLDOWN = 5  # Seconds between alarm triggers
SMOOTHING_WINDOW = 10  # Number of frames for moving average


class DrowsinessDetector:
    def __init__(self):
        self.ear_history = []
        self.mar_history = []
        self.counter = 0
        self.yawn_counter = 0
        self.alarm_on = False
        self.last_alarm_time = 0
        self.status = "Active"
        self.sleepiness_score = 0
        self.eye_closed_frames = 0
        self.yawn_detected_frames = 0

    def update_scores(self, ear, mar):
        # Update history buffers
        self.ear_history.append(ear)
        self.mar_history.append(mar)

        # Keep only the last SMOOTHING_WINDOW values
        if len(self.ear_history) > SMOOTHING_WINDOW:
            self.ear_history.pop(0)
            self.mar_history.pop(0)

        # Calculate moving averages
        avg_ear = np.mean(self.ear_history) if self.ear_history else ear
        avg_mar = np.mean(self.mar_history) if self.mar_history else mar

        # Update counters based on thresholds
        if avg_ear < EAR_THRESHOLD:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = max(0, self.eye_closed_frames - 2)

        if avg_mar > MAR_THRESHOLD:
            self.yawn_detected_frames += 1
        else:
            self.yawn_detected_frames = max(0, self.yawn_detected_frames - 2)

        # Calculate sleepiness score (0-100)
        eye_score = min(100, (self.eye_closed_frames / CLOSED_FRAMES) * 100)
        yawn_score = min(100, (self.yawn_detected_frames / YAWN_FRAMES) * 50)  # Yawns contribute less
        self.sleepiness_score = min(100, eye_score + yawn_score)

        # Update status
        if self.eye_closed_frames >= CLOSED_FRAMES:
            self.status = "DROWSY! WAKE UP!"
            if not self.alarm_on and (time.time() - self.last_alarm_time) > ALARM_COOLDOWN:
                self.trigger_alarm()
        elif self.yawn_detected_frames >= YAWN_FRAMES:
            self.status = "Feeling sleepy? Stay active!"
        elif self.sleepiness_score > 30:
            self.status = "Slightly drowsy"
        else:
            self.status = "Active"

        return avg_ear, avg_mar

    def trigger_alarm(self):
        self.alarm_on = True
        self.last_alarm_time = time.time()
        threading.Thread(target=self.play_alarm).start()

    def play_alarm(self):
        playsound("alarm.wav")
        self.alarm_on = False


def eye_aspect_ratio(eye_landmarks):
    A = dist(eye_landmarks[1], eye_landmarks[5])
    B = dist(eye_landmarks[2], eye_landmarks[4])
    C = dist(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth_landmarks):
    A = dist(mouth_landmarks[0], mouth_landmarks[10])
    B1 = dist(mouth_landmarks[2], mouth_landmarks[9])
    B2 = dist(mouth_landmarks[4], mouth_landmarks[11])
    return (B1 + B2) / (2.0 * A)


def draw_landmarks(frame, points, color):
    for x, y in points:
        cv2.circle(frame, (x, y), 1, color, -1)


def main():
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame")
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = face.landmark

            # Get coordinates
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
            mouth = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LIPS]

            # Calculate metrics
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            mar = mouth_aspect_ratio(mouth)
            avg_ear, avg_mar = detector.update_scores((left_ear + right_ear) / 2, mar)

            # Draw landmarks
            draw_landmarks(frame, left_eye + right_eye, (0, 255, 0))
            draw_landmarks(frame, mouth, (0, 255, 255))

            # Display status and metrics
            active_percent = max(0, 100 - detector.sleepiness_score)
            cv2.putText(frame, f"Status: {detector.status}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Sleepy: {detector.sleepiness_score:.1f}%", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Active: {active_percent:.1f}%", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {avg_mar:.2f}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()