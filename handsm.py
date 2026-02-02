import cv2
import numpy as np
import urllib.request
import mediapipe as mp

# === Load aura images from GitHub URLs ===
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image

# Aura image URLs (from GitHub)
green_aura_0_url = "https://github.com/karan-owalekar/Doctor-Strange/blob/master/Aura_Img/green_aura_0.jpg?raw=true"
green_aura_1_url = "https://github.com/karan-owalekar/Doctor-Strange/blob/master/Aura_Img/green_aura_1.jpg?raw=true"
red_aura_url     = "https://github.com/karan-owalekar/Doctor-Strange/blob/master/Aura_Img/red_aura.jpg?raw=true"

# Load and resize aura images
green_aura_0 = cv2.resize(url_to_image(green_aura_0_url), (300, 300))
green_aura_1 = cv2.resize(url_to_image(green_aura_1_url), (300, 300))
red_aura     = cv2.resize(url_to_image(red_aura_url), (300, 300))

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.6)

# === Video Capture ===
cap = cv2.VideoCapture(0)
rotation_number = 0

# Create named window and set fullscreen flag
window_name = "Magic Spell Mode"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    # === Hand Logic: Aura Display ===
    if hand_results.multi_hand_landmarks:
        for handLms, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'

            cx = int(handLms.landmark[0].x * w)
            cy = int(handLms.landmark[0].y * h)

            x1, x2 = cx - 150, cx + 150
            y1, y2 = cy - 150, cy + 150

            # Clamp ROI coordinates inside frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = frame[y1:y2, x1:x2]
            rotation_number += 1

            # Finger tip and pip indices
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]

            extended_count = 0
            for tip, pip in zip(finger_tips, finger_pips):
                tip_y = handLms.landmark[tip].y
                pip_y = handLms.landmark[pip].y
                extended_count += tip_y < pip_y  # finger extended if tip above pip

            is_fist = extended_count <= 1

            try:
                roi_h, roi_w = roi.shape[:2]
                if is_fist:
                    aura_resized = cv2.resize(red_aura, (roi_w, roi_h))
                    blended = cv2.addWeighted(aura_resized, 1, roi, 1, 0)
                    frame[y1:y2, x1:x2] = blended
                else:
                    aura_img = green_aura_0 if (rotation_number % 3) % 2 == 0 else green_aura_1
                    aura_resized = cv2.resize(aura_img, (roi_w, roi_h))
                    blended = cv2.addWeighted(aura_resized, 1, roi, 1, 0)
                    frame[y1:y2, x1:x2] = blended
            except Exception as e:
                print(f"Blend error: {e}")

            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    else:
        print("No hands detected")

    # === Face Logic: Subtle green glowing eyes ===
    if face_results.multi_face_landmarks:
        for faceLms in face_results.multi_face_landmarks:
            # Eye landmark indices for iris centers approx (use MediaPipe eye landmarks)
            left_eye_landmarks = [33, 133]  # approximate eye corners
            right_eye_landmarks = [362, 263]

            # Get eye centers (midpoint between corners)
            left_x = int((faceLms.landmark[left_eye_landmarks[0]].x + faceLms.landmark[left_eye_landmarks[1]].x) / 2 * w)
            left_y = int((faceLms.landmark[left_eye_landmarks[0]].y + faceLms.landmark[left_eye_landmarks[1]].y) / 2 * h)
            right_x = int((faceLms.landmark[right_eye_landmarks[0]].x + faceLms.landmark[right_eye_landmarks[1]].x) / 2 * w)
            right_y = int((faceLms.landmark[right_eye_landmarks[0]].y + faceLms.landmark[right_eye_landmarks[1]].y) / 2 * h)

            # Draw small solid circle inside eyes for glow core
            cv2.circle(frame, (left_x, left_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (right_x, right_y), 10, (0, 255, 0), -1)

            # Create glow overlay with smaller radius and less blur
            glow = np.zeros_like(frame)
            cv2.circle(glow, (left_x, left_y), 15, (0, 255, 0), -1)
            cv2.circle(glow, (right_x, right_y), 15, (0, 255, 0), -1)

            glow = cv2.GaussianBlur(glow, (15, 15), 10)

            # Blend glow with original frame (lower alpha for subtle effect)
            frame = cv2.addWeighted(frame, 1, glow, 0.3, 0)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
