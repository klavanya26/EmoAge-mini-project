import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import threading
import time
import mediapipe as mp

# Set Streamlit page configuration
st.set_page_config(page_title="Real-Time Face and Gesture Recognition", layout="wide")

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FPS, FPS)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def detect_face(frame):
    """Detect faces in an image using DeepFace and predict age, gender, and emotion."""
    try:
        small_frame = cv2.resize(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2))
        analysis = DeepFace.analyze(small_frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        faces = []

        for person in analysis:
            if isinstance(person, dict):
                region = person.get('region', {})
                x = int(region.get('x', 0)) * 2
                y = int(region.get('y', 0)) * 2
                w = int(region.get('w', 0)) * 2
                h = int(region.get('h', 0)) * 2

                # Check if the face has a reasonable size
                if w > 50 and h > 50:  # You can adjust this threshold as needed
                    age = person.get('age', 'Unknown')
                    gender = person.get('dominant_gender', 'Unknown')
                    emotion = person.get('dominant_emotion', 'Unknown')

                    # Check for overlapping faces and filter them out
                    overlapping = False
                    for existing_face in faces:
                        ex, ey, ew, eh, *_ = existing_face
                        if (x < ex + ew and x + w > ex) and (y < ey + eh and y + h > ey):
                            overlapping = True
                            break

                    if not overlapping:
                        faces.append((x, y, w, h, age, gender, emotion))

        return faces
    except Exception as e:
        st.error(f"Error in face detection: {e}")
        return []

def detect_gesture(frame, hands):
    """Detect gestures using MediaPipe."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    gesture = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            # Detect "Thumbs Up"
            if thumb_tip.y < index_tip.y and thumb_tip.y < pinky_tip.y:
                gesture = "Thumbs up"
            # Detect "Thumbs Down"
            elif thumb_tip.y > index_tip.y and thumb_tip.y > pinky_tip.y:
                gesture = "Thumbs down"

    return gesture

def main():
    # Create a header container
    with st.container():
        st.markdown("""
            <style>
                .header {
                    text-align: center;
                    color: white;
                    background-color: #333333; /* Neutral dark background */
                    padding: 20px;
                    border-radius: 15px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    margin-bottom: 20px;
                }
                .header h1 {
                    font-size: 3em;
                    margin: 0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                .header p {
                    font-size: 1.2em;
                    margin: 10px 0 0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
            </style>
            <div class="header">
                <h2>Real-Time Face and Gesture Recognition</h2>
            </div>
        """, unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.title("Controls")
    detection_interval = st.sidebar.slider("Face Detection Interval (frames)", min_value=1, max_value=30, value=3)
    
    # Initialize and start video stream
    video_stream = VideoStream().start()

    # Streamlit display placeholders
    stframe = st.empty()
    status_text = st.sidebar.empty()
    gesture_text = st.sidebar.empty()  # Placeholder for gesture information
    face_info_text = st.sidebar.empty()  # Placeholder for face information
    
    face_cache = []  # Cache for face detection results
    frame_count = 0

    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    last_gesture = None
    animation_trigger_time = 0  # To track the last time an animation was triggered
    animation_duration = 3  # Animation duration in seconds
    animation_cooldown = 5  # Cooldown period before the next animation can be triggered

    while True:
        frame = video_stream.read()
        if frame is None:
            st.error("Failed to grab frame.")
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Perform face detection every nth frame
        if frame_count % detection_interval == 0:
            face_cache = detect_face(frame)

        # Detect gestures
        gesture = detect_gesture(frame, hands)

        # Draw rectangles and labels for detected faces
        face_info = ""
        for (x, y, w, h, age, gender, emotion) in face_cache:
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, FRAME_WIDTH), min(y + h, FRAME_HEIGHT)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{gender}, {age}, {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            face_info += f"Gender: {gender}\nAge: {age}\nEmotion: {emotion}\n\n"

        # Update face information in the sidebar
        face_info_text.text(face_info.strip())

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update gesture information in the sidebar
        if gesture:
            gesture_text.text(f"Gesture: {gesture.capitalize()}")

        # Handle gestures and animations with smooth transition and cooldown
        current_time = time.time()
        if gesture != last_gesture and (current_time - animation_trigger_time > animation_cooldown):
            if gesture == "Thumbs up":
                st.balloons()
                animation_trigger_time = current_time
            elif gesture == "Thumbs down":
                st.snow()
                animation_trigger_time = current_time
            last_gesture = gesture

        # Increment frame count
        frame_count += 1

        # Sync with the FPS
        time.sleep(1 / FPS)

if __name__ == "__main__":
    main()