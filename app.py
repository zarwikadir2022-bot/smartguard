import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import av
import numpy as np

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="Smart Guard AI", layout="wide", page_icon="🛡️")

st.title("🛡️ Smart Guard AI - Live Monitor")
st.markdown("---")

# --- 2. القائمة الجانبية ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.info("Adjust settings based on lighting.")

# أشرطة التحكم
detection_conf = st.sidebar.slider("Min Detection Confidence", 0.1, 1.0, 0.5, 0.05)
tracking_conf = st.sidebar.slider("Min Tracking Confidence", 0.1, 1.0, 0.5, 0.05)

draw_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)
flip_video = st.sidebar.checkbox("Flip Video", value=False)

# --- 3. تهيئة MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 4. كلاس المعالجة ---
class PoseProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=detection_conf, min_tracking_confidence=tracking_conf)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if flip_video:
            img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        h, w, c = img_bgr.shape

        if results.pose_landmarks:
            # حالة الرصد (Target Locked)
            if draw_skeleton:
                mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # --- الأسطر المصححة (سطر واحد لكل أمر) ---
            cv2.putText(img_bgr, "TARGET LOCKED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img_bgr, (0,0), (w,h), (0, 255, 0), 5)
            
        else:
            # حالة البحث (Searching)
            # --- الأسطر المصححة (سطر واحد لكل أمر) ---
            cv2.putText(img_bgr, "SEARCHING...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img_bgr, "Move back or adjust lighting", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(img_bgr, (0,0), (w,h), (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- 5. تشغيل الكاميرا ---
webrtc_streamer(
    key=f"stream-{detection_conf}-{tracking_conf}-{flip_video}",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
