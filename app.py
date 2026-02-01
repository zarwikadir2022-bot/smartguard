import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import pandas as pd
import importlib

# --- إعداد الهوية ---
st.set_page_config(page_title="Smart Guard AI", layout="wide")

# --- استيراد ذكي وآمن ---
MP_POSE = None
MP_DRAWING = None

try:
    # محاولة فرض تحميل المكونات
    mp = importlib.import_module('mediapipe')
    if hasattr(mp, 'solutions'):
        MP_POSE = mp.solutions.pose
        MP_DRAWING = mp.solutions.drawing_utils
except Exception as e:
    st.sidebar.error(f"محرك AI في وضع الخمول: {e}")

# --- المحرك البرمجي ---
class IntegratedAIProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose_tracker = None
        if MP_POSE:
            self.pose_tracker = MP_POSE.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.pose_tracker:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose_tracker.process(img_rgb)

            if results.pose_landmarks:
                MP_DRAWING.draw_landmarks(img, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS)
                cv2.putText(img, "AUTOMATED MONITORING: ACTIVE", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- الواجهة المتكاملة ---
st.title("🛡️ Smart Guard AI | مركز الحوكمة")

tab1, tab2 = st.tabs(["📺 الرقابة الحية", "📊 سجل المخاطر"])

with tab1:
    webrtc_streamer(
        key="final-fix",
        video_processor_factory=IntegratedAIProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with tab2:
    st.subheader("📋 سجل المخاطر الرقمي")
    st.info("سيتم تسجيل الانتهاكات الأرغونومية هنا آلياً بمجرد تفعيل الكاميرا.")
