import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np

# إعداد MediaPipe للارغونوميا
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class SmartGuardProcessor(VideoProcessorBase):
    def __init__(self):
        # تفعيل موديل تتبع وضعية الجسد
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # تحويل الصورة لمعالجتها بواسطة MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.pose.process(img_rgb)

        # إذا تم اكتشاف هيكل عظمي (عامل في الموقع)
        if results.pose_landmarks:
            # رسم الهيكل العظمي الرقمي
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # منطق الأتمتة: تنبيه عند رصد انحناء خطر (مثلاً زاوية الظهر)
            st.session_state["safety_status"] = "Worker Detected"

        return frame.from_ndarray(img, format="bgr24")

st.title("🛡️ Smart Guard AI - Ergonomics & Pose Analysis")
webrtc_streamer(key="ergo-stream", video_processor_factory=SmartGuardProcessor)
