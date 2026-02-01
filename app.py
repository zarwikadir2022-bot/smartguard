import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import mediapipe as mp  # المكتبة الأساسية للذكاء الاصطناعي

# --- إعدادات MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# دالة رياضية لحساب الزاوية بين المفاصل (الأتمتة الرياضية)
def calculate_angle(a, b, c):
    a = np.array(a) # الكتف
    b = np.array(b) # الحوض
    c = np.array(c) # الركبة
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

class AutomatedSafetyProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose_tracker.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # استخراج إحداثيات (الكتف، الحوض، الركبة) لحساب سلامة الظهر
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            angle = calculate_angle(shoulder, hip, knee)
            
            # رسم الهيكل العظمي
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # عرض الزاوية وتغيير اللون إذا كان هناك خطر (أقل من 140 درجة عند الانحناء)
            color = (0, 0, 255) if angle < 140 else (0, 255, 0)
            cv2.putText(img, f"Back Angle: {int(angle)} deg", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            if angle < 140:
                cv2.putText(img, "WARNING: POOR POSTURE", (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

st.title("🛡️ Smart Guard AI - Automated Ergonomics")
webrtc_streamer(key="smart-guard", video_processor_factory=AutomatedSafetyProcessor)
