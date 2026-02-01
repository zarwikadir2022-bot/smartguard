import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np

# --- إعدادات الصفحة الاحترافية ---
st.set_page_config(page_title="Smart Guard AI - Automated Ergonomics", layout="wide")

# دالة رياضية لحساب الزاوية (قلب نظام الأتمتة)
def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- محرك الذكاء الاصطناعي (AI Processor) ---
class AutomatedSafetyProcessor(VideoProcessorBase):
    def __init__(self):
        # استيراد محلي داخل المحرك لتجنب AttributeError في Streamlit Cloud
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_tracker = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose_tracker.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 1. استخراج إحداثيات (الكتف، الحوض، الركبة) للجانب الأيسر
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            # 2. حساب الزاوية آلياً
            angle = calculate_angle(shoulder, hip, knee)
            
            # 3. تحديد حالة السلامة بناءً على الزاوية
            # إذا كانت الزاوية أقل من 140 درجة، فهناك خطر انحناء خاطئ للظهر
            risk_color = (0, 0, 255) if angle < 140 else (0, 255, 0)
            status_text = "DANGER: BAD POSTURE" if angle < 140 else "SAFE POSTURE"

            # 4. رسم الهيكل العظمي والمعلومات
            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=risk_color, thickness=2)
            )
            
            # عرض الإحصائيات على الشاشة
            cv2.rectangle(img, (0, 0), (350, 80), (0, 0, 0), -1)
            cv2.putText(img, f"Back Angle: {int(angle)} deg", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, status_text, (10, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)

        return frame.from_ndarray(img, format="bgr24")

# --- واجهة المستخدم ---
st.title("🛡️ Smart Guard AI - نظام الأرغونوميا المؤتمت")
st.markdown("""
هذا النظام يستخدم الذكاء الاصطناعي لتحليل وضعية جسد العمال آلياً. 
**الأتمتة الحالية:** كشف زاوية الظهر وتحديد مخاطر الإصابة المهنية.
""")

webrtc_streamer(
    key="smart-guard-automated",
    video_processor_factory=AutomatedSafetyProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

st.sidebar.header("📋 تقرير الحوكمة اللحظي")
st.sidebar.info("يتم الآن تحليل 33 نقطة حيوية في الجسد لضمان الامتثال لمعايير الصحة والسلامة.")
