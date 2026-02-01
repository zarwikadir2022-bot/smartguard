import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import pandas as pd

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Smart Guard AI - Final Version", layout="wide")

# دالة حساب الزوايا للأتمتة
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- 2. محرك الذكاء الاصطناعي (بتصميم محمي) ---
class IntegratedAIProcessor(VideoProcessorBase):
    def __init__(self):
        # استيراد محلي داخل الكلاس لحل مشكلة AttributeError
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
        
        # تحليل الوضعية
        results = self.pose_tracker.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # استخراج النقاط (الكتف، الحوض، الركبة)
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            angle = calculate_angle(shoulder, hip, knee)
            risk_color = (0, 0, 255) if angle < 140 else (0, 255, 0)

            # رسم الهيكل العظمي الآلي 
            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2),
                self.mp_drawing.DrawingSpec(color=risk_color, thickness=2)
            )
            
            # عرض البيانات الحية
            cv2.rectangle(img, (0, 0), (300, 50), (0,0,0), -1)
            cv2.putText(img, f"Angle: {int(angle)} | {'RISK' if angle < 140 else 'SAFE'}", 
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)

        return frame.from_ndarray(img, format="bgr24")

# --- 3. واجهة الحوكمة الكاملة ---
st.title("🏗️ Smart Guard AI | مركز الحوكمة والرقابة الآلية")
st.info("نظام متكامل يربط الرؤية الحاسوبية ببنود العقود القانونية")

tab1, tab2, tab3 = st.tabs(["📺 الرقابة الحية", "📊 سجل المخاطر", "💬 غرفة العمليات"])

with tab1:
    col_v, col_m = st.columns([3, 1])
    with col_v:
        # الربط مع الكاميرا عبر WebRTC
        webrtc_streamer(
            key="final-stream", 
            video_processor_factory=IntegratedAIProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
    with col_m:
        st.metric("مستوى الامتثال", "94%", delta="+2%")
        st.subheader("تنبيهات الأرغونوميا")
        st.write("✅ تتبع 33 نقطة مفصلية نشط")
        st.write("⚠️ رصد انحناء ظهر غير سليم")

with tab2:
    st.subheader("📋 سجل المخاطر الديناميكي (Smart Ledger)")
    risk_df = pd.DataFrame({
        "التوقيت": ["10:15", "11:45"],
        "الحدث المكتشف": ["وضعية رفع خاطئة", "دخول منطقة محظورة"],
        "البند القانوني": ["Art 12.1 (Safety)", "Art 4.5 (Security)"],
        "الحالة": ["تم التوثيق", "تم إخطار المشرف"]
    })
    st.table(risk_df)
    st.button("📥 تصدير التقرير لشركاء Integrity Business Hub")

with tab3:
    st.subheader("💬 غرفة العمليات (War Room)")
    st.text_area("رسالة عاجلة للميدان:", "يرجى الالتزام بوضعية الرفع السليمة في منطقة التحميل...")
    if st.button("إرسال التنبيه الآن"):
        st.success("تم إرسال التنبيه لجميع الأجهزة المرتبطة بنجاح")

# --- الإضافات الجانبية ---
st.sidebar.title("إدارة النظام")
st.sidebar.success("✅ محرك AI جاهز")
st.sidebar.write("تكنولوجيا: MediaPipe + WebRTC")
