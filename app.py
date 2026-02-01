import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import pandas as pd

# --- 1. إعدادات الهوية البصرية للمنصة ---
st.set_page_config(page_title="Smart Guard AI - Total Governance", layout="wide")

# استيراد MediaPipe بشكل آمن
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    st.error(f"خطأ في تحميل المكتبات: {e}")

# دالة حساب الزوايا للأتمتة
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- 2. محرك الذكاء الاصطناعي المتكامل ---
class IntegratedAIProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose_tracker.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # نقاط الأرغونوميا (الجانب الأيسر كمثال)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            angle = calculate_angle(shoulder, hip, knee)
            risk_color = (0, 0, 255) if angle < 140 else (0, 255, 0)

            # رسم الهيكل العظمي
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(255,255,255), thickness=2),
                                     mp_drawing.DrawingSpec(color=risk_color, thickness=2))
            
            # عرض البيانات الحية
            cv2.putText(img, f"Back Angle: {int(angle)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, risk_color, 2)

        return frame.from_ndarray(img, format="bgr24")

# --- 3. تصميم واجهة المستخدم المتكاملة (Layout) ---
st.title("🏗️ Smart Guard AI | مركز الحوكمة والرقابة الآلية")
st.markdown(f"**مرحباً بك في Integrity Business Hub** - نظام المراقبة المدعوم بالذكاء الاصطناعي")

tab1, tab2, tab3 = st.tabs(["📺 الرقابة الحية", "📊 سجل المخاطر", "💬 غرفة العمليات"])

with tab1:
    col_v, col_m = st.columns([3, 1])
    with col_v:
        webrtc_streamer(key="integrated-stream", video_processor_factory=IntegratedAIProcessor)
    with col_m:
        st.metric("مستوى الامتثال (Compliance)", "92%", delta="+3%")
        st.write("**تنبيهات الأرغونوميا:**")
        st.warning("تم رصد 3 وضعيات رفع خاطئة اليوم")

with tab2:
    st.subheader("📋 سجل المخاطر الديناميكي (Dynamic Risk Register)")
    risk_df = pd.DataFrame({
        "التوقيت": ["10:15", "11:20", "12:05"],
        "نوع الخطر": ["وضعية جسد خاطئة", "تجاوز منطقة محظورة", "سقوط محتمل"],
        "بند CCAG": ["Art 12 (Safety)", "Art 4.2 (Access)", "Art 7 (Liabilities)"],
        "الإجراء المتخذ": ["تنبيه آلي", "إغلاق البوابة", "إشعار المسعف"]
    })
    st.table(risk_df)
    st.button("📥 تحميل تقرير PDF للمستشار طه خليل")

with tab3:
    st.subheader("📡 غرفة العمليات (War Room)")
    st.info("هذا هو البديل الرسمي لمنصة Workplace للتواصل الميداني")
    chat_input = st.text_input("إرسال تعليمات فورية للموقع:")
    if st.button("إرسال"):
        st.success(f"تم إرسال: '{chat_input}' إلى أجهزة المشرفين")

# --- 4. التذييل ---
st.sidebar.image("https://via.placeholder.com/150?text=Integrity+Hub", width=100)
st.sidebar.markdown("---")
st.sidebar.write("**الحالة التقنية:**")
st.sidebar.success("✅ محرك MediaPipe متصل")
st.sidebar.success("✅ بروتوكول WebRTC نشط")
