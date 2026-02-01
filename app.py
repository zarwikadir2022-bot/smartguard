import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import av
import numpy as np

# --- 1. إعداد الصفحة والشعار ---
st.set_page_config(page_title="Smart Guard AI", layout="wide", page_icon="🛡️")

st.title("🛡️ Smart Guard AI - Live Monitor")
st.markdown("---")

# --- 2. القائمة الجانبية (الإعدادات) ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.info("اضبط هذه القيم لتلائم إضاءة الموقع.")

# أشرطة التحكم في الحساسية (الحل لمشكلتك)
# القيمة الافتراضية 0.5، جرب خفضها إلى 0.3 إذا لم يظهر شيء
detection_conf = st.sidebar.slider("Min Detection Confidence (حساسية الكشف)", 0.1, 1.0, 0.5, 0.05)
tracking_conf = st.sidebar.slider("Min Tracking Confidence (حساسية التتبع)", 0.1, 1.0, 0.5, 0.05)

# خيارات إضافية للعرض
draw_landmarks = st.sidebar.checkbox("Show Skeleton (إظهار الهيكل)", value=True)
flip_video = st.sidebar.checkbox("Flip Video (قلب الصورة)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Integrity Business Hub")

# --- 3. تهيئة MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 4. كلاس المعالجة (The Brain) ---
class PoseDetector:
    def __init__(self):
        # نقوم بتهيئة الموديل بالقيم القادمة من الـ Sidebar
        self.pose = mp_pose.Pose(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    def recv(self, frame):
        # تحويل الإطار القادم من الويب (AV format) إلى صورة (NumPy array)
        img = frame.to_ndarray(format="bgr24")

        # قلب الصورة إذا طلب المستخدم (للكاميرا الأمامية)
        if flip_video:
            img = cv2.flip(img, 1)

        # تحويل الألوان لـ MediaPipe (يقبل RGB فقط)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # --- المعالجة الذكية ---
        results = self.pose.process(img_rgb)

        # تجهيز الصورة للرسم (إرجاعها لـ BGR)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # ارتفاع وعرض الصورة
        h, w, c = img_bgr.shape

        # --- الرسم والمنطق ---
        if results.pose_landmarks:
            # 1. رسم الهيكل العظمي
            if draw_landmarks:
                mp_drawing.draw_landmarks(
                    img_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # 2. كتابة حالة "تم الرصد"
            cv2.putText(img_bgr, "TARGET
