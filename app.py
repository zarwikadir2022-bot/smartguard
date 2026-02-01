import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import numpy as np

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="Smart Guard AI", layout="wide", page_icon="🛡️")
st.title("🛡️ Smart Guard AI - Core Monitor")
st.caption("Powered by OpenCV (No-Dependency Mode)")
st.markdown("---")

# --- 2. الإعدادات الجانبية ---
st.sidebar.title("⚙️ Control Panel")
# التحكم في حساسية الكشف (Scale Factor & Neighbors)
scale_factor = st.sidebar.slider("Detection Sensitivity", 1.1, 2.0, 1.1, 0.1)
min_neighbors = st.sidebar.slider("Filter Noise (Neighbors)", 1, 10, 3, 1)
flip_video = st.sidebar.checkbox("Flip Video", value=False)

# --- 3. تحميل نموذج كشف الجسم (المدمج في OpenCV) ---
# نستخدم النموذج الجاهز للكشف عن الجسم بالكامل
cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
# بديل: للكشف عن الوجه والجزء العلوي (أكثر دقة في السيلفي)
upper_body_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'

# محاولة تحميل النموذج
try:
    body_cascade = cv2.CascadeClassifier(cascade_path)
    if body_cascade.empty():
        # إذا فشل الجسم الكامل، نستخدم الجزء العلوي كبديل
        body_cascade = cv2.CascadeClassifier(upper_body_path)
except:
    st.error("Error loading Cascade Classifier XML.")

# --- 4. معالج الفيديو ---
class VideoProcessor:
    def recv(self, frame):
        # تحويل الإطار إلى مصفوفة
        img = frame.to_ndarray(format="bgr24")

        # قلب الفيديو
        if flip_video:
            img = cv2.flip(img, 1)

        # تحويل للرمادي (ضروري لعمل خوارزمية Haar)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- عملية الكشف (Detection) ---
        # يقوم المسح بالبحث عن أجسام في الصورة
        bodies = body_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(50, 50) # أقل حجم للجسم
        )

        h, w, c = img.shape

        # --- الرسم والمنطق ---
        if len(bodies) > 0:
            # تم رصد شخص أو أكثر
            for (x, y, width, height) in bodies:
                # رسم مربع أخضر حول الشخص
                cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 3)
                
                # كتابة تنبيه فوق الرأس
                cv2.putText(img, "HUMAN DETECTED", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # واجهة التطبيق العامة
            cv2.rectangle(img, (0, 0), (w, h), (0, 255, 0), 5)
            cv2.putText(img, f"TARGETS: {len(bodies)}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        else:
            # لم يتم رصد أحد
            cv2.putText(img, "SCANNING AREA...", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. تشغيل الكاميرا ---
webrtc_streamer(
    key="opencv-guard",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
