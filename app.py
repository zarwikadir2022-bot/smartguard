import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import av
import numpy as np

# --- 1. إعداد الصفحة ---
st.set_page_config(
    page_title="Smart Guard AI", 
    layout="wide", 
    page_icon="🛡️"
)

st.title("🛡️ Smart Guard AI - Live Monitor")
st.markdown("---")

# --- 2. القائمة الجانبية (Control Panel) ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.info("Adjust settings based on lighting conditions.")

# أشرطة التحكم (Sliders)
# القيمة الافتراضية 0.5. حرك الشريط لليسار (0.3) إذا كان الكشف ضعيفاً
detection_conf = st.sidebar.slider("Min Detection Confidence", 0.1, 1.0, 0.5, 0.05)
tracking_conf = st.sidebar.slider("Min Tracking Confidence", 0.1, 1.0, 0.5, 0.05)

# خيارات العرض
draw_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)
flip_video = st.sidebar.checkbox("Flip Video (Selfie Mode)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("System Status: Online")

# --- 3. تهيئة مكتبة MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 4. كلاس المعالجة (The Processor) ---
class PoseProcessor:
    def __init__(self):
        # تهيئة الموديل بالقيم القادمة من المتغيرات العامة (التي حددها المستخدم)
        self.pose = mp_pose.Pose(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    def recv(self, frame):
        # تحويل الإطار من تنسيق الويب إلى مصفوفة (Image Array)
        img = frame.to_ndarray(format="bgr24")

        # 1. قلب الصورة إذا تم تفعيل الخيار
        if flip_video:
            img = cv2.flip(img, 1)

        # 2. تحويل الألوان للمعالجة (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. تشغيل الذكاء الاصطناعي
        results = self.pose.process(img_rgb)

        # 4. العودة للألوان الطبيعية للرسم (BGR)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # أبعاد الصورة للكتابة والرسم
        h, w, c = img_bgr.shape

        # 5. المنطق: هل وجدنا شخصاً؟
        if results.pose_landmarks:
            # --- حالة: تم الرصد (Target Locked) ---
            
            # رسم الهيكل العظمي
            if draw_skeleton:
                mp_drawing.draw_landmarks(
                    img_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # كتابة النص (بدون إيموجي لتجنب الأخطاء)
            cv2.putText(img_bgr, "TARGET LOCKED", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # رسم إطار أخضر حول الشاشة
            cv2.rectangle(img_bgr, (0,0), (w,h), (0, 255, 0), 5)
            
        else:
            # --- حالة: جاري البحث (Searching) ---
            
            cv2.putText(img_bgr, "SEARCHING...", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (
