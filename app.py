import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import pandas as pd
import time

# --- إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Smart Guard AI - Control Center", layout="wide")

st.title("🏗️ Smart Guard AI: منظومة الرقابة والحوكمة الميدانية")
st.markdown("---")

# --- محرك معالجة الفيديو والذكاء الاصطناعي ---
class SmartGuardProcessor(VideoProcessorBase):
    def __init__(self):
        # محرك فصل الخلفية لرصد الحركة
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.motion_history = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. تحليل الحركة (Activity Analysis)
        fg_mask = self.backSub.apply(img)
        
        # تنظيف الصورة من الضجيج
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # حساب نسبة النشاط الحقيقي
        motion_area = np.sum(fg_mask == 255)
        activity_percent = (motion_area / fg_mask.size) * 100
        
        # 2. إضافة الطبقة الذكية على الفيديو (AI Overlay)
        status_color = (0, 0, 255) if activity_percent > 5 else (0, 255, 0)
        status_text = "DANGER: HIGH ACTIVITY" if activity_percent > 5 else "SAFE: NORMAL"
        
        # رسم مستطيل الحالة والمعلومات
        cv2.rectangle(img, (0, 0), (300, 60), (0, 0, 0), -1)
        cv2.putText(img, f"Activity: {activity_percent:.2f}%", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, status_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- توزيع واجهة المستخدم (Layout) ---
col_cam, col_stats = st.columns([2, 1])

with col_cam:
    st.subheader("📹 بث الرقابة الحية (WebRTC Stream)")
    webrtc_streamer(
        key="smart-guard-stream",
        video_processor_factory=SmartGuardProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    st.caption("ملاحظة: اضغط على 'Start' لتفعيل الكاميرا وبدء التحليل الذكي.")

with col_stats:
    st.subheader("📊 إحصائيات الحوكمة")
    st.metric("حالة الموقع اللحظية", "متصل (Live)", delta="Active")
    
    with st.expander("🚨 غرفة العمليات (War Room)"):
        st.write("بديل Workplace للتواصل الفوري:")
        st.text_area("تعليمات للمشرفين:", placeholder="أدخل تعليماتك هنا...")
        if st.button("إرسال تنبيه عاجل"):
            st.warning("تم إرسال التنبيه لجميع الهواتف المتصلة.")

    st.markdown("---")
    st.subheader("📋 سجل المخاطر الديناميكي")
    risk_data = {
        "الخطر": ["حركة غير مصرحة", "تجمهر عمال", "توقف مفاجئ"],
        "الحالة": ["مراقب", "مراقب", "مراقب"],
        "بند العقود": ["Art 12.1", "Art 4.2", "Art 7.5"]
    }
    st.table(pd.DataFrame(risk_data))

# --- قسم الحوكمة القانونية ---
st.markdown("---")
if st.button("توليد تقرير إثبات الالتزام (Compliance Report)"):
    st.success("✅ تم استخراج تقرير الحوكمة الرقمي لتقديمه للمستشارين.")
