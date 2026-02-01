import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time

# إعداد الصفحة
st.set_page_config(page_title="Smart Guard AI - Real-time Vision", layout="wide")

st.title("🛡️ Smart Guard AI - نظام الرقابة الحية")
st.markdown("تحليل نشاط الموقع عبر كاميرا الحاسوب مباشرة")

# --- دالة تحليل الإحصائيات الحقيقية ---
def process_frame(frame, backSub):
    # 1. تحويل الصورة وفصل الخلفية لرصد الحركة
    fg_mask = backSub.apply(frame)
    
    # 2. تنظيف الضجيج
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    
    # 3. حساب نسبة الحركة (الإحصائية الحقيقية)
    motion_area = np.sum(fg_mask == 255)
    total_area = frame.shape[0] * frame.shape[1]
    activity_percent = (motion_area / total_area) * 100
    
    return fg_mask, round(activity_percent, 2)

# --- واجهة التحكم ---
with st.sidebar:
    st.header("⚙️ إعدادات الحساسية")
    threshold = st.slider("حد تنبيه النشاط المرتفع (%)", 0.0, 20.0, 5.0)
    run_cam = st.toggle("تشغيل الكاميرا الحية", value=False)

# --- منطقة العرض ---
col_video, col_stats = st.columns([2, 1])

if run_cam:
    cap = cv2.VideoCapture(0) # فتح كاميرا الحاسوب
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    video_placeholder = col_video.empty()
    metrics_placeholder = col_stats.empty()
    
    # سجل تاريخي بسيط للإحصائيات
    history = []

    while run_cam:
        ret, frame = cap.read()
        if not ret:
            st.error("فشل الوصول إلى الكاميرا")
            break
        
        # معالجة الإطار واستخراج الإحصائيات
        processed_mask, activity = process_frame(frame, backSub)
        history.append(activity)
        if len(history) > 20: history.pop(0)

        # عرض الفيديو (تحويل الألوان ليتناسب مع Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # تحديث الإحصائيات الحقيقية
        with metrics_placeholder.container():
            st.metric("مستوى النشاط اللحظي", f"{activity}%")
            st.metric("متوسط الحركة (آخر دقيقة)", f"{np.mean(history):.2f}%")
            
            if activity > threshold:
                st.warning(f"⚠️ تنبيه: نشاط غير عادي رُصد في الموقع! ({activity}%)")
                # هنا يمكن ربط نظام التنبيه الصوتي الذي ناقشناه مع شعيب
            else:
                st.success("✅ الوضع مستقر")
            
            # عرض رسم بياني صغير للنشاط
            st.line_chart(history)

        time.sleep(0.05) # تحسين الأداء
    
    cap.release()
else:
    col_video.info("قم بتفعيل الزر الجانبي لبدء الرقابة الحية.")
