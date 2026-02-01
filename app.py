import streamlit as st
import pandas as pd
import numpy as np
import time

# --- إعدادات الواجهة الاحترافية ---
st.set_page_config(page_title="Smart Guard AI - Safety Command Center", layout="wide")

# --- محاكي الرؤية الحاسوبية (AI Logic) ---
def get_ai_detections():
    """محاكاة الكشف الذكي بناءً على خوارزمية تحليل الاحتمالات"""
    return [
        {"Target": "Worker_01", "Status": "No Helmet", "Severity": "Critical", "Prob": 98.5},
        {"Target": "Worker_04", "Status": "Incorrect Lifting", "Severity": "Medium", "Prob": 75.2},
        {"Target": "Zone_B", "Status": "Obstacle Detected", "Severity": "High", "Prob": 88.0}
    ]

# --- إدارة بيانات الجلسة (للتواصل وسجل المخاطر) ---
if 'alerts' not in st.session_state:
    st.session_state.alerts = get_ai_detections()
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

# --- الواجهة الرئيسية ---
st.title("🛡️ Smart Guard AI: منظومة الرقابة والحوكمة الميدانية")
st.markdown("---")

# 1. لوحة المؤشرات العليا (Executive Dashboard)
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("مؤشر الخطر اللحظي", "62%", delta="تنبيه: مرتفع")
with col_m2:
    st.metric("العمال في الموقع", "85", "متصل")
with col_m3:
    st.metric("مخالفات تم رصدها", len(st.session_state.alerts), "اليوم")
with col_m4:
    st.metric("وقت الاستجابة (متوسط)", "1.2 ثانية", "Zero Latency")

st.markdown("---")

# 2. منطقة العمليات المركزية
col_feed, col_ops = st.columns([2, 1])

with col_feed:
    st.subheader("📹 بث الرقابة الذكية (AI Live Feed)")
    # محاكاة واجهة الكاميرا مع المربعات الذكية
    st.image("https://via.placeholder.com/800x450?text=Live+AI+Monitoring+-+Safety+Vigilance", use_column_width=True)
    
    # جدول التنبيهات الذكية المستمد من خوارزمية analyze_target
    st.subheader("⚠️ سجل التنبيهات اللحظي")
    df_alerts = pd.DataFrame(st.session_state.alerts)
    st.table(df_alerts)

with col_ops:
    st.subheader("🚨 غرفة العمليات (War Room)")
    # وحدة التواصل البديلة لـ Workplace لضمان الحوكمة
    with st.container(height=350, border=True):
        for msg in st.session_state.chat_log:
            st.write(msg)
    
    instruction = st.chat_input("أرسل تعليمات فورية للموقع...")
    if instruction:
        st.session_state.chat_log.append(f"👤 مدير الموقع: {instruction}")
        st.rerun()
    
    st.markdown("---")
    st.subheader("⚖️ مطابقة المعايير (Compliance)")
    st.checkbox("توثيق المخالفات في السجل العدلي للموقع", value=True)
    if st.button("توليد تقرير إثبات الالتزام (CCAG)"):
        st.success("✅ تم تصدير التقرير القانوني بنجاح.")

# 3. قسم الصحة المهنية (Ergonomics Module) بناءً على نصيحة شعيب
st.markdown("---")
st.subheader("🧘 وحدة الارغونوميا وتحليل الوضعيات (Ergo-Sense)")
c1, c2 = st.columns(2)
with c1:
    st.info("💡 يتم الآن تحليل وضعيات الرفع لـ 15 عاملاً في المنطقة 'أ'.")
    st.progress(75, text="التزام بوضعية الظهر السليمة")
with c2:
    if st.button("إرسال تنبيه صوتي للعمال ذوي الوضعيات الخاطئة"):
        st.warning("🔊 تم إرسال تنبيه 'صحح وضعيتك' فوراً عبر المكبرات.")

# 4. سجل المخاطر الديناميكي (Dynamic Risk Register) مستوحى من رؤية مارفيل
st.subheader("📋 سجل المخاطر التفاعلي")
risk_data = {
    "الخطر": ["السقوط", "الإصابات العضلية", "التصادم"],
    "الاحتمالية الحالية": ["عالية", "متوسطة", "منخفضة"],
    "التحقق الذكي": ["مفعل 24/7", "مراقب آلياً", "مراقب آلياً"],
    "بند العقد المرتبط": ["Art 12.1", "Art 4.5", "Art 2.2"]
}
st.dataframe(pd.DataFrame(risk_data), use_container_width=True)
