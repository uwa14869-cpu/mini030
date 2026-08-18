# -*- coding: utf-8 -*-
"""
Used Car Price Predictor Web App - Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64

# ============================================================
# ⭐ ข้อมูลผู้พัฒนา (แก้ไขข้อมูลตรงนี้ได้ตามต้องการ)
# ============================================================
DEVELOPER_NAME = "นายภาณุพงศ์ ภุ่มพันธ์วงค์"
DEVELOPER_ID   = "664245030"
DEVELOPER_CLASS = "หมู่เรียน 66/43"
DEVELOPER_FACULTY = "คณะวิทยาศาสตร์และเทคโนโลยี"
DEVELOPER_UNIVERSITY = "มหาวิทยาลัยราชภัฏนครปฐม"
DEVELOPER_IMAGE = "img/030.jpg.jpg"  # วางไฟล์รูปในโฟลเดอร์ img/ ชื่อไฟล์ 030.jpg.jpg

# ==========================================
# ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(
    page_title="🚗 ทำนายราคารถมือสอง", 
    page_icon="🚗", 
    layout="wide"
)

# ==========================================
# Custom CSS สำหรับความสวยงาม
# ==========================================
st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #4C1FFF 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Card */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* ⭐ Developer Section CSS */
    .dev-section-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2rem;
        font-weight: bold;
        margin: 3rem 0 1.5rem 0;
    }
    
    .dev-card-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 15px 35px rgba(76, 31, 255, 0.15);
        margin: 1rem auto;
        max-width: 900px;
        display: flex;
        align-items: center;
        gap: 2.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(76, 31, 255, 0.1);
    }
    
    .dev-card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(76, 31, 255, 0.25);
    }
    
    .dev-image-wrapper {
        flex-shrink: 0;
        position: relative;
    }
    
    .dev-image-wrapper img {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        object-fit: cover;
        border: 5px solid #4C1FFF;
        box-shadow: 0 8px 20px rgba(76, 31, 255, 0.3);
    }
    
    .dev-image-wrapper::after {
        content: "🚗";
        position: absolute;
        top: -5px;
        right: -5px;
        font-size: 2.5rem;
        background: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .dev-info {
        flex: 1;
    }
    
    .dev-name {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0 0 0.8rem 0;
        border-bottom: 3px solid #4C1FFF;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    .dev-detail {
        font-size: 1.05rem;
        color: #4a5568;
        margin: 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .dev-detail-icon {
        font-size: 1.3rem;
        min-width: 30px;
        text-align: center;
    }
    
    .dev-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4C1FFF 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        margin-top: 1rem;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(76, 31, 255, 0.3);
    }
    
    @media (max-width: 768px) {
        .dev-card-container {
            flex-direction: column;
            text-align: center;
            padding: 1.5rem;
        }
        .dev-image-wrapper img {
            width: 150px;
            height: 150px;
        }
        .dev-name {
            font-size: 1.5rem;
            text-align: center;
            display: block;
        }
        .dev-detail {
            justify-content: center;
        }
    }
    
    .dev-emoji-fallback {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4C1FFF 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 5rem;
        border: 5px solid white;
        box-shadow: 0 8px 20px rgba(76, 31, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ฟังก์ชันโหลดรูป
# ============================================================
def get_developer_image_base64():
    """โหลดรูปผู้พัฒนา แปลงเป็น base64"""
    try:
        if os.path.exists(DEVELOPER_IMAGE):
            with open(DEVELOPER_IMAGE, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode()
    except Exception:
        pass
    return None

# ==========================================
# โหลดโมเดล ML
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('best_car_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_brand = joblib.load('le_brand.pkl')
        le_model = joblib.load('le_model.pkl')
        le_fuel = joblib.load('le_fuel.pkl')
        le_trans = joblib.load('le_trans.pkl')
        return model, scaler, le_brand, le_model, le_fuel, le_trans, True
    except FileNotFoundError:
        return None, None, None, None, None, None, False

model, scaler, le_brand, le_model, le_fuel, le_trans, is_loaded = load_assets()

# ==========================================
# UI หลักของแอปพลิเคชัน
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>🚗 ระบบทำนายราคารถมือสอง</h1>
    <p style="margin-top: 0.5rem; opacity: 0.9;">Interactive Machine Learning Prediction System</p>
</div>
""", unsafe_allow_html=True)

if is_loaded:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><h3>📝 กรอกข้อมูลรถ</h3>', unsafe_allow_html=True)
        brand = st.selectbox("ยี่ห้อ", sorted(le_brand.classes_))
        model_name = st.selectbox("รุ่น", sorted(le_model.classes_))
        year = st.number_input("ปีที่ผลิต", min_value=2000, max_value=2026, value=2018, step=1)
        mileage = st.number_input("ระยะทาง (กม.)", min_value=0, value=50000, step=1000)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><h3>⚙️ สเปคเพิ่มเติม</h3>', unsafe_allow_html=True)
        fuel = st.selectbox("ประเภทเชื้อเพลิง", sorted(le_fuel.classes_))
        transmission = st.selectbox("ระบบเกียร์", sorted(le_trans.classes_))
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 ทำนายราคา", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        try:
            input_data = pd.DataFrame({
                'Year': [year], 'Mileage_km': [mileage],
                'Brand_Enc': le_brand.transform([brand]),
                'Model_Enc': le_model.transform([model_name]),
                'Fuel_Enc': le_fuel.transform([fuel]),
                'Trans_Enc': le_trans.transform([transmission])
            })
            
            input_scaled = scaler.transform(input_data)
            predicted_price = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.success(f"💰 ราคาที่ทำนายได้: **{predicted_price:,.0f} บาท**")
            
            st.markdown("### 📋 สรุปข้อมูลที่กรอก")
            summary = pd.DataFrame({
                "ฟีเจอร์": ["ยี่ห้อ", "รุ่น", "ปี", "ไมล์", "เชื้อเพลิง", "เกียร์"],
                "ค่าที่กรอก": [brand, model_name, year, f"{mileage:,} กม.", fuel, transmission]
            })
            st.table(summary)
            
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

else:
    st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่าไฟล์ .pkl อยู่ในโฟลเดอร์เดียวกันกับไฟล์ app.py")

# ============================================================
# ⭐ Developer Section - จัดวางรูป ชื่อ หมู่เรียน ใหม่
# ============================================================
st.markdown("---")
st.markdown('<div class="dev-section-title">👨‍💻 ผู้พัฒนา / Developer</div>', unsafe_allow_html=True)

img_base64 = get_developer_image_base64()
if img_base64:
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" alt="Developer Photo">'
else:
    image_html = '<div class="dev-emoji-fallback">👨‍💻</div>'

st.markdown(f"""
<div class="dev-card-container">
    <div class="dev-image-wrapper">
        {image_html}
    </div>
    <div class="dev-info">
        <h2 class="dev-name">👨‍💻 {DEVELOPER_NAME}</h2>
        <div class="dev-detail">
            <span class="dev-detail-icon">🆔</span>
            <span><b>รหัสนักศึกษา:</b> {DEVELOPER_ID}</span>
        </div>
        <div class="dev-detail">
            <span class="dev-detail-icon">🎓</span>
            <span><b>หมู่เรียน:</b> {DEVELOPER_CLASS}</span>
        </div>
        <div class="dev-detail">
            <span class="dev-detail-icon">🏛️</span>
            <span><b>คณะ:</b> {DEVELOPER_FACULTY}</span>
        </div>
        <div class="dev-detail">
            <span class="dev-detail-icon">🏫</span>
            <span><b>มหาวิทยาลัย:</b> {DEVELOPER_UNIVERSITY}</span>
        </div>
        <span class="dev-badge">✨ Data Science Developer</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>© 2026 Used Car Price Prediction Project | Machine Learning Class</p>
    <p>พัฒนาโดย: {DEVELOPER_NAME}</p>
</div>
""", unsafe_allow_html=True)