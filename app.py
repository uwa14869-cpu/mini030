# -*- coding: utf-8 -*-
"""
Used Car Price Predictor Web App - Streamlit (Enhanced UI)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64

# ============================================================
# ⭐ ข้อมูลผู้พัฒนา
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
    /* 1. พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%) !important;
    }
    
    /* 2. ปรับสีตัวอักษรหลักให้อ่านง่าย */
    .stApp p, .stApp div, .stApp label, .stApp span {
        color: #475569 !important; 
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important; 
        font-weight: 700 !important;
    }

    /* 3. Header */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        color: white !important;
        text-align: center;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
        margin-bottom: 2.5rem;
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin-top: 0.8rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* 4. Card หลัก */
    .card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .card h3 {
        color: #4f46e5 !important;
        margin-bottom: 1.2rem !important;
        font-size: 1.3rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }

    /* 5. ปรับแต่ง Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #1e293b !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        background-color: #ffffff !important;
    }

    /* 6. ปรับแต่งปุ่ม */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }

    /* 7. Developer Section */
    .dev-section-title {
        text-align: center;
        color: #1e293b !important;
        font-size: 2rem;
        font-weight: 800;
        margin: 3rem 0 1.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .dev-card-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.1);
        margin: 1rem auto;
        max-width: 900px;
        display: flex;
        align-items: center;
        gap: 2.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }
    
    .dev-card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
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
        border: 4px solid #ffffff;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
    }
    
    .dev-image-wrapper::after {
        content: "🚗";
        position: absolute;
        bottom: 5px;
        right: 5px;
        font-size: 2.2rem;
        background: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 3px solid #f8fafc;
    }
    
    .dev-info {
        flex: 1;
    }
    
    .dev-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #4f46e5 !important;
        margin: 0 0 0.8rem 0;
        border-bottom: 3px solid #c7d2fe;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    .dev-detail {
        font-size: 1.05rem;
        color: #475569 !important;
        margin: 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-weight: 500;
    }
    
    .dev-detail-icon {
        font-size: 1.3rem;
        min-width: 30px;
        text-align: center;
    }
    
    .dev-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        margin-top: 1.2rem;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    @media (max-width: 768px) {
        .dev-card-container {
            flex-direction: column;
            text-align: center;
            padding: 2rem 1.5rem;
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
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 5rem;
        border: 4px solid white;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
    }

    /* ==========================================
       8. 🌟 สรุปข้อมูลแบบการ์ด (ใหม่!)
       ========================================== */
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .summary-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }
    
    .summary-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.15);
        border-color: #6366f1;
    }
    
    .summary-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .summary-label {
        font-size: 0.8rem;
        color: #64748b !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .summary-value {
        font-size: 1.15rem;
        color: #1e293b !important;
        font-weight: 800;
        word-break: break-word;
    }

    /* 9. Footer */
    .custom-footer {
        background: #1e293b;
        color: #94a3b8 !important;
        padding: 1.5rem;
        border-radius: 16px 16px 0 0;
        margin-top: 3rem;
        text-align: center;
    }
    .custom-footer p {
        color: #94a3b8 !important;
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    .custom-footer strong {
        color: #e2e8f0 !important;
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
    <p>Interactive Machine Learning Prediction System</p>
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
            
            # --- สร้างการ์ดสรุปข้อมูลแบบสวยงาม ---
            summary_items = [
                {"icon": "🏷️", "label": "ยี่ห้อ", "value": brand},
                {"icon": "🚘", "label": "รุ่น", "value": model_name},
                {"icon": "📅", "label": "ปีที่ผลิต", "value": year},
                {"icon": "🛣️", "label": "ระยะทาง", "value": f"{mileage:,} กม."},
                {"icon": "⛽", "label": "เชื้อเพลิง", "value": fuel},
                {"icon": "⚙️", "label": "ระบบเกียร์", "value": transmission}
            ]
            
            summary_html = '<div class="summary-grid">'
            for item in summary_items:
                summary_html += f"""
                <div class="summary-card">
                    <div class="summary-icon">{item['icon']}</div>
                    <div class="summary-label">{item['label']}</div>
                    <div class="summary-value">{item['value']}</div>
                </div>
                """
            summary_html += '</div>'
            
            st.markdown(summary_html, unsafe_allow_html=True)
            # ------------------------------------
            
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

else:
    st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่าไฟล์ .pkl อยู่ในโฟลเดอร์เดียวกันกับไฟล์ app.py")

# ============================================================
# ⭐ Developer Section
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
st.markdown(f"""
<div class="custom-footer">
    <p>© 2026 Used Car Price Prediction Project | Machine Learning Class</p>
    <p>พัฒนาโดย: <strong>{DEVELOPER_NAME}</strong></p>
</div>
""", unsafe_allow_html=True)