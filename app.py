import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# ตั้งค่าหน้าเว็บและข้อมูลผู้พัฒนา
# ==========================================
st.set_page_config(page_title="ทำนายราคารถมือสอง", page_icon="🚗", layout="wide")

# --- ส่วนแสดงข้อมูลผู้พัฒนา ---
DEVELOPER_NAME = "นายสมชาย ใจดี"
STUDENT_ID = "65012345678"
SECTION = "หมู่เรียน 30"
GITHUB_URL = "https://github.com/uwa14869-cpu"

# ✅ ฟังก์ชันค้นหารูปอัตโนมัติ (รองรับหลายนามสกุล/ชื่อไฟล์)
def find_profile_image():
    possible_names = ["030.jpg", "030.png", "profile.jpg", "profile.png", "dev.jpg"]
    base_path = Path("030.jpg").parent
    
    for name in possible_names:
        img_file = base_path / name
        if img_file.exists():
            return str(img_file), name
            
    # ถ้าหาไม่เจอเลย ให้ใช้ลิงก์รูปตัวอย่างแทน
    return "https://via.placeholder.com/150?text=No+Image", None

image_source, found_filename = find_profile_image()

# โหลดโมเดล
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
# UI หลัก
# ==========================================
st.title("🚗 ระบบทำนายราคารถมือสอง")

# แสดงข้อมูลผู้พัฒนาพร้อมรูปภาพ
col_img, col_info = st.columns([1, 4])

with col_img:
    st.image(image_source, width=150, caption=f"ผู้พัฒนา ({found_filename or 'Default'})")
    
    # แจ้งเตือนเฉพาะเมื่อไม่พบไฟล์จริง (แต่แอปยังทำงานได้)
    if found_filename is None:
        st.warning("⚠️ ไม่พบรูปโปรไฟล์ใน GitHub\nกรุณาอัปโหลดไฟล์ 030.jpg หรือ profile.jpg")

with col_info:
    st.info(f"""
    **‍💻 ผู้พัฒนา:** {DEVELOPER_NAME}  
    **🎓 รหัสนักศึกษา:** {STUDENT_ID} | **{SECTION}**  
    **🔗 GitHub:** [{GITHUB_URL}]({GITHUB_URL})  
    ** ปีการศึกษา:** 2569 (2026)
    """)

st.markdown("---")

if is_loaded:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 กรอกข้อมูลรถ")
        brand = st.selectbox("ยี่ห้อ", sorted(le_brand.classes_))
        model_name = st.selectbox("รุ่น", sorted(le_model.classes_))
        year = st.number_input("ปีที่ผลิต", min_value=2000, max_value=2026, value=2018, step=1)
        mileage = st.number_input("ระยะทาง (กม.)", min_value=0, value=50000, step=1000)
    
    with col2:
        st.subheader("⚙️ สเปคเพิ่มเติม")
        fuel = st.selectbox("ประเภทเชื้อเพลิง", sorted(le_fuel.classes_))
        transmission = st.selectbox("ระบบเกียร์", sorted(le_trans.classes_))
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 ทำนายราคา", type="primary", use_container_width=True)

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
            
            st.success(f"💰 ราคาที่ทำนายได้: **{predicted_price:,.0f} บาท**")
            
            st.markdown("---")
            st.subheader("📋 สรุปข้อมูลที่กรอก")
            summary = pd.DataFrame({
                "ฟีเจอร์": ["ยี่ห้อ", "รุ่น", "ปี", "ไมล์", "เชื้อเพลิง", "เกียร์"],
                "ค่าที่กรอก": [brand, model_name, year, f"{mileage:,} กม.", fuel, transmission]
            })
            st.table(summary)
            
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

else:
    st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่าไฟล์ .pkl ถูกอัปโหลดขึ้น GitHub แล้ว")

st.markdown("---")
st.caption(f"© 2026 Used Car Price Prediction Project | Machine Learning Class | [{DEVELOPER_NAME}]({GITHUB_URL})")