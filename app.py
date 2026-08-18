import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# ตั้งค่าหน้าเว็บและข้อมูลผู้พัฒนา
# ==========================================
st.set_page_config(page_title="ทำนายราคารถมือสอง", page_icon="🚗", layout="wide")

# --- ส่วนแสดงข้อมูลผู้พัฒนา (แก้ไขตรงนี้) ---
DEVELOPER_NAME = "นายภาณุพงศ์ ภุ่มพันธ์วงค์"          # <-- ใส่ชื่อ-นามสกุล จริง
STUDENT_ID = "664245030"               # <-- ใส่รหัสนักศึกษา
SECTION = "หมู่เรียน 66/43"                 # <-- ใส่เลขหมู่เรียน
GITHUB_URL = "https://github.com/uwa14869-cpu"  # ✅ ลิงก์ GitHub ของคุณ
# -----------------------------------------

# โหลดโมเดลและเครื่องมือที่บันทึกไว้
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
    except FileNotFoundError as e:
        return None, None, None, None, None, None, False

model, scaler, le_brand, le_model, le_fuel, le_trans, is_loaded = load_assets()

# ==========================================
# UI หลักของ Streamlit Application
# ==========================================
st.title(" ระบบทำนายราคารถมือสอง")

# แสดงกล่องข้อมูลผู้พัฒนาพร้อมลิงก์ GitHub ที่คลิกได้
st.info(f"""
**👨‍ ผู้พัฒนา:** {DEVELOPER_NAME}  
**🎓 รหัสนักศึกษา:** {STUDENT_ID} | **{SECTION}**  
** GitHub Repository:** [{GITHUB_URL}]({GITHUB_URL})  
**📅 ปีการศึกษา:** 2569 (2026)
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

    # Logic การทำนาย
    if predict_btn:
        try:
            # Encode ข้อมูล input
            input_data = pd.DataFrame({
                'Year': [year],
                'Mileage_km': [mileage],
                'Brand_Enc': le_brand.transform([brand]),
                'Model_Enc': le_model.transform([model_name]),
                'Fuel_Enc': le_fuel.transform([fuel]),
                'Trans_Enc': le_trans.transform([transmission])
            })
            
            # Transform ด้วย Scaler
            input_scaled = scaler.transform(input_data)
            
            # ทำนาย
            predicted_price = model.predict(input_scaled)[0]
            
            # แสดงผลลัพธ์แบบเด่นชัด
            st.success(f"💰 ราคาที่ทำนายได้: **{predicted_price:,.0f} บาท**")
            
            # แสดงรายละเอียด Input
            st.markdown("---")
            st.subheader(" สรุปข้อมูลที่กรอก")
            summary = pd.DataFrame({
                "ฟีเจอร์": ["ยี่ห้อ", "รุ่น", "ปี", "ไมล์", "เชื้อเพลิง", "เกียร์"],
                "ค่าที่กรอก": [brand, model_name, year, f"{mileage:,} กม.", fuel, transmission]
            })
            st.table(summary)
            
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

else:
    st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่าไฟล์ .pkl ถูกอัปโหลดขึ้น GitHub แล้ว")
    st.warning("ไฟล์ที่ต้องมี: best_car_price_model.pkl, scaler.pkl, le_*.pkl")

# Footer
st.markdown("---")
st.caption(f"© 2026 Used Car Price Prediction Project | Machine Learning Class | [{DEVELOPER_NAME}]({GITHUB_URL})")