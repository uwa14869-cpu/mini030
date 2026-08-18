import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ทำนายราคารถมือสอง", page_icon="🚗", layout="wide")

# โหลดโมเดลและเครื่องมือที่บันทึกไว้
@st.cache_resource
def load_assets():
    model = joblib.load('best_car_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_brand = joblib.load('le_brand.pkl')
    le_model = joblib.load('le_model.pkl')
    le_fuel = joblib.load('le_fuel.pkl')
    le_trans = joblib.load('le_trans.pkl')
    return model, scaler, le_brand, le_model, le_fuel, le_trans

try:
    model, scaler, le_brand, le_model, le_fuel, le_trans = load_assets()
    is_loaded = True
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์โมเดล! กรุณารัน train_model.py ก่อน")
    is_loaded = False

# ==========================================
# 5. Streamlit Application (5 คะแนน)
# ==========================================
st.title("🚗 ระบบทำนายราคารถมือสอง")
st.markdown("---")
st.caption("พัฒนาโดย: [ชื่อ-นามสกุลของคุณ] | หมู่เรียน: [เลขหมู่]")

if is_loaded:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 กรอกข้อมูลรถ")
        brand = st.selectbox("ยี่ห้อ", le_brand.classes_)
        model_name = st.selectbox("รุ่น", le_model.classes_)
        year = st.number_input("ปีที่ผลิต", min_value=2000, max_value=2026, value=2018)
        mileage = st.number_input("ระยะทาง (กม.)", min_value=0, value=50000, step=1000)
    
    with col2:
        st.subheader("⚙️ สเปคเพิ่มเติม")
        fuel = st.selectbox("ประเภทเชื้อเพลิง", le_fuel.classes_)
        transmission = st.selectbox("ระบบเกียร์", le_trans.classes_)
        
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
            
            # Transform ด้วย Scaler (ถ้าโมเดลต้องการ)
            input_scaled = scaler.transform(input_data)
            
            # ทำนาย
            predicted_price = model.predict(input_scaled)[0]
            
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
            st.error(f"เกิดข้อผิดพลาด: {str(e)}")

else:
    st.warning("⚠️ กรุณาเทรนโมเดลก่อนใช้งานแอปพลิเคชัน")

# Footer
st.markdown("---")
st.caption("© 2026 Used Car Price Prediction Project | Machine Learning Class")