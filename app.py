import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prediksi Diabetes", layout="centered")

# Cegah load model berulang-ulang (lebih efisien)
@st.cache_resource
def load_trained_model():
    return load_model("model_diabetes_mlp.h5")

model = load_trained_model()

# UI Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 Prediksi Diabetes (MLP Model)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan data pasien untuk memprediksi kemungkinan diabetes menggunakan model deep learning.</p>", unsafe_allow_html=True)
st.write("---")

# Input Form
st.subheader("Masukkan Data Pasien:")

col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    if gender == "Perempuan":
        preg = st.number_input("Jumlah Kehamilan", min_value=0, value=0, step=1)
    else:
        preg = 0  # default untuk laki-laki

    glu = st.number_input("Glukosa (mg/dL)", min_value=0, value=120)
    bp = st.number_input("Tekanan Darah (mmHg)", min_value=0, value=70)
    skin = st.number_input("Ketebalan Kulit (mm)", min_value=0, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, value=80)
    bmi = st.number_input("BMI", min_value=0.0,  value=30.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
    age = st.number_input("Usia (tahun)", min_value=0, value=30, step=1)

st.write("---")

# Prediksi hanya dilakukan saat tombol diklik
if st.button("🔍 Prediksi Sekarang"):
    with st.spinner("Sedang memproses..."):
        input_data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)
        prob = float(prediction[0][0])

    st.write("---")
    if prob >= 0.5:
        st.error(f"❌ Hasil: Positif Diabetes\n\n**Probabilitas: {prob:.2%}**")
    else:
        st.success(f"✅ Hasil: Negatif Diabetes\n\n**Probabilitas: {prob:.2%}**")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: small;'>© 2025 Aplikasi Prediksi Diabetes • Dibuat dengan Streamlit</div>", unsafe_allow_html=True)
