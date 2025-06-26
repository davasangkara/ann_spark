import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("ann_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Kategori Harga Rumah (Tanpa Spark)")

x1 = st.number_input("Tanggal Transaksi", value=2013.0)
x2 = st.number_input("Umur Rumah", value=10.0)
x3 = st.number_input("Jarak ke MRT", value=500.0)
x4 = st.number_input("Jumlah Toko Terdekat", value=5)
x5 = st.number_input("Latitude", value=24.98)
x6 = st.number_input("Longitude", value=121.54)

if st.button("Prediksi"):
    input_data = np.array([[x1, x2, x3, x4, x5, x6]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    kategori = ["Murah", "Sedang", "Mahal"][int(pred)]
    st.success(f"Prediksi Harga: **{kategori}**")
