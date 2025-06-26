import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import MultilayerPerceptronClassificationModel

# Jalankan SparkSession (local)
spark = SparkSession.builder.appName("ANN_Predictor").getOrCreate()

# Load model yang sudah dilatih
model = MultilayerPerceptronClassificationModel.load("model/ann_model")

# UI Streamlit
st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")
st.title("🏠 Prediksi Kategori Harga Rumah")

# Form Input
x1 = st.number_input("📅 Tanggal Transaksi (cth: 2013.5)", value=2013.0)
x2 = st.number_input("📏 Umur Rumah (tahun)", value=10.0)
x3 = st.number_input("🚉 Jarak ke MRT (meter)", value=500.0)
x4 = st.number_input("🏪 Jumlah Toko Terdekat", value=5)
x5 = st.number_input("🌍 Latitude", value=24.98)
x6 = st.number_input("🌍 Longitude", value=121.54)

if st.button("🔍 Prediksi"):
    input_df = spark.createDataFrame([(Vectors.dense([x1, x2, x3, x4, x5, x6]),)], ["features"])
    result = model.transform(input_df).collect()[0]["prediction"]
    kategori = ["Murah", "Sedang", "Mahal"][int(result)]
    st.success(f"💡 Hasil Prediksi: **{kategori}**")
