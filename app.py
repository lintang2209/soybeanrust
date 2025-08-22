import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# ==== KONFIGURASI ====
USE_MODEL = "YOLO"  # "CNN" atau "YOLO"
CNN_MODEL_PATH = "models/cnn_soybean_rust.keras"
YOLO_MODEL_PATH = "models/best.pt"

# ==== LOAD MODEL ====
cnn_model = None
yolo_model = None
class_names = ["Daun Sehat", "Soybean Rust"]

if USE_MODEL == "CNN":
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

elif USE_MODEL == "YOLO":
    yolo_model = YOLO(YOLO_MODEL_PATH)

# ==== UI ====
st.title("📷 Deteksi Penyakit Daun Kedelai (CNN vs YOLOv8)")
st.write("Upload gambar daun kedelai untuk deteksi penyakit *Soybean Rust*.")

uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Gambar yang diupload", use_column_wid
