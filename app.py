import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import torch
import os

# ===============================
# CONFIG
# ===============================
CNN_MODEL_PATH = "models/cnn_soybean_rust.keras"   # path model CNN
YOLO_MODEL_PATH = "models/best.pt"  # path model YOLOv8
CLASS_NAMES = ["Daun Sehat", "Soybean Rust"]

# Load CNN model (jika ada)
cnn_model = None
if os.path.exists(CNN_MODEL_PATH):
    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal load CNN model: {e}")

# Load YOLOv8 model (jika ada)
yolo_model = None
if os.path.exists(YOLO_MODEL_PATH):
    try:
        yolo_model = torch.hub.load("ultralytics/yolov5", "custom", path=YOLO_MODEL_PATH, force_reload=True)
    except Exception as e:
        st.error(f"Gagal load YOLOv8 model: {e}")

# ===============================
# STREAMLIT UI
# ===============================
st.title("🌱 Soybean Rust Detection")
st.write("Pilih model deteksi (CNN untuk klasifikasi, YOLOv8 untuk deteksi bounding box)")

model_choice = st.radio("Pilih Model:", ["CNN", "YOLOv8"])

uploaded_file = st.file_uploader("Upload gambar daun kedelai", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diunggah", use_column_width=True)

    if model_choice == "CNN" and cnn_model:
        # Preprocessing
        img_resized = image.resize((224, 224))  # samakan dengan ukuran training CNN
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        prediction = cnn_model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        st.success(f"**Prediksi: {CLASS_NAMES[class_idx]} ({confidence:.2f}%)**")

    elif model_choice == "YOLOv8" and yolo_model:
        # Run YOLO inference
        results = yolo_model(image)

        # Tampilkan hasil deteksi
        st.image(np.squeeze(results.render()), caption="Hasil Deteksi YOLOv8", use_column_width=True)

    else:
        st.warning("Model belum tersedia atau gagal diload.")
