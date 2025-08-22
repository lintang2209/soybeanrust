import streamlit as st
import numpy as np
from PIL import Image
import os

# Path model
CNN_MODEL_PATH = "models/cnn_soybean_rust.keras"
YOLO_MODEL_PATH = "models/best.pt"
# Load models
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

# Class names
CLASS_NAMES = ["Daun Sehat", "Soybean Rust"]

st.title("📷 Deteksi Penyakit Daun Kedelai (CNN vs YOLOv8)")

uploaded_file = st.file_uploader("Unggah gambar daun kedelai", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # CNN Prediction
    img_resized = image.resize((224,224)) # sesuaikan ukuran input CNN
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    cnn_pred = cnn_model.predict(img_array)
    cnn_class = CLASS_NAMES[np.argmax(cnn_pred)]

    # YOLOv8 Prediction
    yolo_results = yolo_model.predict(np.array(image), imgsz=640, conf=0.35)
    yolo_img = yolo_results[0].plot()  # annotated image

    st.subheader("Hasil Prediksi CNN")
    st.write(f"Prediksi: **{cnn_class}**")
    st.write(f"Probabilitas: {np.max(cnn_pred)*100:.2f}%")

    st.subheader("Hasil Prediksi YOLOv8")
    st.image(yolo_img, caption="Hasil deteksi YOLOv8", use_column_width=True)
