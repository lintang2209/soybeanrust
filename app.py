import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import torch

# Prevent some torch hub errors
torch.hub._validate_not_a_forked_repo = lambda a,b,c: True

st.title("Soybean Rust Detection 🌱")
st.write("Deteksi penyakit daun kedelai menggunakan CNN dan YOLOv8")

# --- Load Model ---
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("models/cnn_soybean_rust.keras")

@st.cache_resource
def load_yolo_model():
    return YOLO("models/best.pt")

cnn_model = load_cnn_model()
yolo_model = load_yolo_model()

# --- Upload Gambar ---
uploaded_file = st.file_uploader("Upload gambar daun kedelai", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # --- Prediksi CNN ---
    img_resized = image.resize((224,224)) # sesuaikan dengan input CNN
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)
    cnn_pred = cnn_model.predict(img_array)
    cnn_class = np.argmax(cnn_pred, axis=1)[0]
    cnn_label = "Daun Sehat" if cnn_class == 0 else "Soybean Rust"

    st.subheader("Hasil Prediksi CNN")
    st.write(f"**{cnn_label}** (Probabilitas: {cnn_pred[0][cnn_class]:.2f})")

    # --- Prediksi YOLO ---
    st.subheader("Hasil Deteksi YOLOv8")
    results = yolo_model(image)
    for r in results:
        st.image(r.plot(), caption="Deteksi YOLOv8", use_column_width=True)
