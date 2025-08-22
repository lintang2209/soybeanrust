import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import torch

# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# Path model (pastikan file model ada di folder 'models')
CNN_MODEL_PATH = "models/cnn_soybean_rust.keras"
YOLO_MODEL_PATH = "models/best.pt"

# Load models (gunakan cache agar tidak load berulang kali)
@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model(CNN_MODEL_PATH)

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL_PATH)

cnn_model = load_cnn()
yolo_model = load_yolo()

# UI
st.title("Soybean Rust Detection App 🌱")
st.write("Implementasi CNN dan YOLOv8 di Streamlit Cloud")

uploaded_file = st.file_uploader("Upload gambar daun kedelai", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # ---- Prediksi CNN ----
    img_resized = image.resize((224, 224))  # sesuaikan ukuran dengan dataset CNN
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    cnn_pred = cnn_model.predict(img_array)
    cnn_class = np.argmax(cnn_pred, axis=1)[0]
    cnn_label = "Daun Sehat" if cnn_class == 0 else "Soybean Rust"

    st.subheader("Hasil Prediksi CNN")
    st.write(f"**{cnn_label}** (Probabilitas: {cnn_pred[0][cnn_class]:.2f})")

    # ---- Prediksi YOLO ----
    st.subheader("Hasil Prediksi YOLOv8")
    results = yolo_model(image)
    for r in results:
        st.image(r.plot(), caption="Deteksi YOLOv8", use_column_width=True)
