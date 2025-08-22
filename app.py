import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# =============================
# Path model
# =============================
CNN_MODEL_PATH = "models/cnn_soybean_rust.keras"
YOLO_MODEL_PATH = "models/best.pt"

# =============================
# Load CNN Model
# =============================
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(CNN_MODEL_PATH)

# =============================
# Load YOLO Model
# =============================
@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)

cnn_model = load_cnn_model()
yolo_model = load_yolo_model()

# =============================
# Kelas CNN
# =============================
CLASS_NAMES = ["daun sehat", "soybean rust"]

# =============================
# Streamlit UI
# =============================
st.title("🍃 Deteksi Penyakit Daun Kedelai (CNN & YOLOv8)")
st.write("Upload gambar daun kedelai untuk memprediksi penyakit menggunakan **CNN** dan **YOLOv8**.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar daun kedelai", type=["jpg", "jpeg", "png"])

# Confidence Threshold Slider
conf_threshold = st.slider("Confidence Threshold (YOLO)", 0.05, 1.0, 0.25, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # =============================
    # Prediksi CNN
    # =============================
    st.subheader("🔹 Prediksi CNN")
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = cnn_model.predict(img_array)
    score = np.max(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.write(f"**Kelas Prediksi:** {predicted_class}")
    st.write(f"**Confidence:** {score:.2f}")

    # =============================
    # Prediksi YOLOv8
    # =============================
    st.subheader("🔹 Prediksi YOLOv8")
    results = yolo_model.predict(np.array(image), conf=conf_threshold)

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results[0].names[cls_id]

            st.write(f"- **{label}** dengan confidence **{conf:.2f}**")
        st.image(results[0].plot(), caption="Hasil Deteksi YOLOv8", use_column_width=True)
    else:
        st.warning("⚠️ Tidak ada objek terdeteksi di gambar.")

