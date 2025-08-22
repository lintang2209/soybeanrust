import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Judul aplikasi
st.set_page_config(page_title="Deteksi Soybean Rust", page_icon="🌱", layout="centered")
st.title("🌱 Deteksi Penyakit Daun Kedelai (Soybean Rust)")
st.write("Unggah gambar daun kedelai, lalu model akan mendeteksi apakah **Sehat** atau **Soybean Rust**.")

# Load model YOLO
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload gambar daun kedelai", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang diupload", use_column_width=True)

    # Prediksi
    results = model.predict(image, conf=0.25)

    # Ambil hasil deteksi
    if results and len(results[0].boxes) > 0:
        names = results[0].names
        probs = results[0].probs if hasattr(results[0], "probs") else None

        detected_classes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            detected_classes.append(cls_name)

        st.subheader("📊 Hasil Deteksi")

        if "soybean rust" in detected_classes:
            st.error("🍂 Daun terdeteksi **Soybean Rust (sakit)**!")
        elif "daun sehat" in detected_classes:
            st.success("🌿 Daun terdeteksi **Sehat** ✅")
        else:
            st.warning("⚠️ Tidak ada daun yang terdeteksi atau kelas tidak dikenal.")

        # Tampilkan gambar hasil deteksi (dengan bounding box)
        res_img = results[0].plot()  # gambar numpy array
        st.image(res_img, caption="🔍 Hasil Deteksi", use_column_width=True)
    else:
        st.warning("⚠️ Tidak ada objek terdeteksi di gambar.")
