import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# Path model
CNN_MODEL_PATH = "models/cnn_soybean_rust.keras"
YOLO_MODEL_PATH = "models/best.pt"

# Load models
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

# Class names
CLASS_NAMES = ["Daun Sehat", "Soybean Rust"]

st.title("📷 Deteksi Penyakit Daun Kedelai (Pipeline CNN → YOLOv8)")

uploaded_file = st.file_uploader("Unggah gambar daun kedelai", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # ==== Tahap 1: Prediksi CNN ====
    img_resized = image.resize((224, 224))  # sesuaikan ukuran input CNN
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    cnn_pred = cnn_model.predict(img_array)
    cnn_class_idx = np.argmax(cnn_pred)
    cnn_class = CLASS_NAMES[cnn_class_idx]
    cnn_conf = np.max(cnn_pred)

    st.subheader("Hasil Prediksi CNN")
    st.write(f"Prediksi: **{cnn_class}**")
    st.write(f"Probabilitas: {cnn_conf*100:.2f}%")

    # ==== Tahap 2: Prediksi YOLO hanya jika CNN prediksi Soybean Rust ====
    if cnn_class == "Soybean Rust":
        st.subheader("Hasil Prediksi YOLOv8")
        yolo_results = yolo_model.predict(
            np.array(image),
            imgsz=640,
            conf=0.45,  # dinaikkan
            iou=0.65,   # NMS lebih ketat
            verbose=False
        )

        # Ambil hasil pertama
        res = yolo_results[0]
        im_h, im_w = res.orig_img.shape[:2]

        # Filter area bbox
        filtered_boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area = (w * h) / (im_w * im_h)
            conf_score = float(b.conf[0])

            # Terima hanya jika area di 1%–40% dari gambar dan confidence ≥ 0.45
            if 0.01 <= area <= 0.40 and conf_score >= 0.45:
                filtered_boxes.append(b)

        # Ganti prediksi asli dengan hasil filter
        if filtered_boxes:
            import torch
            res.boxes = type(res.boxes)(torch.stack([b.data[0] for b in filtered_boxes]))
        else:
            st.warning("Tidak ada lesi terdeteksi sesuai kriteria filter.")

        # Tampilkan hasil deteksi
        yolo_img = res.plot()
        st.image(yolo_img, caption="Hasil deteksi YOLOv8 (difilter)", use_column_width=True)
    else:
        st.success("Daun terdeteksi SEHAT oleh CNN, YOLO tidak dijalankan.")
