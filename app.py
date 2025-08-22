import streamlit as st
import numpy as np
from PIL import Image
import os

# ==== PILIH MODE ====
USE_MODEL = "YOLO"  # "CNN" atau "YOLO"

# ==== LOAD MODEL ====
if USE_MODEL == "CNN":
    import tensorflow as tf
    MODEL_PATH = "models/cnn_soybean_rust.keras"
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    class_names = ["Daun Sehat", "Soybean Rust"]

elif USE_MODEL == "YOLO":
    from ultralytics import YOLO
    MODEL_PATH = "models/best.pt"
    yolo_model = YOLO(MODEL_PATH)

# ==== UI ====
st.title("Deteksi Penyakit Daun Kedelai 🌱")
st.write("Upload gambar daun kedelai untuk deteksi penyakit Soybean Rust.")

uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    if USE_MODEL == "CNN":
        # Preprocessing sesuai arsitektur CNN
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        # Prediksi
        prediction = cnn_model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        if class_names[class_id].lower() == "daun sehat":
            st.success(f"✅ Prediksi: {class_names[class_id]} (confidence {confidence:.2f})")
        else:
            st.error(f"⚠️ Prediksi: {class_names[class_id]} (confidence {confidence:.2f})")

    elif USE_MODEL == "YOLO":
        # Jalankan deteksi
        results = yolo_model(image)

        # Tampilkan hasil deteksi dengan bounding box
        results_img = results[0].plot()
        st.image(results_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

        # Ambil info kelas + confidence
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]

                if label.lower() == "daun sehat":
                    st.success(f"✅ Deteksi: {label} (confidence {conf:.2f})")
                else:
                    st.error(f"⚠️ Deteksi: {label} (confidence {conf:.2f})")
