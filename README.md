# Deteksi_Objek_Yolov8
python

import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import os

st.set_page_config(page_title="Web Deteksi Objek di Video", layout="centered")

st.title("🎥🚗 Deteksi Objek di Video dengan YOLOv8")

# Upload video
uploaded_video = st.file_uploader("Upload Video (MP4/MOV)", type=["mp4", "mov"])

if uploaded_video is not None:
    # Simpan video sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Load model YOLOv8
    model = YOLO("yolov8n.pt")  # Ganti dengan 'mobil.pt' kalau pakai model custom

    # Load video
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    # Loop frame per frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek di frame
        results = model.predict(source=frame, conf=0.5, save=False, verbose=False)
        result_frame = results[0].plot()  # Tambahkan bounding box

        # Tampilkan frame hasil deteksi di streamlit
        stframe.image(result_frame, channels="BGR", use_column_width=True)

    cap.release()
    os.unlink(tfile.name)

st.write("© 2024 — Aplikasi Deteksi Objek YOLOv8")
