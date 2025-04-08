import streamlit as st
from PIL import Image
import cv2
import os
import tempfile
from pytube import YouTube
from ultralytics import YOLO
import time

# ----------------- Background Setup -----------------
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        encoded = img_bytes.encode("base64") if hasattr(img_bytes, "encode") else img_bytes
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Background image not found. Using default theme.")

# ----------------- Run YOLOv8 on Media -----------------
def run_yolo(source, task):
    model = YOLO("yolov8n.pt")
    stframe = st.empty()

    if source.startswith("http") or source.startswith("rtsp"):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if task == "Detection":
            results = model(frame, conf=0.4)[0]
            annotated_frame = results.plot()
        elif task == "Segmentation":
            model = YOLO("yolov8n-seg.pt")
            results = model(frame, conf=0.4)[0]
            annotated_frame = results.plot()
        else:
            st.error("Invalid task selected.")
            break
        stframe.image(annotated_frame, channels="BGR")
    cap.release()

# ----------------- App Layout -----------------
st.set_page_config(page_title="YOLOv8 Dashboard", layout="centered")
set_background("background.png")

st.title("🔍 YOLOv8 Dashboard – Object Detection & Segmentation")
st.markdown("Supports Image, Video, Webcam, RTSP, and YouTube 📹")

task = st.radio("Choose Task", ["Detection", "Segmentation"])
source_type = st.selectbox("Select Input Source", ["Image", "Video", "Webcam", "RTSP", "YouTube"])

# ----------------- Image -----------------
if source_type == "Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run YOLO"):
            model = YOLO("yolov8n.pt" if task == "Detection" else "yolov8n-seg.pt")
            results = model(image)
            res_plotted = results[0].plot()
            st.image(res_plotted, channels="BGR")

# ----------------- Video -----------------
elif source_type == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        if st.button("Run YOLO"):
            run_yolo(tfile.name, task)

# ----------------- Webcam -----------------
elif source_type == "Webcam":
    if st.button("Start Webcam"):
        run_yolo(0, task)

# ----------------- RTSP -----------------
elif source_type == "RTSP":
    rtsp_url = st.text_input("Enter RTSP Stream URL")
    if st.button("Run Stream"):
        run_yolo(rtsp_url, task)

# ----------------- YouTube -----------------
elif source_type == "YouTube":
    yt_url = st.text_input("Enter YouTube Video URL")
    if st.button("Download & Run YOLO"):
        try:
            st.info("Downloading YouTube video...")
            yt = YouTube(yt_url)
            stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
            out_path = stream.download(filename="yt_video.mp4")
            st.success("Download complete!")
            run_yolo(out_path, task)
        except Exception as e:
            st.error(f"Failed to download video: {e}")
