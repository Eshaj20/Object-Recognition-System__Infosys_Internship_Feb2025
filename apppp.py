import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import base64
import tempfile
import os
import pytube  # For YouTube video processing

# Set Background Image
BACKGROUND_IMAGE_PATH = r"C:\Users\Maggie chinnu KrishS\Downloads\301499e5-a5e9-47b7-a9b8-0a967269f08f.png"
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background(BACKGROUND_IMAGE_PATH)

# Load YOLO Models (Detection and Segmentation)
@st.cache_resource
def load_detection_model():
    try:
        return YOLO("C:/Users/Maggie chinnu KrishS/Downloads/best_yolo (1).pt")
    except Exception as e:
        st.error(f"Error loading detection model: {str(e)}")
        return None

@st.cache_resource
def load_segmentation_model():
    try:
        return YOLO("yolov8n-seg.pt")  # Automatically downloads the segmentation model
    except Exception as e:
        st.error(f"Error loading segmentation model: {str(e)}")
        return None

detection_model = load_detection_model()
segmentation_model = load_segmentation_model()

# Sidebar - Configuration
st.sidebar.title("⚙ Image/Video Config")
source_type = st.sidebar.radio("Select Source", ["Image", "Video", "Webcam", "RTSP", "YouTube"], key="source_select")
task_type = st.sidebar.radio("Select Task", ["Detection", "Segmentation"], key="task_select")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 25, 100, 50, step=5)
st.sidebar.markdown("---")

# Select the appropriate model based on task
model = detection_model if task_type == "Detection" else segmentation_model

# Main Title (White with Dark Shadow)
st.markdown("""
    <h1 style='color: #FFFFFF; text-shadow: 3px 3px 6px #000000, 0 0 10px #000000;'>🎯 YOLOv8 Object Detection & Segmentation Dashboard</h1>
    <hr style='border: 1px solid #FFFFFF;'>
""", unsafe_allow_html=True)

# Image Processing
if source_type == "Image":
    uploaded_file = st.sidebar.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"], key="image_upload")
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>Original Image</p>", unsafe_allow_html=True)
            st.image(original_image, channels="BGR", use_container_width=True)

        with col2:
            with st.spinner(f"Performing {task_type.lower()}..."):
                results = model.predict(source=original_image, conf=confidence_threshold / 100, show=False)
                annotated_image = results[0].plot()  # Plots bounding boxes for detection, masks for segmentation
                st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>{task_type} Result</p>", unsafe_allow_html=True)
                st.image(annotated_image, channels="BGR", use_container_width=True)
            st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>✅ {task_type} Completed!</p>", unsafe_allow_html=True)

# Video Processing
elif source_type == "Video":
    video_file = st.sidebar.file_uploader("📹 Upload a Video", type=["mp4", "avi", "mov"], key="video_upload")
    
    if video_file:
        st.sidebar.markdown("🎞 Choose a video:")
        video_name = st.sidebar.selectbox("Choose a video...", ["Uploaded Video"])
        st.video(video_file)

        # Save uploaded video temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()

        # Add a label for the video output
        st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>{task_type} Video Frames</p>", unsafe_allow_html=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=confidence_threshold / 100, show=False)
            annotated_frame = results[0].plot()  # Plots bounding boxes or masks based on the model
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        os.unlink(temp_video.name)  # Clean up temporary file
        st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>✅ {task_type} Video Processing Completed!</p>", unsafe_allow_html=True)

# Webcam Processing
elif source_type == "Webcam":
    st.markdown("""
        <p style='color: #32CD32; font-size: 18px; font-weight: bold; text-shadow: 1px 1px 2px #000000, 0 0 4px #000000;'>
            🔴 Press 'Start' to begin webcam {task_type.lower()}.
        </p>
    """, unsafe_allow_html=True)
    start_webcam = st.button(f"🎥 Start Webcam {task_type}")

    if start_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        # Add a label for the webcam output
        st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>{task_type} Webcam Frames</p>", unsafe_allow_html=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=confidence_threshold / 100, show=False)
            annotated_frame = results[0].plot()  # Plots bounding boxes or masks based on the model
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>✅ Webcam {task_type} Stopped!</p>", unsafe_allow_html=True)

# RTSP Processing
elif source_type == "RTSP":
    st.markdown("""
        <p style='color: #32CD32; font-size: 18px; font-weight: bold; text-shadow: 1px 1px 2px #000000, 0 0 4px #000000;'>
            🔴 Enter the RTSP URL to start live stream {task_type.lower()}.
        </p>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        input::placeholder {
            color: #32CD32 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    rtsp_url = st.text_input("Enter RTSP URL", placeholder="rtsp://username:password@ip_address:port")
    
    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            st.error("❌ Unable to open RTSP stream. Please check the URL and try again.")
        else:
            stframe = st.empty()
            # Add a label for the RTSP output
            st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>{task_type} RTSP Stream Frames</p>", unsafe_allow_html=True)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame from RTSP stream.")
                    break

                results = model.predict(source=frame, conf=confidence_threshold / 100, show=False)
                annotated_frame = results[0].plot()  # Plots bounding boxes or masks based on the model
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()
            st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>✅ RTSP Stream {task_type} Stopped!</p>", unsafe_allow_html=True)

# YouTube Processing
elif source_type == "YouTube":
    st.markdown("""
        <p style='color: #32CD32; font-size: 18px; font-weight: bold; text-shadow: 1px 1px 2px #000000, 0 0 4px #000000;'>
            🔴 Enter the YouTube video URL to start {task_type.lower()}.
        </p>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        input::placeholder {
            color: #32CD32 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    youtube_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if youtube_url:
        try:
            # Download YouTube video using pytube
            yt = pytube.YouTube(youtube_url)
            stream = yt.streams.filter(file_extension="mp4", progressive=True).first()
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            stream.download(filename=temp_video.name)

            # Process the downloaded video
            cap = cv2.VideoCapture(temp_video.name)
            stframe = st.empty()

            # Add a label for the YouTube output
            st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>{task_type} YouTube Video Frames</p>", unsafe_allow_html=True)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, conf=confidence_threshold / 100, show=False)
                annotated_frame = results[0].plot()  # Plots bounding boxes or masks based on the model
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()
            os.unlink(temp_video.name)  # Clean up temporary file
            st.markdown(f"<p style='color: #CCFF00; font-size: 18px; font-weight: bold; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>✅ YouTube Video {task_type} Completed!</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error processing YouTube video: {str(e)}")

# Footer (White with Dark Shadow)
st.markdown("""
    <div style='color: #FFFFFF; text-align: center; text-shadow: 2px 2px 4px #000000, 0 0 8px #000000;'>
        Developed with ❤ using Streamlit & YOLOv8 | © 2025
    </div>
""", unsafe_allow_html=True)

