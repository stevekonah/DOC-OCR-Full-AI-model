import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        margin-top: 1.5rem;
        border-bottom: 2px solid #FFD43B;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: rgba(75, 139, 190, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4B8BBE;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .step-box {
        background-color: rgba(255, 212, 59, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #FFD43B;
    }
    .uploader-box {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 2px dashed #4B8BBE;
        margin: 1.5rem 0;
    }
    .success-box {
        background-color: rgba(52, 168, 83, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #34A853;
        margin-top: 1rem;
    }
    .emoji {
        font-size: 1.2em;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔍 Object Detection with YOLOv8</h1>', unsafe_allow_html=True)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🌟 What does this model do?</div>', unsafe_allow_html=True)
st.markdown("""
This powerful model utilizes **YOLOv8** (You Only Look Once) to identify and locate objects within your uploaded images.

- 🚀 **Fast & Accurate**: YOLOv8 is one of the fastest and most precise object detection algorithms available
- 📦 **Multiple Objects**: Detects and classifies numerous objects simultaneously
- 🎯 **Precision Bounding**: Draws accurate bounding boxes around detected objects
- 🌈 **Visual Clarity**: Clear visualization with labeled predictions
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">📋 How to use this tool</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-box"><span class="emoji">1️⃣</span> Click <b>Browse files</b> below to upload your image</div>
<div class="step-box"><span class="emoji">2️⃣</span> Wait a moment while the model processes your image</div>
<div class="step-box"><span class="emoji">3️⃣</span> View the detected objects with bounding boxes and labels</div>
""", unsafe_allow_html=True)

st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
st.markdown('<h3>📤 Upload Your Image</h3>', unsafe_allow_html=True)
st.markdown('<p>Supported formats: PNG, JPG, JPEG</p>', unsafe_allow_html=True)
image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if image is not None:
    image_bytes = image.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    model = YOLO('yolov8n.pt')
    
    with st.spinner('🔄 Analyzing your image... Please wait'):
        results = model.predict(image_cv, save=False, show=False)
    
    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    st.markdown('<div class="sub-header">📊 Detection Results</div>', unsafe_allow_html=True)
    st.image(result_image, caption="Detected Objects", use_container_width=True)
    
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
    <h3>✅ Detection Complete!</h3>
    <p>The objects in your image have been successfully identified and highlighted with bounding boxes.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)   
    