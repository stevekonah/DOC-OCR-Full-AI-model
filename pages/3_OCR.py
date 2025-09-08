import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd

st.set_page_config(
    page_title="OCR Text Extraction Tool",
    page_icon="📜",
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
    .result-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .emoji {
        font-size: 1.2em;
        margin-right: 0.5rem;
    }
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📜 OCR Text Extraction Tool</h1>', unsafe_allow_html=True)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🌟 Extract Text from Images</div>', unsafe_allow_html=True)
st.markdown("""
This powerful tool extracts text from your images using advanced Optical Character Recognition (OCR) technology.

- 🌐 **Multi-language Support**: Detects and extracts both Arabic & English text
- 🎯 **Precision Detection**: Highlights text with accurate bounding boxes
- 📊 **Structured Data**: Presents extracted text in an organized DataFrame
- 💫 **High Accuracy**: Utilizes advanced AI models for reliable text recognition
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">📋 How to use this tool</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-box"><span class="emoji">1️⃣</span> Click <b>Browse files</b> below to upload your image</div>
<div class="step-box"><span class="emoji">2️⃣</span> Wait a moment while the OCR processes your image</div>
<div class="step-box"><span class="emoji">3️⃣</span> View the detected text with bounding boxes</div>
<div class="step-box"><span class="emoji">4️⃣</span> Explore the extracted text in the structured table</div>
<div class="step-box"><span class="emoji">5️⃣</span> Copy the full extracted text for your use</div>
""", unsafe_allow_html=True)

st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
st.markdown('<h3>📤 Upload Your Image</h3>', unsafe_allow_html=True)
st.markdown('<p>Supported formats: PNG, JPG, JPEG</p>', unsafe_allow_html=True)
image_ocr = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

if image_ocr is not None:
    image_bytes = image_ocr.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with st.spinner('🔍 Processing image and detecting text...'):
        results = reader.readtext(img_rgb)
    
    for (bbox, text, confidence) in results:
        if text.strip() != "":
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 2)
    
    st.markdown('<div class="sub-header">📊 Detection Results</div>', unsafe_allow_html=True)
    st.image(img_rgb, caption="Detected Text in Image", use_container_width=True)
    
    filtered_data = {
        "Text": [text for (_, text, _) in results if text.strip() != ""],
        "Confidence": [f"{confidence:.2%}" for (_, _, confidence) in results if text.strip() != ""],
        "Position": [f"{bbox[0]} to {bbox[2]}" for (bbox, _, _) in results if text.strip() != ""]
    }
    
    df = pd.DataFrame(filtered_data)
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown('<h3>📋 Extracted Text Data</h3>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    all_text = " ".join(df["Text"].tolist())
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown('<h3>📝 Full Extracted Text</h3>', unsafe_allow_html=True)
    st.text_area("Extracted Text", all_text, height=150, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("""
    <h3>✅ Text Extraction Complete!</h3>
    <p>The text in your image has been successfully extracted and organized.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

