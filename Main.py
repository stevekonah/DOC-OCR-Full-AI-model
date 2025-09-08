import streamlit as st

st.set_page_config(
    page_title="🌟 AI & ML Multi-Tool Project",
    page_icon="🚀",
    layout="wide"
)



st.markdown("""
<style>
/* ---------- Background Gradient ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
}

/* ---------- Card style ---------- */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 0 25px rgba(255, 255, 255, 0.1);
}

/* ---------- Header ---------- */
.main-header {
    font-size: 3rem;
    color: #FFD43B;
    text-align: center;
    margin-bottom: 1rem;
    text-shadow: 0px 2px 8px rgba(0,0,0,0.3);
}

/* ---------- Subheader ---------- */
.sub-header {
    font-size: 1.5rem;
    color: #ffffff;
    margin-bottom: 1rem;
    text-shadow: 0px 1px 3px rgba(0,0,0,0.3);
}

/* ---------- Emoji style ---------- */
.emoji {
    font-size: 1.5rem;
    margin-right: 0.5rem;
}

/* ---------- Dark/Light mode adaptive ---------- */
[data-testid="stAppViewContainer"][style*="background: linear-gradient"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
[data-testid="stAppViewContainer"][style*="background: white"] {
    background: linear-gradient(135deg, #ffffff, #f0f0f0, #e0e0e0);
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 AI & ML Multi-Tool Project</h1>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">🌟 Welcome to the Ultimate AI & ML Playground!</h2>', unsafe_allow_html=True)
st.markdown("""
This project is designed to provide a **full-stack interactive AI & ML experience**. Explore multiple powerful tools, all in one place:

- 🤖 **AI Regression Models**: Train, tune, and predict using Linear Regression, Random Forest, XGBoost, Gradient Boosting, SVR, ElasticNet, and more.
- 🔍 **YOLOv8 Object Detection**: Detect multiple objects in images with high precision and visualize them with bounding boxes.
- 📜 **OCR Text Extraction**: Extract text from images, including English & Arabic, and display results in a structured table.
- 📝 **My Notes App**: Write, save, and manage your notes conveniently with timestamps.
- 📊 **Train Your Model**: Upload your dataset, select regression or classification tasks, choose models, tune hyperparameters, and visualize results interactively.

✨ Features:
- Interactive UI with **dark/light mode compatibility**
- Beautiful backgrounds & backlights for modern visual appeal
- Emoji-enhanced instructions for easy guidance
- Real-time predictions and visualizations

💡 Whether you are a **student, data scientist, or AI enthusiast**, this platform provides everything you need to explore, experiment, and visualize your machine learning projects efficiently.

""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">📌 How to Navigate</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="emoji">1️⃣</div> Select a page from the sidebar corresponding to the tool you want to explore.<br>
<div class="emoji">2️⃣</div> Follow the instructions within each tool for uploading data or images.<br>
<div class="emoji">3️⃣</div> Input necessary parameters or values for prediction or extraction.<br>
<div class="emoji">4️⃣</div> View results, metrics, and visualizations interactively.<br>
<div class="emoji">5️⃣</div> Enjoy the modern interface with adaptive dark/light mode for comfort. 
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">🎯 Why This Project?</h2>', unsafe_allow_html=True)
st.markdown("""
- Combines **regression, classification, object detection, and OCR** in one platform.
- Provides a **hands-on learning experience** for AI and machine learning concepts.
- Ideal for **prototyping and experimenting** with multiple ML models.
- Fully **interactive, visually appealing**, and **user-friendly**.
- Serves as a **comprehensive portfolio project** showcasing modern AI & ML workflows.
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

