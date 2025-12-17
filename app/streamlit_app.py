import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

from src.data_transforms import val_transforms
from src.utils.image_processing import process_ecg_image

# ──────────────────────────────────────────────────────────────
# Page Configuration
st.set_page_config(
    page_title="CardioScan AI | ECG Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────
# Custom CSS for Modern Medical UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    /* Root Variables */
    :root {
        --primary: #E63946;
        --primary-dark: #C1121F;
        --secondary: #1D3557;
        --accent: #457B9D;
        --light: #F1FAEE;
        --dark: #0D1B2A;
        --success: #2A9D8F;
        --warning: #E9C46A;
        --gradient-primary: linear-gradient(135deg, #E63946 0%, #C1121F 100%);
        --gradient-dark: linear-gradient(135deg, #1D3557 0%, #0D1B2A 100%);
        --gradient-accent: linear-gradient(135deg, #457B9D 0%, #1D3557 100%);
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, #0D1B2A 0%, #1D3557 50%, #0D1B2A 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide Streamlit branding and sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Completely hide the sidebar */
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    .st-emotion-cache-1cypcdb {display: none !important;}
    .st-emotion-cache-vk3wp9 {display: none !important;}
    .st-emotion-cache-1oe5cao {display: none !important;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Custom Header */
    .hero-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(230, 57, 70, 0.15) 0%, rgba(29, 53, 87, 0.3) 100%);
        border-radius: 24px;
        border: 1px solid rgba(230, 57, 70, 0.3);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(230, 57, 70, 0.1) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #E63946 0%, #FF6B6B 50%, #E63946 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        color: #A8DADC;
        letter-spacing: 3px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(29, 53, 87, 0.5);
        border-radius: 16px;
        padding: 8px;
        gap: 8px;
        border: 1px solid rgba(69, 123, 157, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        color: #A8DADC;
        background: transparent;
        border-radius: 12px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(230, 57, 70, 0.2);
        color: #F1FAEE;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #E63946 0%, #C1121F 100%) !important;
        color: #F1FAEE !important;
        box-shadow: 0 4px 15px rgba(230, 57, 70, 0.4);
    }
    
    /* Card Styles */
    .info-card {
        background: linear-gradient(145deg, rgba(29, 53, 87, 0.6) 0%, rgba(13, 27, 42, 0.8) 100%);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(69, 123, 157, 0.3);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        border-color: rgba(230, 57, 70, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(230, 57, 70, 0.15);
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #F1FAEE;
        margin-bottom: 0.8rem;
    }
    
    .card-text {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        color: #A8DADC;
        line-height: 1.7;
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: linear-gradient(145deg, rgba(69, 123, 157, 0.2) 0%, rgba(29, 53, 87, 0.4) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(69, 123, 157, 0.2);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        border-color: #E63946;
        transform: scale(1.02);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }
    
    .feature-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #F1FAEE;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-family: 'Outfit', sans-serif;
        color: #A8DADC;
        font-size: 0.9rem;
    }
    
    /* Stats Section */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 2rem 0;
        padding: 1.5rem;
        background: rgba(230, 57, 70, 0.1);
        border-radius: 16px;
        border: 1px solid rgba(230, 57, 70, 0.2);
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #E63946;
    }
    
    .stat-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.9rem;
        color: #A8DADC;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Steps Section */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 1.5rem;
        padding: 1.5rem;
        background: rgba(29, 53, 87, 0.4);
        border-radius: 16px;
        margin-bottom: 1rem;
        border-left: 4px solid #E63946;
        transition: all 0.3s ease;
    }
    
    .step-item:hover {
        background: rgba(230, 57, 70, 0.1);
        transform: translateX(8px);
    }
    
    .step-number {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #E63946;
        background: rgba(230, 57, 70, 0.2);
        width: 50px;
        height: 50px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    
    .step-content h4 {
        font-family: 'Outfit', sans-serif;
        color: #F1FAEE;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .step-content p {
        font-family: 'Outfit', sans-serif;
        color: #A8DADC;
        font-size: 0.95rem;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background: transparent !important;
    }
    
    .stFileUploader label {
        color: #A8DADC !important;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: rgba(29, 53, 87, 0.4);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(69, 123, 157, 0.3);
    }
    
    .stRadio label {
        color: #A8DADC !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(29, 53, 87, 0.6) !important;
        border-radius: 12px !important;
        color: #F1FAEE !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        border: 1px solid rgba(69, 123, 157, 0.3) !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #E63946 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(13, 27, 42, 0.8) !important;
        border: 1px solid rgba(69, 123, 157, 0.2) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(145deg, rgba(42, 157, 143, 0.2) 0%, rgba(29, 53, 87, 0.4) 100%);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid #2A9D8F;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .result-card.warning {
        background: linear-gradient(145deg, rgba(230, 57, 70, 0.2) 0%, rgba(29, 53, 87, 0.4) 100%);
        border-color: #E63946;
    }
    
    .result-prediction {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #F1FAEE;
        margin-bottom: 0.5rem;
    }
    
    .result-confidence {
        font-family: 'Space Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: #2A9D8F;
    }
    
    .result-card.warning .result-confidence {
        color: #E63946;
    }
    
    /* Progress Bars for Probabilities */
    .prob-bar-container {
        background: rgba(29, 53, 87, 0.6);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .prob-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-family: 'Outfit', sans-serif;
        color: #F1FAEE;
    }
    
    .prob-bar {
        height: 8px;
        background: rgba(69, 123, 157, 0.3);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Success/Info Messages */
    .stSuccess, .stInfo {
        background: rgba(42, 157, 143, 0.2) !important;
        border: 1px solid #2A9D8F !important;
        border-radius: 12px !important;
        color: #F1FAEE !important;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(69, 123, 157, 0.3);
    }
    
    .custom-footer a {
        color: #E63946;
        text-decoration: none;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .custom-footer a:hover {
        color: #FF6B6B;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #F1FAEE !important;
    }
    
    p, li {
        font-family: 'Outfit', sans-serif !important;
        color: #A8DADC !important;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0 1.5rem 0;
    }
    
    .section-header-icon {
        font-size: 2rem;
        background: linear-gradient(135deg, #E63946 0%, #C1121F 100%);
        width: 60px;
        height: 60px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .section-header-text {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #F1FAEE;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.6s ease forwards;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(69, 123, 157, 0.3);
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* Hide default header decorations */
    .stDeployButton {display: none;}
    
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">CardioScan AI</div>
    <div class="hero-subtitle">ECG Analysis & Heart Disease Detection</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Top Tabs for navigation
tab_home, tab_ecg = st.tabs(["🏠 Home", "🩺 ECG Analysis"])

# ──────────────────────────────────────────────────────────────
# Home Tab
with tab_home:
    # Stats Section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-value">4</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">12</div>
            <div class="stat-label">ECG Leads Analyzed</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">ResNet-18</div>
            <div class="stat-label">AI Architecture</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">&lt;5s</div>
            <div class="stat-label">Analysis Time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # About Section
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🫀</span>
        <div class="card-title">About CardioScan AI</div>
        <div class="card-text">
            CardioScan AI is a cutting-edge medical imaging application that leverages deep learning 
            to analyze electrocardiogram (ECG) images and detect potential cardiac abnormalities. 
            Our AI model is trained to recognize patterns associated with common heart conditions, 
            providing rapid preliminary screening to assist healthcare professionals.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-item">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Deep Learning</div>
            <div class="feature-desc">Powered by fine-tuned ResNet-18 neural network</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Instant Results</div>
            <div class="feature-desc">Get predictions in under 5 seconds</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Signal Analysis</div>
            <div class="feature-desc">12-lead ECG extraction & visualization</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detectable Conditions
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🩺</span>
        <div class="card-title">Detectable Conditions</div>
        <div class="card-text">Our AI model can identify the following cardiac conditions:</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon">💚</div>
            <div class="feature-title">Normal</div>
            <div class="feature-desc">Healthy heart rhythm</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon">💛</div>
            <div class="feature-title">Abnormal Heartbeat</div>
            <div class="feature-desc">Irregular rhythm patterns</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon">🧡</div>
            <div class="feature-title">History of MI</div>
            <div class="feature-desc">Past myocardial infarction</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon">❤️</div>
            <div class="feature-title">Myocardial Infarction</div>
            <div class="feature-desc">Active heart attack signs</div>
        </div>
        """, unsafe_allow_html=True)
    
    # How to Use
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📋</div>
        <div class="section-header-text">How to Use</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-item">
        <div class="step-number">01</div>
        <div class="step-content">
            <h4>Navigate to ECG Analysis</h4>
            <p>Click on the "ECG Analysis" tab above to access the analysis tool</p>
        </div>
    </div>
    <div class="step-item">
        <div class="step-number">02</div>
        <div class="step-content">
            <h4>Upload Your ECG</h4>
            <p>Upload an ECG image (PNG, JPG, JPEG) or capture one using your camera</p>
        </div>
    </div>
    <div class="step-item">
        <div class="step-number">03</div>
        <div class="step-content">
            <h4>Explore Processing Steps</h4>
            <p>View grayscale conversion, 12-lead extraction, and 1D signal waveforms</p>
        </div>
    </div>
    <div class="step-item">
        <div class="step-number">04</div>
        <div class="step-content">
            <h4>Get AI Prediction</h4>
            <p>Receive instant AI-powered diagnosis with confidence scores</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="info-card" style="border-color: rgba(233, 196, 106, 0.5); background: linear-gradient(145deg, rgba(233, 196, 106, 0.1) 0%, rgba(29, 53, 87, 0.4) 100%);">
        <span class="card-icon">⚠️</span>
        <div class="card-title">Medical Disclaimer</div>
        <div class="card-text">
            This tool is designed for educational and preliminary screening purposes only. 
            It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult a qualified healthcare provider for accurate diagnosis and treatment recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# ECG Disease Detection Tab
with tab_ecg:
    @st.cache_resource
    def load_model():
        m = models.resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, 4)
        m.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
        m.eval()
        return m

    class_names = ['Abnormal Heartbeat', 'History of MI', 'Myocardial Infarction', 'Normal']
    class_colors = ['#E9C46A', '#F4A261', '#E63946', '#2A9D8F']
    model = load_model()

    # Upload Section Header
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📤</div>
        <div class="section-header-text">Upload ECG Image</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload method selection
    upload_method = st.radio(
        "Select input method:",
        ("📁 Upload from Computer", "📷 Capture from Camera"),
        horizontal=True
    )

    uploaded_file = None
    
    if upload_method == "📁 Upload from Computer":
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload an ECG image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: PNG, JPG, JPEG"
        )
    elif upload_method == "📷 Capture from Camera":
        uploaded_file = st.camera_input("Capture an ECG image using your camera")

    # Continue if file is uploaded or captured
    if uploaded_file:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🖼️</div>
            <div class="section-header-text">Uploaded ECG</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(uploaded_file, caption="Original ECG Image", use_container_width=True)

        # Process image
        with st.spinner("🔄 Processing ECG image..."):
            df, gray, leads = process_ecg_image(uploaded_file)

        # Processing Steps
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">⚙️</div>
            <div class="section-header-text">Processing Steps</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🖤 Step 1: Grayscale Conversion", expanded=False):
            st.markdown("""
            <div class="card-text" style="margin-bottom: 1rem;">
                Converting to grayscale removes color information and focuses on the ECG waveform intensity patterns.
            </div>
            """, unsafe_allow_html=True)
            st.image(gray, caption="Grayscale ECG", use_container_width=True, clamp=True, channels="GRAY")

        with st.expander("🧩 Step 2: 12-Lead Extraction", expanded=False):
            st.markdown("""
            <div class="card-text" style="margin-bottom: 1rem;">
                The ECG is divided into 12 standard lead regions for comprehensive cardiac analysis.
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(4)
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            for i, img in enumerate(leads):
                with cols[i % 4]:
                    st.image(img, caption=f"Lead {lead_names[i] if i < len(lead_names) else i+1}", use_container_width=True, clamp=True, channels="GRAY")

        with st.expander("📈 Step 3: 1D Signal Extraction", expanded=False):
            st.markdown("""
            <div class="card-text" style="margin-bottom: 1rem;">
                Waveforms are extracted as normalized 1D signals for detailed signal analysis.
            </div>
            """, unsafe_allow_html=True)
            
            # Custom styled matplotlib plot
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#0D1B2A')
            ax.set_facecolor('#0D1B2A')
            
            colors = plt.cm.rainbow(np.linspace(0, 1, 12))
            for i in range(12):
                ax.plot(df[f"Lead_{i+1}"], label=f"Lead {i+1}", color=colors[i], alpha=0.8, linewidth=1.2)
            
            ax.legend(ncol=6, fontsize="small", loc='upper center', bbox_to_anchor=(0.5, -0.1))
            ax.set_xlabel("Time", color='#A8DADC', fontsize=11)
            ax.set_ylabel("Normalized Amplitude", color='#A8DADC', fontsize=11)
            ax.tick_params(colors='#A8DADC')
            ax.spines['bottom'].set_color('#457B9D')
            ax.spines['left'].set_color('#457B9D')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, color='#457B9D')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with st.expander("🧮 Step 4: Signal Data Table", expanded=False):
            st.markdown("""
            <div class="card-text" style="margin-bottom: 1rem;">
                Tabular view of extracted signal values for each lead.
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(20), use_container_width=True)

        # Prediction Section
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🧠</div>
            <div class="section-header-text">AI Diagnosis</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔬 Analyzing ECG with AI..."):
            img = Image.open(uploaded_file).convert("RGB")
            tensor = val_transforms(img).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out[0], dim=0).detach().numpy()

        idx = np.argmax(probs)
        conf = probs[idx] * 100
        
        # Determine if result is concerning
        is_warning = class_names[idx] != 'Normal'
        
        # Result Card
        result_class = "warning" if is_warning else ""
        st.markdown(f"""
        <div class="result-card {result_class}">
            <div style="font-size: 1rem; color: #A8DADC; margin-bottom: 0.5rem; font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: 2px;">
                Prediction Result
            </div>
            <div class="result-prediction">{class_names[idx]}</div>
            <div class="result-confidence">{conf:.1f}%</div>
            <div style="font-size: 0.9rem; color: #A8DADC; margin-top: 0.5rem;">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability Breakdown
        with st.expander("📊 Detailed Probability Breakdown", expanded=True):
            for i, name in enumerate(class_names):
                prob_pct = probs[i] * 100
                bar_color = class_colors[i]
                st.markdown(f"""
                <div class="prob-bar-container">
                    <div class="prob-label">
                        <span>{name}</span>
                        <span style="color: {bar_color}; font-family: 'Space Mono', monospace;">{prob_pct:.1f}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: {prob_pct}%; background: {bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Recommendation based on result
        if is_warning:
            st.markdown("""
            <div class="info-card" style="border-color: #E63946; background: linear-gradient(145deg, rgba(230, 57, 70, 0.2) 0%, rgba(29, 53, 87, 0.4) 100%);">
                <span class="card-icon">🏥</span>
                <div class="card-title">Medical Attention Recommended</div>
                <div class="card-text">
                    The AI has detected potential cardiac abnormalities in this ECG. 
                    Please consult a cardiologist or healthcare professional for proper evaluation and diagnosis.
                    This is an AI-assisted screening tool and should not replace professional medical advice.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card" style="border-color: #2A9D8F; background: linear-gradient(145deg, rgba(42, 157, 143, 0.2) 0%, rgba(29, 53, 87, 0.4) 100%);">
                <span class="card-icon">✅</span>
                <div class="card-title">Normal ECG Detected</div>
                <div class="card-text">
                    The AI analysis suggests a normal heart rhythm pattern. 
                    However, for comprehensive cardiac health assessment, regular check-ups with a healthcare provider are still recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Footer
st.markdown("""
<div class="custom-footer">
    <p style="color: #A8DADC; font-family: 'Outfit', sans-serif; margin-bottom: 0.5rem;">
        Built with ❤️ using Streamlit & PyTorch
    </p>
    <a href="https://github.com/harishkumbarSs/Cardiovascular_Disease_Detection" target="_blank">
        View on GitHub →
    </a>
</div>
""", unsafe_allow_html=True)
