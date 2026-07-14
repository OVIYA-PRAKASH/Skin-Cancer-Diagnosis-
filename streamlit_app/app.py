"""
Phase 18: Streamlit Web Application.
Implements the interactive clinician dashboard app.py under streamlit_app/.
Features two-page session navigation: line-by-line input collection on Page 1,
and diagnostic analysis / explainability visualizations on Page 2.
"""

import sys
import os
import logging
import streamlit as st
from PIL import Image
import numpy as np

# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import set_seed
from utils.inference_engine import InferenceEngine
from explainability.gradcam import GradCAM

logger = logging.getLogger("SkinCancerAI.StreamlitApp")

# =====================================================================
# SYSTEM INITIALIZATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="SkinCancerAI Clinician Diagnostics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium CSS styling overrides (Multi-Page Navigation & Line-by-Line Forms)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* 1. Global Page Background & Soft Multi-Gradient Canvas */
    .stApp {
        background: linear-gradient(to right, #f3e7ff, #d4fcff) !important;
        background-attachment: fixed !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Hide default Streamlit banners & force remove sidebar completely */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    
    /* Top Navigation Bar */
    .top-nav {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 14px 24px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 40px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .top-nav a {
        text-decoration: none;
        color: #1e293b !important;
        font-weight: 700;
        font-size: 15px;
        transition: color 0.2s ease;
        letter-spacing: 0.3px;
    }
    .top-nav a:hover {
        color: #0ea5e9 !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1100px !important;
        animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 2. Text Visibility Guarantees Across All Streamlit Elements */
    p, span, label, h1, h2, h3, h4, h5, h6, input, select, option,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    .stSelectbox label, .stTextInput label, .stSlider label, .stFileUploader label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    small {
        color: #334155 !important;
        font-weight: 500 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        color: #0f172a !important;
    }
    
    /* 3. Top Header Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%) !important;
        padding: 32px 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 14px 35px -10px rgba(37, 99, 235, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-banner p, .hero-banner div, .hero-banner span {
        color: white !important;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #ffffff 0%, #bae6fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #e2e8f0 !important;
        font-weight: 400 !important;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #38bdf8 !important;
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    /* 4. Line-by-Line Section Card Wrappers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7fc 100%) !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 18px !important;
        padding: 20px 24px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px) !important;
        border-color: #0d9488 !important;
        box-shadow: 0 12px 28px rgba(13, 148, 136, 0.12) !important;
    }
    
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 18px;
        font-weight: 700;
        background: linear-gradient(135deg, #0369a1 0%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Patient Context Summary Bar on Results Page */
    .patient-summary-bar {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 24px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.15);
    }
    .patient-summary-bar span, .patient-summary-bar strong {
        color: white !important;
    }
    
    /* 5. Complete BaseWeb Selectbox Popover & Dropdown Fix */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1.5px solid #0d9488 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
    }
    div[data-baseweb="select"] svg {
        fill: #0f172a !important;
    }
    div[data-baseweb="select"] input {
        color: #0f172a !important;
    }
    
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #ffffff !important;
        border: 1.5px solid #0d9488 !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.2) !important;
    }
    div[data-baseweb="popover"] *, ul[role="listbox"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    li[role="option"], div[role="option"], ul[role="listbox"] li {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        margin: 2px 4px !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"], ul[role="listbox"] li:hover {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%) !important;
        color: #0369a1 !important;
    }
    
    input[type="text"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 10px !important;
    }
    .stSlider > div [data-baseweb="slider"] {
        background-color: transparent !important;
    }
    
    /* 6. File Uploader Custom Light Dropzone */
    div[data-testid="stFileUploader"], 
    div[data-testid="stFileUploader"] section, 
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"] > div {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
        border: 2px dashed #0d9488 !important;
        border-radius: 16px !important;
        color: #0f172a !important;
    }
    div[data-testid="stFileUploader"] * {
        color: #0f172a !important;
    }
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #0f172a 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }
    
    /* 7. Stepper Status Cards with Gradients */
    .stepper-container {
        display: flex;
        gap: 14px;
        margin: 20px 0 28px 0;
        width: 100%;
    }
    .stepper-card {
        flex: 1;
        background: linear-gradient(135deg, #ffffff 0%, #f0f7fc 100%);
        border: 1.5px solid #cbd5e1;
        border-radius: 14px;
        padding: 16px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    .stepper-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.1);
    }
    .step-num {
        font-size: 10px;
        font-weight: 700;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .step-title {
        font-size: 14px;
        font-weight: 700;
        color: #0f172a !important;
        margin-top: 4px;
    }
    .step-icon {
        position: absolute;
        top: 14px;
        right: 14px;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
    }
    .step-desc {
        font-size: 12px;
        color: #334155 !important;
        margin-top: 8px;
        line-height: 1.4;
    }
    .step-success {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    .step-success .step-icon {
        background-color: #a7f3d0;
        color: #047857 !important;
        border: 1px solid #34d399;
    }
    .step-fail {
        border-left: 4px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    .step-fail .step-icon {
        background-color: #fca5a5;
        color: #991b1b !important;
        border: 1px solid #f87171;
    }
    .step-pending {
        border-left: 4px solid #94a3b8;
        opacity: 0.75;
    }
    .step-pending .step-icon {
        background-color: #e2e8f0;
        color: #475569 !important;
        border: 1px solid #cbd5e1;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .badge-malignant {
        background-color: #fee2e2;
        color: #991b1b !important;
        border: 1px solid #fca5a5;
    }
    .badge-benign {
        background-color: #d1fae5;
        color: #065f46 !important;
        border: 1px solid #6ee7b7;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e !important;
        border: 1px solid #fcd34d;
    }
    
    .guideline-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #f0fdf4 100%);
        border-left: 4px solid #0d9488;
        padding: 22px;
        border-radius: 0 16px 16px 0;
        margin-top: 20px;
        border-top: 1px solid #cbd5e1;
        border-right: 1px solid #cbd5e1;
        border-bottom: 1px solid #cbd5e1;
    }
    .probability-label {
        font-size: 13px;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 6px;
    }
    
    /* 8. Medical Action Button with White Text */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(37, 99, 235, 0); }
        100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
    }
    
    .stButton > button, 
    .stDownloadButton > button, 
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 18px 32px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
        width: 100% !important;
    }
    .stButton > button *, 
    .stDownloadButton > button *, 
    div[data-testid="stDownloadButton"] > button *,
    div[data-testid="stFormSubmitButton"] > button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stButton > button {
        animation: pulse-glow 2.5s infinite !important;
    }
    .stButton > button:hover, 
    .stDownloadButton > button:hover, 
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.45) !important;
        opacity: 0.96 !important;
    }
    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(0px) !important;
    }
</style>
""", unsafe_allow_html=True)


# Cache resource to prevent loading model weights multiple times
@st.cache_resource
def load_inference_engine(config_path: str, model_path: str, preprocessor_path: str, cache_buster: int = 1) -> InferenceEngine:
    """Loads and caches the inference engine."""
    return InferenceEngine(
        config_path=config_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )


# Resolve checkpoint directory paths absolute to the script location
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir, ".."))

model_file = os.path.join(project_root, "checkpoints", "best_model.pth")
preprocessor_file = os.path.join(project_root, "checkpoints", "metadata_preprocessor.pkl")
config_file = os.path.join(project_root, "configs", "default_config.yaml")

checkpoints_exist = os.path.exists(model_file) and os.path.exists(preprocessor_file)

# =====================================================================
# SESSION STATE NAVIGATION INITIALIZATION
# =====================================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "input"

if "patient_id" not in st.session_state:
    st.session_state.patient_id = "CASE_2026_883"

if "age" not in st.session_state:
    st.session_state.age = 45

if "sex" not in st.session_state:
    st.session_state.sex = "Male"

if "localization" not in st.session_state:
    st.session_state.localization = "Back"

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


# =====================================================================
# GLOBAL TOP NAVIGATION BAR
# =====================================================================
st.markdown("""
<div class="top-nav">
    <a href="#">Home</a>
    <a href="#">About</a>
    <a href="#">Contact</a>
    <a href="#">Signup</a>
    <a href="#">Login</a>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGE 1: LINE-BY-LINE CASE INPUT FORM
# =====================================================================
if st.session_state.current_page == "input":
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">SkinCancerAI Clinical Diagnostic Console</div>
            <div class="hero-subtitle">Step 1 of 2: Line-by-Line Patient Case Entry & Dermoscopic Image Upload</div>
        </div>
        <div class="hero-badge">
            <span>●</span> Form Entry Mode
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not checkpoints_exist:
        st.warning(
            "⚠️ Trained checkpoints not found in checkpoints/! Please run the training pipeline first."
        )

    # Line 1: Patient Case ID
    with st.container(border=True):
        st.markdown('<div class="section-header">1. Patient Case Identification</div>', unsafe_allow_html=True)
        patient_id_val = st.text_input("Enter unique medical record / case identifier", value=st.session_state.patient_id)
        st.session_state.patient_id = patient_id_val

    # Line 2: Patient Age
    with st.container(border=True):
        st.markdown('<div class="section-header">2. Patient Age Demographics</div>', unsafe_allow_html=True)
        age_val = st.slider("Select biological age of the patient (Years)", min_value=0, max_value=100, value=int(st.session_state.age))
        st.session_state.age = age_val

    # Line 3: Biological Sex
    with st.container(border=True):
        st.markdown('<div class="section-header">3. Biological Sex Specification</div>', unsafe_allow_html=True)
        sex_idx = ["Male", "Female", "Unknown"].index(st.session_state.sex) if st.session_state.sex in ["Male", "Female", "Unknown"] else 0
        sex_val = st.selectbox("Select patient biological sex category", ["Male", "Female", "Unknown"], index=sex_idx)
        st.session_state.sex = sex_val

    # Line 4: Anatomical Site
    with st.container(border=True):
        st.markdown('<div class="section-header">4. Anatomical Site Localization</div>', unsafe_allow_html=True)
        loc_options = [
            "Back", "Abdomen", "Chest", "Face", "Neck", "Scalp", "Trunk",
            "Upper Extremity", "Lower Extremity", "Hand", "Foot", "Genital", "Acral",
            "Ear", "Unknown"
        ]
        loc_idx = loc_options.index(st.session_state.localization) if st.session_state.localization in loc_options else 0
        loc_val = st.selectbox("Select primary body lesion site", loc_options, index=loc_idx)
        st.session_state.localization = loc_val

    # Line 5: Dermoscopic Image Upload
    with st.container(border=True):
        st.markdown('<div class="section-header">5. Dermoscopic Image Photograph</div>', unsafe_allow_html=True)
        uploaded_file_val = st.file_uploader(
            "Upload raw high-resolution skin lesion photograph (JPG, JPEG, PNG format)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file_val is not None:
            st.session_state.uploaded_file = uploaded_file_val
            image_preview = Image.open(uploaded_file_val)
            st.image(image_preview, caption="Uploaded Lesion Photograph Preview", width=350)

    st.markdown("---")
    
    # Submit Action Button
    submit_btn = st.button("🚀 Submit Case Parameters & View AI Diagnostics", use_container_width=True)
    if submit_btn:
        if st.session_state.uploaded_file is None:
            st.error("❌ Please upload a dermoscopic image in Line 5 above before submitting.")
        elif not checkpoints_exist:
            st.error("❌ Cannot process diagnostics: Model weights missing from checkpoints/.")
        else:
            st.session_state.current_page = "results"
            st.rerun()


# =====================================================================
# PAGE 2: DIAGNOSTIC RESULTS & CLINICAL EXPLAINABILITY
# =====================================================================
elif st.session_state.current_page == "results":
    # Top Navigation Header
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("← Edit Patient Inputs", use_container_width=True):
            st.session_state.current_page = "input"
            st.rerun()

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">SkinCancerAI Clinical Diagnostic Console</div>
            <div class="hero-subtitle">Step 2 of 2: AI Multi-Modal Diagnostics, Grad-CAM Visualizations & Clinical Guidelines</div>
        </div>
        <div class="hero-badge">
            <span>●</span> Results Display
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Patient Context Summary Bar
    st.markdown(f"""
    <div class="patient-summary-bar">
        <div><strong>Patient ID:</strong> <span>{st.session_state.patient_id}</span></div>
        <div><strong>Age:</strong> <span>{st.session_state.age} Years</span></div>
        <div><strong>Sex:</strong> <span>{st.session_state.sex}</span></div>
        <div><strong>Site:</strong> <span>{st.session_state.localization}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Load image
    image = Image.open(st.session_state.uploaded_file)
    
    with st.spinner("Executing multi-stage validation and neural forward pass..."):
        try:
            engine = load_inference_engine(
                config_path=config_file,
                model_path=model_file,
                preprocessor_path=preprocessor_file,
                cache_buster=8
            )
            
            temp_img_path = os.path.join(project_root, "temp_lesion.jpg")
            image.convert("RGB").save(temp_img_path)
            
            is_valid, validation_results = engine.run_pipeline_validation(temp_img_path)
            
            with st.container(border=True):
                st.markdown('<div class="section-header">📋 Multi-Stage Image Validation Pipeline</div>', unsafe_allow_html=True)
                
                val_res = validation_results["image_validation"]
                step1_class = "step-success" if val_res["passed"] else "step-fail"
                step1_icon = "✓" if val_res["passed"] else "✗"
                step1_sub = f"Size: {val_res['metrics'].get('width')}x{val_res['metrics'].get('height')}" if val_res["passed"] else val_res["message"]

                skin_res = validation_results["skin_detection"]
                if skin_res["message"] == "Pending":
                    step2_class = "step-pending"
                    step2_icon = "—"
                    step2_sub = "Pending"
                elif skin_res["passed"]:
                    step2_class = "step-success"
                    step2_icon = "✓"
                    step2_sub = f"Skin Ratio: {skin_res['metrics'].get('skin_ratio', 0.0)*100:.1f}%"
                else:
                    step2_class = "step-fail"
                    step2_icon = "✗"
                    step2_sub = skin_res["message"]

                lesion_res = validation_results["lesion_detection"]
                if lesion_res["message"] == "Pending":
                    step3_class = "step-pending"
                    step3_icon = "—"
                    step3_sub = "Pending"
                elif lesion_res["passed"]:
                    step3_class = "step-success"
                    step3_icon = "✓"
                    step3_sub = f"Lesion Ratio: {lesion_res['metrics'].get('max_lesion_area_ratio', 0.0)*100:.2f}%"
                else:
                    step3_class = "step-fail"
                    step3_icon = "✗"
                    step3_sub = lesion_res["message"]

                step4_class = "step-success" if is_valid else "step-pending"
                step4_icon = "✓" if is_valid else "—"
                step4_sub = "Executed" if is_valid else "Skipped"

                st.markdown(f"""
                <div class="stepper-container">
                    <div class="stepper-card {step1_class}">
                        <div class="step-num">1</div>
                        <div class="step-title">Image Check</div>
                        <div class="step-icon">{step1_icon}</div>
                        <div class="step-desc">{step1_sub}</div>
                    </div>
                    <div class="stepper-card {step2_class}">
                        <div class="step-num">2</div>
                        <div class="step-title">Skin Search</div>
                        <div class="step-icon">{step2_icon}</div>
                        <div class="step-desc">{step2_sub}</div>
                    </div>
                    <div class="stepper-card {step3_class}">
                        <div class="step-num">3</div>
                        <div class="step-title">Lesion Focus</div>
                        <div class="step-icon">{step3_icon}</div>
                        <div class="step-desc">{step3_sub}</div>
                    </div>
                    <div class="stepper-card {step4_class}">
                        <div class="step-num">4</div>
                        <div class="step-title">Inference</div>
                        <div class="step-icon">{step4_icon}</div>
                        <div class="step-desc">{step4_sub}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if not is_valid:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                st.error("❌ Diagnostics halted: The uploaded image failed validation checks. Refer to status cards above.")
                st.stop()

            pred_class, probs, report_path = engine.predict_and_explain(
                image_path=temp_img_path,
                age=float(st.session_state.age),
                sex=st.session_state.sex.lower(),
                localization=st.session_state.localization.lower(),
                patient_id=st.session_state.patient_id,
                report_filename=f"{st.session_state.patient_id}_clinical_report.html"
            )
            
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            img_tensor = engine.image_transform(image.convert("RGB")).unsqueeze(0).to(engine.device)
            patient_df = pd.DataFrame([{"age": float(st.session_state.age), "sex": st.session_state.sex.lower(), "localization": st.session_state.localization.lower()}])
            meta_tensor = engine.preprocessor.transform(patient_df).to(engine.device)
            
            target_layer = engine.model.cnn_extractor.features[-1]
            gradcam = GradCAM(model=engine.model, target_layer=target_layer)
            
            heatmap, pred_idx, base_confidence = gradcam.generate_heatmap(img_tensor, meta_tensor)
            raw_np = np.array(image.convert("RGB").resize((engine.config.data.image_size, engine.config.data.image_size)))
            blended_np = GradCAM.overlay_heatmap(image_np=raw_np, heatmap=heatmap, alpha=0.45)
            gradcam.remove_hooks()

            disease_info = engine.report_generator.DISEASE_INFO.get(pred_class, {})
            is_malignant = "Malignant" in disease_info.get("severity", "")
            is_undetermined = "Undetermined" in disease_info.get("severity", "")
            badge_style = "badge-malignant" if is_malignant else "badge-warning" if is_undetermined else "badge-benign"
            
            st.success("✅ Multi-modal diagnostics compiled!")
            
            confidence_val = probs.get(pred_class, max(probs.values()))
            
            with st.container(border=True):
                if pred_class == "unknown":
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); padding: 24px; border-radius: 18px; margin-bottom: 10px; border: 1.5px solid #fecaca; box-shadow: 0 4px 15px rgba(229, 62, 62, 0.08);">
                        <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #c53030; font-weight: 700; margin-bottom: 6px;">Primary Diagnostic Prediction</div>
                        <div style="font-size: 26px; font-weight: 700; color: #9b1c1c; margin: 4px 0; font-family: 'Outfit', sans-serif;">{disease_info.get('name', 'Unknown Category (Low Confidence)')}</div>
                        <div style="margin-top: 12px; display: flex; align-items: center; gap: 15px;">
                            <span class="badge {badge_style}">{disease_info.get('severity', 'Undetermined')}</span>
                            <span style="font-weight: 600; font-size: 15px; color: #c53030;">{(confidence_val*100):.2f}% Calibrated Confidence (Below Threshold)</span>
                        </div>
                        <p style="margin-top: 18px; font-size: 14px; color: #742a2a; line-height: 1.6; font-style: italic; border-top: 1px solid rgba(229, 62, 62, 0.15); padding-top: 12px;">
                            <strong>System Note:</strong> {disease_info.get('guideline')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #e0f2fe 0%, #f0fdf4 100%); padding: 24px; border-radius: 18px; margin-bottom: 10px; border: 1.5px solid #bae6fd; box-shadow: 0 4px 15px rgba(2, 132, 199, 0.08);">
                        <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; color: #0369a1; font-weight: 700; margin-bottom: 6px;">Predicted Diagnosis</div>
                        <div style="font-size: 28px; font-weight: 700; color: #0c4a6e; margin: 4px 0; font-family: 'Outfit', sans-serif;">{disease_info.get('name', pred_class)}</div>
                        <div style="margin-top: 12px; display: flex; align-items: center; gap: 15px;">
                            <span class="badge {badge_style}">{disease_info.get('severity', 'Benign')}</span>
                            <span style="font-weight: 600; font-size: 16px; color: #0284c7;">{(confidence_val*100):.2f}% Calibrated Confidence</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<div class="section-header">🔍 Explainable AI Visualizations</div>', unsafe_allow_html=True)
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(raw_np, caption="Original Dermoscopic Input", use_container_width=True)
                with col_img2:
                    st.image(blended_np, caption="Grad-CAM CNN Hotspot Overlay", use_container_width=True)

            with st.container(border=True):
                st.markdown('<div class="section-header">📊 Calibrated Probability Distribution</div>', unsafe_allow_html=True)
                
                # Distinct color palette for each disease category
                disease_color_map = {
                    "mel": "#ef4444",     # Melanoma: Crimson Red
                    "nv": "#10b981",      # Melanocytic Nevus: Emerald Green
                    "bcc": "#dc2626",     # Basal Cell Carcinoma: Deep Ruby Red
                    "akiec": "#f97316",   # Actinic Keratoses: Amber Orange
                    "bkl": "#14b8a6",     # Benign Keratosis: Mint Cyan
                    "df": "#6366f1",      # Dermatofibroma: Royal Indigo
                    "vasc": "#a855f7",    # Vascular Lesions: Vivid Purple
                    "unknown": "#64748b"  # Low Confidence / Undetermined: Charcoal Slate
                }
                
                sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                for code, val in sorted_probs:
                    name = engine.report_generator.DISEASE_INFO.get(code, {}).get("name", code)
                    pct = val * 100
                    bar_color = disease_color_map.get(code.lower(), "#3b82f6")
                    st.markdown(f"""
                    <div class="probability-label">{name} ({code.upper()})</div>
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <div style="background-color: #cbd5e1; height: 10px; border-radius: 5px; overflow: hidden; flex-grow: 1; border: 1px solid #94a3b8;">
                            <div style="width: {pct}%; height: 100%; background: {bar_color}; border-radius: 5px; box-shadow: 0 0 8px {bar_color}60;"></div>
                        </div>
                        <div style="font-weight: 700; width: 65px; text-align: right; font-size: 13px; color: #0f172a; font-family: 'Outfit', sans-serif; margin-left: 10px;">{pct:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<div class="section-header">📋 Clinical Notes & Guidelines</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="guideline-box">
                    <p style="margin: 0 0 10px 0; color: #0f172a;"><strong>Pathology Summary:</strong> {disease_info.get('desc')}</p>
                    <p style="margin: 0; color: #0f766e;"><strong>Clinical Recommendation:</strong> {disease_info.get('guideline')}</p>
                </div>
                """, unsafe_allow_html=True)

            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    html_report = f.read()
                
                st.markdown("---")
                st.download_button(
                    label="📥 Download Clinical Diagnostic Report (HTML)",
                    data=html_report,
                    file_name=f"clinical_report_{st.session_state.patient_id}.html",
                    mime="text/html",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"An error occurred during diagnostics processing: {str(e)}")
            logger.exception("Streamlit inference processing error:")
