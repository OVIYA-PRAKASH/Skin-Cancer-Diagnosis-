"""
Phase 18: Streamlit Web Application.
Implements the interactive clinician dashboard app.py under streamlit_app/.
Features single-page demographics inputs, drag-and-drop image uploads, diagnostic predictions,
side-by-side Grad-CAM visualizations, and printable HTML report compilation.
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

# Custom premium CSS styling overrides (Vibrant Light Gradient Theme, Full BaseWeb Selectbox Fix)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* 1. Global Page Background & Vibrant Multi-Gradient Canvas */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #e6eff5 50%, #f0fdf4 100%) !important;
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
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1280px !important;
        animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 2. Universal Text Contrast & Visibility Overrides */
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
    
    /* 3. Top Header Hero Banner with Rich Cyan-Teal Gradient */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 45%, #0d9488 100%) !important;
        padding: 32px 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 14px 35px -10px rgba(3, 105, 161, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-banner p, .hero-banner div, .hero-banner span {
        color: white !important;
    }
    .hero-title {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        margin: 6px 0 0 0;
        font-size: 15px;
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
    
    /* 4. Streamlit Native Bordered Containers as Gradient Cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7fc 100%) !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05) !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px) !important;
        border-color: #0d9488 !important;
        box-shadow: 0 15px 32px rgba(13, 148, 136, 0.12) !important;
    }
    
    /* Section Title Gradient Text */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(135deg, #0369a1 0%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 10px;
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
    
    /* BaseWeb Popover / Dropdown Menu Items */
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
    
    /* 6. File Uploader Custom Light Gradient Dropzone */
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
    
    /* 8. Medical Action Button with Teal Gradient */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(3, 105, 161, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(3, 105, 161, 0); }
        100% { box-shadow: 0 0 0 0 rgba(3, 105, 161, 0); }
    }
    
    .stButton > button, 
    .stDownloadButton > button, 
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 50%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 18px 32px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 6px 20px rgba(3, 105, 161, 0.3) !important;
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
        box-shadow: 0 12px 28px rgba(3, 105, 161, 0.45) !important;
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
# MAIN USER INTERFACE LAYOUT
# =====================================================================

# 1. Top Hero Banner
st.markdown("""
<div class="hero-banner">
    <div>
        <div class="hero-title">SkinCancerAI Clinical Diagnostic Console</div>
        <div class="hero-subtitle">Multi-Modal Deep Learning System • Dual CNN-Vision Transformer Fusion • Explainable AI</div>
    </div>
    <div class="hero-badge">
        <span>●</span> GPU Accelerated (GTX 1650)
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Adjusted Demographics Card (Consolidated Bordered Container)
with st.container(border=True):
    st.markdown('<div class="section-header">👤 Patient Demographic Parameters</div>', unsafe_allow_html=True)
    col_demo1, col_demo2, col_demo3, col_demo4 = st.columns(4)

    with col_demo1:
        patient_id = st.text_input("Patient Case ID", value="CASE_2026_883")

    with col_demo2:
        age = st.slider("Patient Age (Years)", min_value=0, max_value=100, value=45)

    with col_demo3:
        sex = st.selectbox("Biological Sex", ["Male", "Female", "Unknown"])

    with col_demo4:
        localization = st.selectbox(
            "Anatomical Site",
            [
                "Back", "Abdomen", "Chest", "Face", "Neck", "Scalp", "Trunk",
                "Upper Extremity", "Lower Extremity", "Hand", "Foot", "Genital", "Acral",
                "Ear", "Unknown"
            ]
        )

if not checkpoints_exist:
    st.warning(
        "⚠️ Trained checkpoints not found! Please run the training pipeline to "
        "generate the best model weights and metadata preprocessor pickle."
    )

# 3. Two-Column Workspace Layout (Image Upload & Diagnostic Pipeline)
col_left, col_right = st.columns([1, 1])

with col_left:
    with st.container(border=True):
        st.markdown('<div class="section-header">📷 Dermoscopic Image Input</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload raw skin lesion photograph (JPG, JPEG, PNG format)",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Lesion Photograph Context", use_container_width=True)

with col_right:
    with st.container(border=True):
        st.markdown('<div class="section-header">⚡ Diagnostic Analysis & XAI</div>', unsafe_allow_html=True)
        
        if uploaded_file is None:
            st.info("ℹ️ Please upload a dermoscopic image photograph above to initiate multi-stage validation and AI inference.")
        elif not checkpoints_exist:
            st.error("❌ Diagnostic engine cannot launch because pre-trained weights are missing from checkpoints/.")
        else:
            run_btn = st.button("🚀 Run Multi-Modal Diagnostic Analysis", use_container_width=True)
            
            if run_btn:
                with st.spinner("Executing multi-stage validation and neural forward pass..."):
                    try:
                        engine = load_inference_engine(
                            config_path=config_file,
                            model_path=model_file,
                            preprocessor_path=preprocessor_file,
                            cache_buster=7
                        )
                        
                        temp_img_path = os.path.join(project_root, "temp_lesion.jpg")
                        image.convert("RGB").save(temp_img_path)
                        
                        is_valid, validation_results = engine.run_pipeline_validation(temp_img_path)
                        
                        st.markdown("### 📋 Multi-Stage Validation Pipeline")
                        
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
                            age=float(age),
                            sex=sex.lower(),
                            localization=localization.lower(),
                            patient_id=patient_id,
                            report_filename=f"{patient_id}_clinical_report.html"
                        )
                        
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)

                        img_tensor = engine.image_transform(image.convert("RGB")).unsqueeze(0).to(engine.device)
                        patient_df = pd.DataFrame([{"age": float(age), "sex": sex.lower(), "localization": localization.lower()}])
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
                        
                        if pred_class == "unknown":
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); padding: 24px; border-radius: 18px; margin-bottom: 25px; border: 1.5px solid #fecaca; box-shadow: 0 4px 15px rgba(229, 62, 62, 0.08);">
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
                            <div style="background: linear-gradient(135deg, #e0f2fe 0%, #f0fdf4 100%); padding: 24px; border-radius: 18px; margin-bottom: 25px; border: 1.5px solid #bae6fd; box-shadow: 0 4px 15px rgba(2, 132, 199, 0.08);">
                                <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; color: #0369a1; font-weight: 700; margin-bottom: 6px;">Predicted Diagnosis</div>
                                <div style="font-size: 28px; font-weight: 700; color: #0c4a6e; margin: 4px 0; font-family: 'Outfit', sans-serif;">{disease_info.get('name', pred_class)}</div>
                                <div style="margin-top: 12px; display: flex; align-items: center; gap: 15px;">
                                    <span class="badge {badge_style}">{disease_info.get('severity', 'Benign')}</span>
                                    <span style="font-weight: 600; font-size: 16px; color: #0284c7;">{(confidence_val*100):.2f}% Calibrated Confidence</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("### 🔍 Explainable AI Visualizations")
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(raw_np, caption="Original Dermoscopic Input", use_container_width=True)
                        with col_img2:
                            st.image(blended_np, caption="Grad-CAM CNN Hotspot Overlay", use_container_width=True)

                        st.markdown("### 📊 Probability Distribution")
                        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                        for code, val in sorted_probs:
                            name = engine.report_generator.DISEASE_INFO.get(code, {}).get("name", code)
                            pct = val * 100
                            bar_color = "#ef4444" if code in ["mel", "bcc", "akiec"] else "#10b981" if code in ["nv", "bkl"] else "#6366f1"
                            st.markdown(f"""
                            <div class="probability-label">{name} ({code.upper()})</div>
                            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                                <div style="background-color: #cbd5e1; height: 10px; border-radius: 5px; overflow: hidden; flex-grow: 1; border: 1px solid #94a3b8;">
                                    <div style="width: {pct}%; height: 100%; background: {bar_color}; border-radius: 5px; box-shadow: 0 0 8px {bar_color}40;"></div>
                                </div>
                                <div style="font-weight: 700; width: 65px; text-align: right; font-size: 13px; color: #0f172a; font-family: 'Outfit', sans-serif; margin-left: 10px;">{pct:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("### 📋 Clinical Notes & Guidelines")
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
                                file_name=f"clinical_report_{patient_id}.html",
                                mime="text/html",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"An error occurred during diagnostics processing: {str(e)}")
                        logger.exception("Streamlit inference processing error:")
