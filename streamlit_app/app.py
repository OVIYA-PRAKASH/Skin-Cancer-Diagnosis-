"""
Phase 18: Streamlit Web Application (Formal Enterprise Medical Interface).
Implements the interactive clinician dashboard app.py under streamlit_app/.
Features two-page session navigation: line-by-line input collection on Page 1,
and diagnostic analysis / explainability visualizations on Page 2.
Strictly formal medical color palette (Navy/Slate/Blue) and text-only presentation (zero emojis/icons).
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
# SYSTEM INITIALIZATION & FORMAL STYLING
# =====================================================================
st.set_page_config(
    page_title="SkinCancerAI Clinical Diagnostic Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Formal Medical CSS Styling (Strictly Text-Only, Slate & Deep Navy Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* 1. Global Page Canvas - Neutral Light Slate */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }
    
    /* Hide default Streamlit headers & sidebar completely */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1120px !important;
    }
    
    /* 2. Strict Text Visibility & Typography Hierarchy */
    p, span, label, h1, h2, h3, h4, h5, h6, input, select, option,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    .stSelectbox label, .stTextInput label, .stSlider label, .stFileUploader label {
        color: #0f172a !important;
        font-weight: 500 !important;
    }
    
    small {
        color: #475569 !important;
        font-weight: 400 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #0f172a !important;
        letter-spacing: -0.3px;
    }
    
    /* 3. Formal Executive Header Banner */
    .hero-banner {
        background-color: #0f172a !important;
        padding: 28px 36px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 24px;
        border: 1px solid #1e293b;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-banner p, .hero-banner div, .hero-banner span {
        color: #ffffff !important;
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff !important;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    .hero-subtitle {
        margin: 6px 0 0 0;
        font-size: 13px;
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }
    .hero-badge {
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #cbd5e1 !important;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* 4. Formal Card Sections */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
    }
    
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e2e8f0;
        letter-spacing: -0.2px;
    }
    
    /* 5. Patient Context Summary Bar */
    .patient-summary-bar {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 14px 22px;
        margin-bottom: 20px;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #334155;
    }
    .patient-summary-bar span, .patient-summary-bar strong {
        color: #f8fafc !important;
        font-size: 13px;
    }
    
    /* 6. Form Inputs & Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
    }
    div[data-baseweb="select"] svg {
        fill: #475569 !important;
    }
    div[data-baseweb="select"] input {
        color: #0f172a !important;
    }
    
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08) !important;
    }
    div[data-baseweb="popover"] *, ul[role="listbox"] * {
        color: #0f172a !important;
        font-weight: 500 !important;
    }
    li[role="option"], div[role="option"], ul[role="listbox"] li {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        margin: 2px !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"], ul[role="listbox"] li:hover {
        background-color: #f1f5f9 !important;
        color: #1e40af !important;
    }
    
    input[type="text"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    .stSlider > div [data-baseweb="slider"] {
        background-color: transparent !important;
    }
    
    /* 7. File Uploader - Formal Dashed Dropzone */
    div[data-testid="stFileUploader"], 
    div[data-testid="stFileUploader"] section, 
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"] > div {
        background-color: #ffffff !important;
        border: 1.5px dashed #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
    }
    div[data-testid="stFileUploader"] * {
        color: #0f172a !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #1e40af !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 6px 14px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploader"] button * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploader"] button::before {
        content: "Select File" !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    
    /* 8. Stepper Status Cards */
    .stepper-container {
        display: flex;
        gap: 12px;
        margin: 16px 0 20px 0;
        width: 100%;
    }
    .stepper-card {
        flex: 1;
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 14px;
        position: relative;
    }
    .step-num {
        font-size: 11px;
        font-weight: 700;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .step-title {
        font-size: 14px;
        font-weight: 700;
        color: #0f172a !important;
        margin-top: 2px;
    }
    .step-status {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        margin-top: 6px;
        letter-spacing: 0.5px;
    }
    .step-desc {
        font-size: 12px;
        color: #475569 !important;
        margin-top: 4px;
        line-height: 1.3;
    }
    .step-success {
        border-left: 3px solid #16a34a;
    }
    .step-success .step-status {
        color: #16a34a !important;
    }
    .step-fail {
        border-left: 3px solid #dc2626;
    }
    .step-fail .step-status {
        color: #dc2626 !important;
    }
    .step-pending {
        border-left: 3px solid #94a3b8;
    }
    .step-pending .step-status {
        color: #64748b !important;
    }
    
    /* 9. Formal Disease Severity Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-malignant {
        background-color: #fef2f2;
        color: #991b1b !important;
        border: 1px solid #fecaca;
    }
    .badge-benign {
        background-color: #f0fdf4;
        color: #166534 !important;
        border: 1px solid #bbf7d0;
    }
    .badge-warning {
        background-color: #fffbeeb;
        color: #92400e !important;
        border: 1px solid #fef3c7;
    }
    
    .guideline-box {
        background-color: #f8fafc;
        border-left: 3px solid #1e40af;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 14px;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    .probability-label {
        font-size: 13px;
        font-weight: 600;
        color: #0f172a !important;
        margin-bottom: 4px;
    }
    
    /* 10. Formal Medical Action Buttons */
    .stButton > button, 
    .stDownloadButton > button, 
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #1e40af !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: background-color 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(30, 64, 175, 0.15) !important;
        width: 100% !important;
    }
    .stButton > button *, 
    .stDownloadButton > button *, 
    div[data-testid="stDownloadButton"] > button *,
    div[data-testid="stFormSubmitButton"] > button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover, 
    .stDownloadButton > button:hover, 
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #1e3a8a !important;
        box-shadow: 0 4px 8px rgba(30, 58, 138, 0.25) !important;
    }
    .stButton > button:active,
    .stDownloadButton > button:active {
        background-color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)


# Cache resource to prevent loading model weights multiple times
@st.cache_resource
def load_inference_engine(config_path: str, model_path: str, preprocessor_path: str, cache_buster: int = 2) -> InferenceEngine:
    """Loads and caches the inference engine."""
    return InferenceEngine(
        config_path=config_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )


# Resolve checkpoint directory paths absolute to script location
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
# PAGE 1: FORMAL CASE INPUT FORM
# =====================================================================
if st.session_state.current_page == "input":
    # Formal Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">SkinCancerAI Clinical Diagnostic Console</div>
            <div class="hero-subtitle">Step 1 of 2: Patient Case Entry and Dermoscopic Image Upload</div>
        </div>
        <div class="hero-badge">
            Form Entry Mode
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not checkpoints_exist:
        st.warning(
            "Trained checkpoints not found in checkpoints/ folder. Please run the training pipeline first."
        )

    # Section 1: Patient Case ID
    with st.container(border=True):
        st.markdown('<div class="section-header">Section 1: Patient Case Identification</div>', unsafe_allow_html=True)
        patient_id_val = st.text_input("Medical record or case identifier", value=st.session_state.patient_id)
        st.session_state.patient_id = patient_id_val

    # Section 2: Patient Age
    with st.container(border=True):
        st.markdown('<div class="section-header">Section 2: Patient Age Demographics</div>', unsafe_allow_html=True)
        age_val = st.slider("Patient biological age (Years)", min_value=0, max_value=100, value=int(st.session_state.age))
        st.session_state.age = age_val

    # Section 3: Biological Sex
    with st.container(border=True):
        st.markdown('<div class="section-header">Section 3: Biological Sex Specification</div>', unsafe_allow_html=True)
        sex_idx = ["Male", "Female", "Unknown"].index(st.session_state.sex) if st.session_state.sex in ["Male", "Female", "Unknown"] else 0
        sex_val = st.selectbox("Patient biological sex category", ["Male", "Female", "Unknown"], index=sex_idx)
        st.session_state.sex = sex_val

    # Section 4: Anatomical Site
    with st.container(border=True):
        st.markdown('<div class="section-header">Section 4: Anatomical Site Localization</div>', unsafe_allow_html=True)
        loc_options = [
            "Back", "Abdomen", "Chest", "Face", "Neck", "Scalp", "Trunk",
            "Upper Extremity", "Lower Extremity", "Hand", "Foot", "Genital", "Acral",
            "Ear", "Unknown"
        ]
        loc_idx = loc_options.index(st.session_state.localization) if st.session_state.localization in loc_options else 0
        loc_val = st.selectbox("Primary body lesion site", loc_options, index=loc_idx)
        st.session_state.localization = loc_val

    # Section 5: Dermoscopic Image Upload
    with st.container(border=True):
        st.markdown('<div class="section-header">Section 5: Dermoscopic Image Upload</div>', unsafe_allow_html=True)
        uploaded_file_val = st.file_uploader(
            "Upload high-resolution skin lesion photograph (JPG, JPEG, PNG format)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file_val is not None:
            st.session_state.uploaded_file = uploaded_file_val
            image_preview = Image.open(uploaded_file_val)
            st.image(image_preview, caption="Uploaded Lesion Photograph Preview", width=320)

    st.markdown("---")
    
    # Submit Action Button (Text-Only)
    submit_btn = st.button("SUBMIT CASE PARAMETERS FOR DIAGNOSTIC ANALYSIS", use_container_width=True)
    if submit_btn:
        if st.session_state.uploaded_file is None:
            st.error("Error: Please upload a dermoscopic image in Section 5 above before submitting.")
        elif not checkpoints_exist:
            st.error("Error: Cannot process diagnostics. Model weights missing from checkpoints/ folder.")
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
        if st.button("EDIT PATIENT INPUTS", use_container_width=True):
            st.session_state.current_page = "input"
            st.rerun()

    # Formal Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">SkinCancerAI Clinical Diagnostic Console</div>
            <div class="hero-subtitle">Step 2 of 2: Multi-Modal AI Diagnostics, Grad-CAM Visualizations and Clinical Guidelines</div>
        </div>
        <div class="hero-badge">
            Diagnostic Analysis Mode
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
    
    with st.spinner("Executing multi-stage validation and neural network inference..."):
        try:
            engine = load_inference_engine(
                config_path=config_file,
                model_path=model_file,
                preprocessor_path=preprocessor_file,
                cache_buster=9
            )
            
            temp_img_path = os.path.join(project_root, "temp_lesion.jpg")
            image.convert("RGB").save(temp_img_path)
            
            is_valid, validation_results = engine.run_pipeline_validation(temp_img_path)
            
            with st.container(border=True):
                st.markdown('<div class="section-header">Multi-Stage Image Validation Pipeline</div>', unsafe_allow_html=True)
                
                val_res = validation_results["image_validation"]
                step1_class = "step-success" if val_res["passed"] else "step-fail"
                step1_status = "PASSED" if val_res["passed"] else "FAILED"
                step1_sub = f"Size: {val_res['metrics'].get('width')}x{val_res['metrics'].get('height')}" if val_res["passed"] else val_res["message"]

                skin_res = validation_results["skin_detection"]
                if skin_res["message"] == "Pending":
                    step2_class = "step-pending"
                    step2_status = "PENDING"
                    step2_sub = "Pending"
                elif skin_res["passed"]:
                    step2_class = "step-success"
                    step2_status = "PASSED"
                    step2_sub = f"Skin Ratio: {skin_res['metrics'].get('skin_ratio', 0.0)*100:.1f}%"
                else:
                    step2_class = "step-fail"
                    step2_status = "FAILED"
                    step2_sub = skin_res["message"]

                lesion_res = validation_results["lesion_detection"]
                if lesion_res["message"] == "Pending":
                    step3_class = "step-pending"
                    step3_status = "PENDING"
                    step3_sub = "Pending"
                elif lesion_res["passed"]:
                    step3_class = "step-success"
                    step3_status = "PASSED"
                    step3_sub = f"Lesion Ratio: {lesion_res['metrics'].get('max_lesion_area_ratio', 0.0)*100:.2f}%"
                else:
                    step3_class = "step-fail"
                    step3_status = "FAILED"
                    step3_sub = lesion_res["message"]

                step4_class = "step-success" if is_valid else "step-pending"
                step4_status = "PASSED" if is_valid else "SKIPPED"
                step4_sub = "Executed" if is_valid else "Skipped"

                st.markdown(f"""
                <div class="stepper-container">
                    <div class="stepper-card {step1_class}">
                        <div class="step-num">Step 1</div>
                        <div class="step-title">Image Validation</div>
                        <div class="step-status">{step1_status}</div>
                        <div class="step-desc">{step1_sub}</div>
                    </div>
                    <div class="stepper-card {step2_class}">
                        <div class="step-num">Step 2</div>
                        <div class="step-title">Skin Detection</div>
                        <div class="step-status">{step2_status}</div>
                        <div class="step-desc">{step2_sub}</div>
                    </div>
                    <div class="stepper-card {step3_class}">
                        <div class="step-num">Step 3</div>
                        <div class="step-title">Lesion Search</div>
                        <div class="step-status">{step3_status}</div>
                        <div class="step-desc">{step3_sub}</div>
                    </div>
                    <div class="stepper-card {step4_class}">
                        <div class="step-num">Step 4</div>
                        <div class="step-title">Neural Inference</div>
                        <div class="step-status">{step4_status}</div>
                        <div class="step-desc">{step4_sub}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if not is_valid:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                st.error("Diagnostics Halted: The uploaded image failed validation checks. Refer to pipeline status steps above.")
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
            
            st.info("Multi-modal diagnostic inference complete.")
            
            confidence_val = probs.get(pred_class, max(probs.values()))
            
            with st.container(border=True):
                if pred_class == "unknown":
                    st.markdown(f"""
                    <div style="background-color: #fef2f2; padding: 20px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #fecaca;">
                        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #991b1b; font-weight: 700; margin-bottom: 4px;">Primary Diagnostic Category</div>
                        <div style="font-size: 24px; font-weight: 700; color: #7f1d1d; margin: 4px 0; font-family: 'Inter', sans-serif;">{disease_info.get('name', 'Unknown Category (Low Confidence)')}</div>
                        <div style="margin-top: 10px; display: flex; align-items: center; gap: 14px;">
                            <span class="badge {badge_style}">{disease_info.get('severity', 'Undetermined')}</span>
                            <span style="font-weight: 600; font-size: 14px; color: #991b1b;">{(confidence_val*100):.2f}% Calibrated Confidence (Below Threshold)</span>
                        </div>
                        <p style="margin-top: 14px; font-size: 13px; color: #7f1d1d; line-height: 1.5; border-top: 1px solid #fecaca; padding-top: 10px;">
                            <strong>System Note:</strong> {disease_info.get('guideline')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #cbd5e1;">
                        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #1e40af; font-weight: 700; margin-bottom: 4px;">Predicted Diagnostic Category</div>
                        <div style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 4px 0; font-family: 'Inter', sans-serif;">{disease_info.get('name', pred_class)}</div>
                        <div style="margin-top: 10px; display: flex; align-items: center; gap: 14px;">
                            <span class="badge {badge_style}">{disease_info.get('severity', 'Benign')}</span>
                            <span style="font-weight: 600; font-size: 15px; color: #1e40af;">{(confidence_val*100):.2f}% Calibrated Confidence</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<div class="section-header">Explainable AI Visualizations</div>', unsafe_allow_html=True)
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(raw_np, caption="Original Dermoscopic Input Image", use_container_width=True)
                with col_img2:
                    st.image(blended_np, caption="Grad-CAM Neural Attention Hotspot Overlay", use_container_width=True)

            with st.container(border=True):
                st.markdown('<div class="section-header">Calibrated Probability Distribution</div>', unsafe_allow_html=True)
                
                # Formal Muted Clinical Color Palette for Metrics
                disease_color_map = {
                    "mel": "#991b1b",     # Melanoma: Muted Dark Red
                    "bcc": "#b91c1c",     # Basal Cell Carcinoma: Muted Crimson
                    "akiec": "#c2410c",   # Actinic Keratoses: Muted Rust
                    "bkl": "#475569",     # Benign Keratosis: Muted Slate Gray
                    "df": "#334155",      # Dermatofibroma: Muted Dark Slate
                    "vasc": "#1e40af",    # Vascular Lesions: Formal Medical Blue
                    "nv": "#15803d",      # Melanocytic Nevus: Muted Dark Green
                    "unknown": "#64748b"  # Low Confidence / Undetermined: Neutral Gray
                }
                
                sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                for code, val in sorted_probs:
                    name = engine.report_generator.DISEASE_INFO.get(code, {}).get("name", code)
                    pct = val * 100
                    bar_color = disease_color_map.get(code.lower(), "#1e40af")
                    st.markdown(f"""
                    <div class="probability-label">{name} ({code.upper()})</div>
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="background-color: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden; flex-grow: 1;">
                            <div style="width: {pct}%; height: 100%; background-color: {bar_color}; border-radius: 4px;"></div>
                        </div>
                        <div style="font-weight: 600; width: 65px; text-align: right; font-size: 13px; color: #0f172a; font-family: 'Inter', sans-serif; margin-left: 10px;">{pct:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<div class="section-header">Clinical Pathology Notes and Guidelines</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="guideline-box">
                    <p style="margin: 0 0 8px 0; color: #0f172a;"><strong>Pathology Summary:</strong> {disease_info.get('desc')}</p>
                    <p style="margin: 0; color: #1e40af;"><strong>Clinical Recommendation:</strong> {disease_info.get('guideline')}</p>
                </div>
                """, unsafe_allow_html=True)

            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    html_report = f.read()
                
                st.markdown("---")
                st.download_button(
                    label="DOWNLOAD CLINICAL DIAGNOSTIC REPORT (HTML)",
                    data=html_report,
                    file_name=f"clinical_report_{st.session_state.patient_id}.html",
                    mime="text/html",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"An error occurred during diagnostic processing: {str(e)}")
            logger.exception("Streamlit inference processing error:")
