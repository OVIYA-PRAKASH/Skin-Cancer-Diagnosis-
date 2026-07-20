"""
Verification script for Phase 14, 15 & 16: Grad-CAM, Confidence Scores, and Clinical Reports.
Instantiates hooks, executes a backpropagation pass to generate Grad-CAM overlays,
and compiles a self-contained clinical HTML diagnostic report.
"""

import sys
import os

# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch
import numpy as np
from PIL import Image

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed, get_device
from models.multimodal.multimodal_net import HybridMultiModalClassifier
from explainability.gradcam import GradCAM
from reports.report_generator import ClinicalReportGenerator


def verify_explainability_pipeline() -> None:
    """
    Validates hooks, backpropagation gradients, visual overlays, and report generation.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Explainability Suite Verification...")

    # 2. Load settings and set seed
    config_path = "configs/default_config.yaml"
    config = Config.load_from_yaml(config_path)
    set_seed(config.data.seed)
    
    # 3. Detect and allocate device (GPU/CPU)
    device = get_device()
    img_size = config.data.image_size

    # 4. Build Model wrapper
    logger.info("Building model wrapper...")
    model = HybridMultiModalClassifier(
        num_classes=7,
        pretrained=False,
        use_cnn=True,
        use_vit=False,       # Disable ViT for speed
        use_metadata=True
    ).to(device)

    # 5. Initialize Grad-CAM targeting the last convolution layer of CNN backbone
    logger.info("Attaching hooks to last convolutional layer of CNN backbone...")
    target_layer = model.cnn_extractor.features[-1]
    gradcam = GradCAM(model=model, target_layer=target_layer)

    # 6. Generate mock single patient sample [1, 3, 224, 224] & [1, 19]
    mock_image = torch.randn(1, 3, img_size, img_size).to(device)
    mock_metadata = torch.randn(1, 19).to(device)

    # 7. Generate heatmap and prediction parameters
    logger.info("Executing backward pass to extract Grad-CAM activations...")
    heatmap, pred_idx, confidence = gradcam.generate_heatmap(
        images=mock_image,
        metadata=mock_metadata,
        class_idx=None
    )

    logger.info(f"Heatmap shape: {heatmap.shape}")
    logger.info(f"Predicted Class Index: {pred_idx}")
    logger.info(f"Prediction Confidence Score: {confidence:.4f}")

    # Assertions
    assert heatmap.shape == (img_size, img_size), f"Expected shape {(img_size, img_size)}, got {heatmap.shape}"
    assert 0.0 <= confidence <= 1.0, "Confidence score out of probability bounds."
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0, "Heatmap values not normalized."
    logger.info("[OK] Grad-CAM Activation map extraction: PASSED.")

    # 8. Test blending overlay
    logger.info("Testing visual overlay blending...")
    # Create a mock original image array (values [0, 255])
    mock_orig_np = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
    blended_np = GradCAM.overlay_heatmap(image_np=mock_orig_np, heatmap=heatmap, alpha=0.4)
    
    assert blended_np.shape == (img_size, img_size, 3), "Overlay dimension mismatch."
    assert blended_np.dtype == np.uint8, "Overlay datatype mismatch."
    logger.info("[OK] Visual Blend overlay generation: PASSED.")

    # 9. Test Report Generator
    test_report_dir = "tests/test_reports"
    logger.info("Initializing ClinicalReportGenerator...")
    report_generator = ClinicalReportGenerator(output_dir=test_report_dir)

    # Convert arrays to Pillow images for report compiler
    orig_pil = Image.fromarray(mock_orig_np)
    blended_pil = Image.fromarray(blended_np)

    demographics = {
        "age": 52,
        "sex": "Male",
        "localization": "Back"
    }

    # Mock predictions
    predictions = {
        "akiec": 0.01,
        "bcc": 0.04,
        "bkl": 0.05,
        "df": 0.01,
        "mel": 0.82,     # Suspicious diagnosis predicted
        "nv": 0.06,
        "vasc": 0.01
    }

    report_path = report_generator.generate_report(
        patient_id="CASE_TEST_99",
        demographics=demographics,
        predictions=predictions,
        original_image=orig_pil,
        heatmap_image=blended_pil,
        filename="test_report.html"
    )

    # Verify that the report exists and holds non-empty characters
    assert os.path.exists(report_path), "clinical report HTML was not generated."
    assert os.path.getsize(report_path) > 1000, "clinical report file size is empty."
    logger.info("[OK] Clinical HTML Report assembly: PASSED.")

    # 10. Cleanup
    gradcam.remove_hooks()
    os.remove(report_path)
    os.rmdir(test_report_dir)
    logger.info("Cleanup complete. Hooks and test folders detached successfully.")

    logger.info("Explainability Suite verification successful!")


if __name__ == "__main__":
    verify_explainability_pipeline()
