"""
Verification script for Phase 17: Inference Engine.
Mocks model weights, serializes a dummy preprocessor state, creates a temporary test image,
and verifies that the inference pipeline runs a complete diagnosis and outputs reports.
"""

import sys
import os

# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch
import numpy as np
import pickle
from PIL import Image, ImageDraw

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed, get_device
from dataset.metadata import MetadataPreprocessor
from models.multimodal.multimodal_net import HybridMultiModalClassifier
from utils.inference_engine import InferenceEngine


def verify_inference_engine() -> None:
    """
    Creates dummy serialization states, instantiates InferenceEngine,
    runs single-patient test diagnostics, and cleans up.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Inference Engine Verification...")

    # 2. Load settings and set seed
    config_path = "configs/default_config.yaml"
    config = Config.load_from_yaml(config_path)
    set_seed(config.data.seed)
    
    # 3. Detect and allocate device (GPU/CPU)
    device = get_device()

    # Paths for mock files
    test_weights_path = "tests/mock_best_model.pth"
    test_preprocessor_path = "tests/mock_metadata_preprocessor.pkl"
    test_image_path = "tests/mock_lesion_input.jpg"

    # =================================================================
    # SETUP MOCK SERIALIZED FILES
    # =================================================================
    logger.info("Setting up mock weights and preprocessors...")
    
    # Reconstruct classifier model matching config ablation flags and save its dummy states
    mode = config.ablation.mode.lower().strip()
    image_mode = config.ablation.image_mode.lower().strip()

    use_metadata = (mode in ["multimodal", "metadata_only"])
    if mode == "metadata_only":
        use_cnn = False
        use_vit = False
    else:
        use_cnn = (image_mode in ["hybrid", "cnn_only"])
        use_vit = (image_mode in ["hybrid", "vit_only"])

    model = HybridMultiModalClassifier(
        num_classes=7,
        pretrained=False,
        use_cnn=use_cnn,
        use_vit=use_vit,
        use_metadata=use_metadata,
        meta_embed_dim=config.model.metadata_embed_dim,
        joint_dim=config.model.fusion_dim,
        dropout_prob=config.model.dropout
    ).to(device)

    checkpoint_state = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint_state, test_weights_path)

    # Reconstruct metadata preprocessor, fit on dummy dataframe, and serialize
    preprocessor = MetadataPreprocessor()
    
    # Generate dummy data covering all categories to match the standard 19-dim shape
    sex_list = ["male", "female", "unknown"]
    loc_list = [
        "abdomen", "acral", "back", "chest", "ear", "face", "foot",
        "genital", "hand", "lower extremity", "neck", "scalp", "trunk",
        "unknown", "upper extremity"
    ]
    
    # Construct a dummy dataframe covering all categories
    rows = []
    for i in range(max(len(sex_list), len(loc_list))):
        rows.append({
            "age": float(30 + i),
            "sex": sex_list[i % len(sex_list)],
            "localization": loc_list[i % len(loc_list)]
        })
    dummy_train_df = pd.DataFrame(rows)
    preprocessor.fit(dummy_train_df)
    
    with open(test_preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    # Save a temporary organic skin-and-lesion image to satisfy validation pipeline
    img = Image.new("RGB", (224, 224), color=(210, 160, 140))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 174, 174], fill=(80, 40, 30), outline=(70, 35, 25))
    img_np = np.array(img)
    noise = np.random.normal(0, 3, img_np.shape).astype(np.uint8)
    Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8)).save(test_image_path)

    logger.info("Mock files set up successfully. Initializing InferenceEngine...")

    # =================================================================
    # RUN INFERENCE ENGINE
    # =================================================================
    # Change outputs path in config to tests directory to prevent pollution
    config.paths.output_dir = "tests/test_outputs"
    os.makedirs(os.path.join(config.paths.output_dir, "reports"), exist_ok=True)

    engine = InferenceEngine(
        config_path=config_path,
        model_path=test_weights_path,
        preprocessor_path=test_preprocessor_path,
        device=device
    )

    # Run single-patient diagnosis
    logger.info("Executing prediction and visual heatmap synthesis...")
    pred_class, probs, report_path = engine.predict_and_explain(
        image_path=test_image_path,
        age=45.0,
        sex="female",
        localization="back",
        patient_id="CASE_MOCK_112",
        report_filename="mock_clinical_report.html"
    )

    logger.info(f"Diagnosis result: {pred_class.upper()}")
    logger.info(f"Confidence map: {probs}")
    logger.info(f"Report compiled at: {report_path}")

    # Assertions
    assert pred_class in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "unknown"], "Invalid diagnosis classification."
    assert len(probs) == 7, "Prediction distribution size mismatch."
    assert os.path.exists(report_path), "Clinical HTML report not saved."
    assert os.path.getsize(report_path) > 1000, "HTML report file is empty."
    logger.info("[OK] Inference and explainability loop: PASSED.")

    # =================================================================
    # CLEANUP TEMPORARY FILES
    # =================================================================
    logger.info("Detaching resources and cleaning up mock files...")
    os.remove(test_weights_path)
    os.remove(test_preprocessor_path)
    os.remove(test_image_path)
    os.remove(report_path)
    os.rmdir(os.path.join(config.paths.output_dir, "reports"))
    os.rmdir(config.paths.output_dir)
    
    logger.info("Cleanup complete.")
    logger.info("Inference Engine verification successful!")


if __name__ == "__main__":
    verify_inference_engine()
