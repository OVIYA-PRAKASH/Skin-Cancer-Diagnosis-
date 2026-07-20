"""
Verification script for Phase 9: Multi-modal Network.
Instantiates HybridMultiModalClassifier, runs forward passes using mock image and
metadata batches under standard and ablation modes, and asserts output logit shapes.
"""

import sys
import os

# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed, get_device
from models.multimodal.multimodal_net import HybridMultiModalClassifier


def verify_multimodal_net() -> None:
    """
    Validates HybridMultiModalClassifier forward pass shapes under normal and ablation modes.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Multi-Modal Classifier Verification...")

    # 2. Load settings and set seed
    config_path = "configs/default_config.yaml"
    config = Config.load_from_yaml(config_path)
    set_seed(config.data.seed)
    
    # 3. Detect and allocate device (GPU if CUDA binds, else CPU)
    device = get_device()
    batch_size = 4
    img_size = config.data.image_size

    # Create mock inputs matching dataset shape
    logger.info("Generating mock dataset tensors...")
    mock_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    mock_metadata = torch.randn(batch_size, 19).to(device)

    logger.info(f"- Mock images tensor: {mock_images.shape} on {device}")
    logger.info(f"- Mock metadata tensor: {mock_metadata.shape} on {device}")

    # =================================================================
    # TEST 1: Full Multi-Modal Classifier
    # =================================================================
    logger.info("TEST 1: Instantiating Full Multi-Modal configuration...")
    model = HybridMultiModalClassifier(
        num_classes=7,
        cnn_backbone="efficientnet_b0",
        vit_backbone="vit_b_16",
        pretrained=False,
        use_cnn=True,
        use_vit=True,
        use_metadata=True
    ).to(device)

    model.eval()

    logger.info("Running forward pass for Test 1...")
    with torch.no_grad():
        logits = model(mock_images, mock_metadata)

    logger.info(f"Full classifier logits output shape: {logits.shape}")
    assert logits.shape == (batch_size, 7), f"Expected shape {(batch_size, 7)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Found NaNs in Full classifier logits."
    logger.info("[OK] Test 1 (Full Multi-Modal): PASSED.")

    # =================================================================
    # TEST 2: Visual-Only Ablation Configuration (Metadata Disabled)
    # =================================================================
    logger.info("TEST 2: Instantiating Visual-Only Ablation configuration...")
    ablation_model = HybridMultiModalClassifier(
        num_classes=7,
        cnn_backbone="efficientnet_b0",
        vit_backbone="vit_b_16",
        pretrained=False,
        use_cnn=True,
        use_vit=True,
        use_metadata=False  # Disable patient metadata
    ).to(device)

    ablation_model.eval()

    logger.info("Running forward pass for Test 2...")
    with torch.no_grad():
        ablation_logits = ablation_model(mock_images, mock_metadata)

    logger.info(f"Ablation classifier logits output shape: {ablation_logits.shape}")
    assert ablation_logits.shape == (batch_size, 7), f"Expected shape {(batch_size, 7)}, got {ablation_logits.shape}"
    assert not torch.isnan(ablation_logits).any(), "Found NaNs in Ablation classifier logits."
    logger.info("[OK] Test 2 (Visual-Only Ablation): PASSED.")

    logger.info("Multi-Modal Classifier Network verification successful!")


if __name__ == "__main__":
    verify_multimodal_net()
