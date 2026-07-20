"""
Verification script for Phase 8: Feature Fusion.
Instantiates MultiModalFeatureFusion, runs a forward pass using mock feature tensors
representing CNN, ViT, and preprocessed metadata inputs, and asserts output dimensions.
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
from models.fusion.feature_fusion import MultiModalFeatureFusion


def verify_feature_fusion() -> None:
    """
    Validates MultiModalFeatureFusion forward pass outputs.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Feature Fusion Verification...")

    # 2. Load settings and set seed
    config_path = "configs/default_config.yaml"
    config = Config.load_from_yaml(config_path)
    set_seed(config.data.seed)
    
    # 3. Detect and allocate device (GPU if CUDA binds, else CPU)
    device = get_device()
    batch_size = 8

    # Input sizes based on our backbone outputs
    cnn_dim = 1280   # EfficientNet-B0
    vit_dim = 768    # ViT-B/16
    meta_dim = 19    # Demographics features size

    # Create mock inputs matching batch patterns
    logger.info("Generating mock feature tensors...")
    mock_cnn = torch.randn(batch_size, cnn_dim).to(device)
    mock_vit = torch.randn(batch_size, vit_dim).to(device)
    mock_meta = torch.randn(batch_size, meta_dim).to(device)

    logger.info(f"- CNN mock features: {mock_cnn.shape} on {device}")
    logger.info(f"- ViT mock features: {mock_vit.shape} on {device}")
    logger.info(f"- Metadata mock features: {mock_meta.shape} on {device}")

    # 4. Instantiate MultiModalFeatureFusion
    fusion_block = MultiModalFeatureFusion(
        cnn_dim=cnn_dim,
        vit_dim=vit_dim,
        meta_dim=meta_dim,
        visual_proj_dim=512,
        meta_embed_dim=128,
        joint_dim=256,
        dropout_prob=0.3
    ).to(device)

    # Set to evaluation mode to disable Dropout / BatchNorm updating behavior
    fusion_block.eval()

    # 5. Forward pass
    logger.info("Executing forward pass through fusion block...")
    with torch.no_grad():
        joint_representation = fusion_block(mock_cnn, mock_vit, mock_meta)

    logger.info(f"Fused joint representation output shape: {joint_representation.shape}")

    # Assertions
    expected_shape = (batch_size, 256)
    assert joint_representation.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {joint_representation.shape}"
    assert not torch.isnan(joint_representation).any(), "Found NaNs in fused features."

    logger.info("[OK] Fused feature dimension check: PASSED.")
    logger.info("Feature Fusion verification successful!")


if __name__ == "__main__":
    verify_feature_fusion()
