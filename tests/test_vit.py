"""
Verification script for Phase 7: Vision Transformer.
Instantiates ViTFeatureExtractor, runs a forward pass using a mock batch of image tensors,
and asserts output tensor dimensions.
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
from models.transformer.vit_backbone import ViTFeatureExtractor


def verify_vit_extractor() -> None:
    """
    Validates ViTFeatureExtractor configurations and forward pass output shapes.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing ViT Backbone Verification...")

    # 2. Load settings and set seed
    config_path = "configs/default_config.yaml"
    config = Config.load_from_yaml(config_path)
    set_seed(config.data.seed)
    
    # 3. Detect and allocate device (GPU if CUDA binds, else CPU)
    device = get_device()
    batch_size = 4
    img_size = config.data.image_size  # Typically 224

    # Create a mock batch of images [Batch, Channels, Height, Width]
    mock_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    logger.info(f"Generated mock image batch tensor of shape: {mock_images.shape} on {device}")

    # =================================================================
    # TEST 1: ViT-B/16
    # =================================================================
    logger.info("Testing Test Case 1: ViT-B/16 backbone...")
    vit_extractor = ViTFeatureExtractor(backbone_name="vit_b_16", pretrained=False)
    vit_extractor = vit_extractor.to(device)
    
    # Set to evaluation mode
    vit_extractor.eval()

    with torch.no_grad():
        vit_features = vit_extractor(mock_images)

    logger.info(f"ViT-B/16 features shape: {vit_features.shape}")
    
    # Assertions
    assert vit_features.shape == (batch_size, 768), \
        f"Expected shape {(batch_size, 768)}, got {vit_features.shape}"
    assert not torch.isnan(vit_features).any(), "Found NaNs in ViT features."
    logger.info("[OK] ViT-B/16 dimension check: PASSED.")

    logger.info("ViT Model verification successful!")


if __name__ == "__main__":
    verify_vit_extractor()
