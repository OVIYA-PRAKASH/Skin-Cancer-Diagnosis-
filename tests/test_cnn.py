"""
Verification script for Phase 6: CNN Model.
Instantiates CNNFeatureExtractor with various backbones, runs a forward pass
using a mock batch of image tensors, and asserts output tensor dimensions.
"""

import sys
import os

# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch
import torch.nn as nn

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed, get_device
from models.cnn.cnn_backbone import CNNFeatureExtractor


def verify_cnn_extractor() -> None:
    """
    Validates CNNFeatureExtractor configurations and forward pass output shapes.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing CNN Backbone Verification...")

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
    # TEST 1: EfficientNet-B0
    # =================================================================
    logger.info("Testing Test Case 1: EfficientNet-B0 backbone...")
    effnet_extractor = CNNFeatureExtractor(backbone_name="efficientnet_b0", pretrained=False)
    effnet_extractor = effnet_extractor.to(device)
    
    # Set to evaluation mode to disable batch norm updates during validation pass
    effnet_extractor.eval()

    with torch.no_grad():
        effnet_features = effnet_extractor(mock_images)

    logger.info(f"EfficientNet-B0 features shape: {effnet_features.shape}")
    
    # Assertions
    assert effnet_features.shape == (batch_size, 1280), \
        f"Expected shape {(batch_size, 1280)}, got {effnet_features.shape}"
    assert not torch.isnan(effnet_features).any(), "Found NaNs in EfficientNet features."
    logger.info("[OK] EfficientNet-B0 dimension check: PASSED.")

    # =================================================================
    # TEST 2: ResNet18
    # =================================================================
    logger.info("Testing Test Case 2: ResNet18 backbone...")
    resnet_extractor = CNNFeatureExtractor(backbone_name="resnet18", pretrained=False)
    resnet_extractor = resnet_extractor.to(device)
    resnet_extractor.eval()

    with torch.no_grad():
        resnet_features = resnet_extractor(mock_images)

    logger.info(f"ResNet18 features shape: {resnet_features.shape}")

    # Assertions
    assert resnet_features.shape == (batch_size, 512), \
        f"Expected shape {(batch_size, 512)}, got {resnet_features.shape}"
    assert not torch.isnan(resnet_features).any(), "Found NaNs in ResNet18 features."
    logger.info("[OK] ResNet18 dimension check: PASSED.")

    logger.info("CNN Model verification successful! All backbones pass check.")


if __name__ == "__main__":
    verify_cnn_extractor()
