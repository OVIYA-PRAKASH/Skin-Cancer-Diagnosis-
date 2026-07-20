"""
Verification script for Phase 4: Image Preprocessing.
Loads data splits, wraps them in DermoscopicImageDataset, fetches sample items,
and verifies tensor shapes, types, and values.
"""

import sys
import os
# Configure OpenMP duplicate library handling
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed
from dataset.loader import MetadataLoader, LesionLevelSplitter
from dataset.preprocessing import DermoscopicImageDataset


def verify_image_preprocessing() -> None:
    """
    Validates the dataset image preprocessor and PyTorch Dataset loaders.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Image Preprocessing Verification...")

    # 2. Load settings
    config_path = "configs/default_config.yaml"
    logger.info(f"Loading config from: {config_path}")
    config = Config.load_from_yaml(config_path)

    # Force reproducible random states
    set_seed(config.data.seed)

    # 3. Load metadata and partition splits
    loader = MetadataLoader(
        metadata_path=config.paths.metadata_path,
        image_dirs=config.paths.image_dirs
    )
    df = loader.load_metadata()

    # Map diagnoses classes to integers for classification head
    # Seven target labels in alphabetical order:
    # 0: akiec, 1: bcc, 2: bkl, 3: df, 4: mel, 5: nv, 6: vasc
    classes = sorted(df["dx"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    logger.info(f"Label mappings (dx -> integer idx): {class_to_idx}")
    df["label"] = df["dx"].map(class_to_idx)

    train_r, val_r, test_r = config.data.train_val_test_split
    splitter = LesionLevelSplitter(
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=config.data.seed
    )
    train_df, val_df, test_df = splitter.split(df)

    # 4. Instantiate PyTorch Dataset loaders
    logger.info("Instantiating DermoscopicImageDataset for Training split...")
    train_dataset = DermoscopicImageDataset(
        image_paths=train_df["image_path"].tolist(),
        labels=train_df["label"].tolist(),
        image_size=config.data.image_size,
        is_training=True
    )
    
    logger.info("Instantiating DermoscopicImageDataset for Validation split...")
    val_dataset = DermoscopicImageDataset(
        image_paths=val_df["image_path"].tolist(),
        labels=val_df["label"].tolist(),
        image_size=config.data.image_size,
        is_training=False
    )

    logger.info(f"Train dataset size: {len(train_dataset)} samples.")
    logger.info(f"Validation dataset size: {len(val_dataset)} samples.")

    # 5. Fetch sample items and perform integrity assertions
    logger.info("Fetching validation sample [Index 0]...")
    img_tensor, label_tensor = val_dataset[0]
    
    logger.info(f"Image tensor shape: {img_tensor.shape}")
    logger.info(f"Image tensor dtype: {img_tensor.dtype}")
    logger.info(f"Label tensor: {label_tensor} (dtype: {label_tensor.dtype})")

    # Assertions
    # 3 color channels, height and width matching configured image size
    target_shape = torch.Size([3, config.data.image_size, config.data.image_size])
    assert img_tensor.shape == target_shape, f"Expected shape {target_shape}, got {img_tensor.shape}"
    assert img_tensor.dtype == torch.float32, f"Expected float32, got {img_tensor.dtype}"
    assert label_tensor.dtype == torch.long, f"Expected int64 (long), got {label_tensor.dtype}"

    # Verify normalization (values shouldn't be standard 0-255 or positive 0-1)
    # Normalized features usually contain both positive and negative values due to zero-mean adjustments
    min_val, max_val = img_tensor.min().item(), img_tensor.max().item()
    logger.info(f"Pixel value range: min={min_val:.4f}, max={max_val:.4f}")
    assert min_val < 0.0, "Normalization check failed. Min value should be negative."
    assert max_val > 0.0, "Normalization check failed. Max value should be positive."

    # 6. Fetch training sample to check data augmentation pipelines
    logger.info("Fetching training sample [Index 0] (Augmentations active)...")
    train_img_tensor, train_label = train_dataset[0]
    logger.info(f"Train Image shape: {train_img_tensor.shape}")
    assert train_img_tensor.shape == target_shape, "Training image shape mismatch."

    logger.info("[OK] Image Preprocessing pipeline checks: PASSED.")
    logger.info("Image Preprocessing pipeline verified successfully!")


if __name__ == "__main__":
    verify_image_preprocessing()
