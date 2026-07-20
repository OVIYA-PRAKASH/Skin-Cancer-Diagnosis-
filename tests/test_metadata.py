"""
Verification script for Phase 5: Metadata Loader.
Loads data, fits preprocessor on training subset, transforms train/val/test splits,
and verifies tensor dimensions, missing age imputation, and unseen category handling.
"""

import sys
import os
import pandas as pd
import torch

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed
from dataset.loader import MetadataLoader, LesionLevelSplitter
from dataset.metadata import MetadataPreprocessor


def verify_metadata_preprocessor() -> None:
    """
    Validates the MetadataPreprocessor pipeline.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Metadata Preprocessor Verification...")

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

    train_r, val_r, test_r = config.data.train_val_test_split
    splitter = LesionLevelSplitter(
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=config.data.seed
    )
    train_df, val_df, test_df = splitter.split(df)

    # 4. Instantiate and fit the preprocessor on the train split only
    preprocessor = MetadataPreprocessor()
    preprocessor.fit(train_df)

    # Check total feature dimension
    expected_dim = 19  # 1 (age) + 3 (sex: male/female/unknown) + 15 (anatomical sites)
    logger.info(f"Preprocessor feature dimension: {preprocessor.feature_dim}")
    assert preprocessor.feature_dim == expected_dim, f"Expected dimension {expected_dim}, got {preprocessor.feature_dim}"
    logger.info("[OK] Feature Dimension Check: PASSED.")

    # 5. Transform all dataset splits and verify shapes
    train_tensor = preprocessor.transform(train_df)
    val_tensor = preprocessor.transform(val_df)
    test_tensor = preprocessor.transform(test_df)

    logger.info(f"Train metadata tensor shape: {train_tensor.shape}")
    logger.info(f"Val metadata tensor shape: {val_tensor.shape}")
    logger.info(f"Test metadata tensor shape: {test_tensor.shape}")

    assert train_tensor.shape == (len(train_df), expected_dim), "Train split shape mismatch."
    assert val_tensor.shape == (len(val_df), expected_dim), "Val split shape mismatch."
    assert test_tensor.shape == (len(test_df), expected_dim), "Test split shape mismatch."
    logger.info("[OK] Split Transforms Shape Check: PASSED.")

    # 6. Verify that no NaN values remain
    assert not torch.isnan(train_tensor).any(), "NaN values found in train feature tensor."
    assert not torch.isnan(val_tensor).any(), "NaN values found in validation feature tensor."
    assert not torch.isnan(test_tensor).any(), "NaN values found in test feature tensor."
    logger.info("[OK] Missing Values Clean Check: PASSED.")

    # 7. Stress test: Test handling of missing ages and unseen localization sites
    logger.info("Running stress test with missing data and novel categories...")
    mock_patient_df = pd.DataFrame([
        {
            "age": float("nan"),          # Missing age: should impute to training mean (0.0 scaled)
            "sex": "Female",              # Known sex
            "localization": "Mars"         # Unseen site: should map to 'unknown'
        }
    ])
    
    # Transform mock data
    mock_tensor = preprocessor.transform(mock_patient_df)
    logger.info(f"Mock patient feature tensor: {mock_tensor}")

    # Index 0 is age. Since age is NaN, it should map to the mean age, which has a normalized value of 0.0
    age_val = mock_tensor[0, 0].item()
    logger.info(f"Imputed age scaled value: {age_val:.4f}")
    assert abs(age_val) < 1e-5, f"Imputed age should be 0.0, got {age_val:.4f}"

    # Verify that the unseen localization 'Mars' was mapped to 'unknown' (the last entry index of localization)
    # Let's find index of 'unknown' localization
    loc_cats = preprocessor.localization_categories
    unknown_loc_idx = loc_cats.index("unknown")
    # In the concatenated vector, index starts after: 1 (age) + 3 (sex) = 4
    absolute_unknown_loc_idx = 4 + unknown_loc_idx
    unknown_loc_val = mock_tensor[0, absolute_unknown_loc_idx].item()
    logger.info(f"Unseen localization mapped to 'unknown' indicator index {absolute_unknown_loc_idx}: {unknown_loc_val}")
    assert unknown_loc_val == 1.0, f"Unseen localization should map to 1.0 at index {absolute_unknown_loc_idx}, got {unknown_loc_val}"

    logger.info("[OK] Stress Test: PASSED.")
    logger.info("Metadata Preprocessor verification successful!")


if __name__ == "__main__":
    verify_metadata_preprocessor()
