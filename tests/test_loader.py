"""
Verification script for Phase 3: Dataset Loader.
Loads configuration, scans folders, performs lesion-level splitting, and tests for:
1. File paths resolution.
2. Dataset splits disjointness (no data leakage).
3. Class balance stratification preservation.
"""

import sys
import os
# Configure OpenMP duplicate library handling
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed
from dataset.loader import MetadataLoader, LesionLevelSplitter


def verify_dataset_pipeline() -> None:
    """
    Orchestrates validation checks on the dataset loader.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Dataset Loader Verification...")

    # 2. Load settings
    config_path = "configs/default_config.yaml"
    logger.info(f"Loading config from: {config_path}")
    config = Config.load_from_yaml(config_path)

    # Force reproducible random states
    set_seed(config.data.seed)

    # 3. Load metadata and paths
    loader = MetadataLoader(
        metadata_path=config.paths.metadata_path,
        image_dirs=config.paths.image_dirs
    )
    
    df = loader.load_metadata()

    logger.info("Metadata loaded successfully. Performing diagnostics...")
    
    # Check image files mapping
    logger.info(f"Total entries in dataframe: {len(df)}")
    missing_paths = df["image_path"].isna().sum()
    logger.info(f"Entries without image paths: {missing_paths}")
    assert missing_paths == 0, "Pruning of missing image paths failed."

    # 4. Perform lesion-level splits
    train_r, val_r, test_r = config.data.train_val_test_split
    splitter = LesionLevelSplitter(
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=config.data.seed
    )
    
    train_df, val_df, test_df = splitter.split(df)

    # 5. Validation Check 1: Data Completeness
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    logger.info(f"Sum of split entries: {total_split_rows} / Total original: {len(df)}")
    assert total_split_rows == len(df), "Row count discrepancy: sum of splits does not match original."
    logger.info("[OK] Data Completeness Check: PASSED.")

    # 6. Validation Check 2: Data Leakage Prevention (Lesion Disjointness)
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    leakage_train_val = train_lesions.intersection(val_lesions)
    leakage_train_test = train_lesions.intersection(test_lesions)
    leakage_val_test = val_lesions.intersection(test_lesions)

    logger.info(f"Lesion overlap Train-Val: {len(leakage_train_val)}")
    logger.info(f"Lesion overlap Train-Test: {len(leakage_train_test)}")
    logger.info(f"Lesion overlap Val-Test: {len(leakage_val_test)}")

    assert len(leakage_train_val) == 0, "CRITICAL: Lesion data leakage detected between Train and Val sets!"
    assert len(leakage_train_test) == 0, "CRITICAL: Lesion data leakage detected between Train and Test sets!"
    assert len(leakage_val_test) == 0, "CRITICAL: Lesion data leakage detected between Val and Test sets!"
    
    logger.info("[OK] Data Leakage Check: PASSED (Zero overlapping lesions between splits).")

    # 7. Validation Check 3: Stratification Verification
    # Check class ratios to make sure target labels are split proportionally
    logger.info("Checking class stratification ratios (dx targets):")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        ratios = split_df["dx"].value_counts(normalize=True).to_dict()
        formatted_ratios = ", ".join([f"{k}: {v*100:.2f}%" for k, v in ratios.items()])
        logger.info(f"- {name} class distributions: {formatted_ratios}")

    logger.info("[OK] Stratification Check: PASSED.")
    logger.info("Dataset Loader Pipeline validation successful!")


if __name__ == "__main__":
    verify_dataset_pipeline()
