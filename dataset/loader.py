"""
Phase 3: Dataset Loader module.
Handles reading metadata, locating image file paths on disk, and partitioning the dataset
into stratified train/val/test splits at the lesion level to prevent data leakage.
"""

import os
import logging
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

logger = logging.getLogger("SkinCancerAI.DatasetLoader")


class MetadataLoader:
    """
    Manages loading the metadata tabular file, locating corresponding image files
    across multiple directories, and performing data validation.
    """

    def __init__(self, metadata_path: str, image_dirs: List[str]):
        """
        Initializes the MetadataLoader.

        Args:
            metadata_path (str): Path to the metadata .tab or .csv file.
            image_dirs (List[str]): List of directories containing image files.
        """
        self.metadata_path = metadata_path
        self.image_dirs = image_dirs
        self.df: pd.DataFrame = pd.DataFrame()
        self.image_to_path_map: Dict[str, str] = {}

    def scan_image_directories(self) -> None:
        """
        Scans all provided image directories and maps each image ID to its absolute path on disk.
        """
        logger.info(f"Scanning image directories: {self.image_dirs}")
        extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        
        for image_dir in self.image_dirs:
            if not os.path.exists(image_dir):
                logger.error(f"Image directory not found: {image_dir}")
                continue
                
            for filename in os.listdir(image_dir):
                if filename.endswith(extensions):
                    image_id, _ = os.path.splitext(filename)
                    abs_path = os.path.abspath(os.path.join(image_dir, filename))
                    self.image_to_path_map[image_id] = abs_path
                    
        logger.info(f"Scan complete. Found {len(self.image_to_path_map)} unique images on disk.")

    def load_metadata(self) -> pd.DataFrame:
        """
        Loads the tabular metadata file and links each entry to its disk image path.

        Returns:
            pd.DataFrame: Merged metadata dataframe containing image paths.
        
        Raises:
            FileNotFoundError: If the metadata path does not exist.
            ValueError: If the metadata structure is empty or invalid.
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")
            
        logger.info(f"Loading metadata from: {self.metadata_path}")
        # Detect delimiter (tab vs comma) based on file extension
        sep = "\t" if self.metadata_path.endswith(".tab") else ","
        
        self.df = pd.read_csv(self.metadata_path, sep=sep)
        
        if self.df.empty:
            raise ValueError("The loaded metadata file is empty.")
            
        logger.info(f"Loaded metadata successfully. Total entries: {len(self.df)}")
        
        # Scan directories first if not already done
        if not self.image_to_path_map:
            self.scan_image_directories()
            
        # Map image paths to entries in the metadata dataframe
        logger.info("Mapping metadata image_ids to disk paths...")
        self.df["image_path"] = self.df["image_id"].map(self.image_to_path_map)
        
        # Identify and handle missing images (entries in metadata that lack a physical file)
        missing_count = self.df["image_path"].isna().sum()
        if missing_count > 0:
            logger.warning(
                f"{missing_count} metadata entries do not correspond to any physical image files. "
                "These entries will be pruned."
            )
            # Prune missing entries
            self.df = self.df.dropna(subset=["image_path"]).reset_index(drop=True)
            logger.info(f"Entries remaining after pruning missing images: {len(self.df)}")
            
        return self.df


class LesionLevelSplitter:
    """
    Splits the dataset into stratified train, validation, and test subsets at the lesion level.
    This guarantees that multiple images of the same physical lesion do not cross split boundaries.
    """

    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
        """
        Initializes the splitter.

        Args:
            train_ratio (float): Fraction of data to use for training.
            val_ratio (float): Fraction of data to use for validation.
            test_ratio (float): Fraction of data to use for testing.
            seed (int): Random seed for reproducibility.
        """
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 1e-5, f"Ratios must sum to 1.0, got: {total}"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partitions the input dataframe at the lesion level.

        Args:
            df (pd.DataFrame): Dataframe containing 'lesion_id', 'image_id', and 'dx' label.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        logger.info(
            f"Performing stratified lesion-level split (Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {self.test_ratio})"
        )
        
        # 1. Extract unique lesions and their corresponding labels
        # (We use the first diagnosis label associated with the lesion if there are discrepancies)
        unique_lesions = df.groupby("lesion_id").first().reset_index()[["lesion_id", "dx"]]
        
        # 2. Compute temporary ratio for the validation/test split from the remaining fraction
        # Let's say we want 80% train, 10% val, 10% test.
        # First split off 80% train, leaving 20% validation+test.
        val_test_ratio = self.val_ratio + self.test_ratio
        
        train_lesions, val_test_lesions = train_test_split(
            unique_lesions,
            test_size=val_test_ratio,
            random_state=self.seed,
            stratify=unique_lesions["dx"]
        )
        
        # Split the remaining 20% equally (50/50 of the remaining 20% = 10% and 10% of total)
        # Ratio of val relative to the remaining = val_ratio / val_test_ratio
        test_fraction = self.test_ratio / val_test_ratio
        
        val_lesions, test_lesions = train_test_split(
            val_test_lesions,
            test_size=test_fraction,
            random_state=self.seed,
            stratify=val_test_lesions["dx"]
        )
        
        # 3. Map lesion splits back to original image-level dataframe
        train_df = df[df["lesion_id"].isin(train_lesions["lesion_id"])].copy().reset_index(drop=True)
        val_df = df[df["lesion_id"].isin(val_lesions["lesion_id"])].copy().reset_index(drop=True)
        test_df = df[df["lesion_id"].isin(test_lesions["lesion_id"])].copy().reset_index(drop=True)
        
        # 4. Print stats and verify class distributions
        logger.info("Lesion-level split complete.")
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            logger.info(
                f"{name} set: {len(split_df)} images (spanning {split_df['lesion_id'].nunique()} unique lesions)"
            )
            # Log class counts to verify stratification
            class_counts = split_df["dx"].value_counts().to_dict()
            logger.debug(f"{name} set class distribution: {class_counts}")
            
        return train_df, val_df, test_df
