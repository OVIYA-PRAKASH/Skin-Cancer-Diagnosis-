"""
Phase 5: Metadata Loader.
Implements demographic feature engineering, including missing age imputation,
standard scaling of numerical age, and one-hot encoding of categorical demographics
(sex and anatomical site localization).
"""

import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple

logger = logging.getLogger("SkinCancerAI.MetadataLoader")


class MetadataPreprocessor:
    """
    Fits statistics on the training demographic split and transforms tabular patient metadata
    (age, sex, localization) into continuous-categorical feature vectors.
    """

    def __init__(self):
        """Initializes the preprocessor with empty statistics."""
        self.age_mean: float = 0.0
        self.age_std: float = 1.0
        self.sex_categories: List[str] = []
        self.localization_categories: List[str] = []
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Calculates and stores mean/std for age and unique categories for sex and localization
        from the training split to prevent any data leakage.

        Args:
            train_df (pd.DataFrame): Tabular training split metadata.
        """
        logger.info("Fitting MetadataPreprocessor on training split demographics...")

        # 1. Age stats: compute mean and standard deviation on non-null values
        valid_ages = train_df["age"].dropna()
        if not valid_ages.empty:
            self.age_mean = float(valid_ages.mean())
            self.age_std = float(valid_ages.std())
            # Handle edge case where std is 0 to avoid division by zero
            if self.age_std < 1e-5:
                self.age_std = 1.0
        else:
            self.age_mean = 50.0  # Fallback age if all ages are missing
            self.age_std = 15.0

        # 2. Sex categories: standard categories + ensure 'unknown' is mapped
        # Unique categories from training data: lowercase and stripped
        train_sex = train_df["sex"].fillna("unknown").astype(str).str.lower().str.strip()
        self.sex_categories = sorted(list(train_sex.unique()))
        if "unknown" not in self.sex_categories:
            self.sex_categories.append("unknown")

        # 3. Localization categories: anatomical sites + ensure 'unknown'
        train_loc = train_df["localization"].fillna("unknown").astype(str).str.lower().str.strip()
        self.localization_categories = sorted(list(train_loc.unique()))
        if "unknown" not in self.localization_categories:
            self.localization_categories.append("unknown")

        self.is_fitted = True
        logger.info(
            f"Fit complete. Training Age Mean: {self.age_mean:.2f}, Std: {self.age_std:.2f}. "
            f"Sex categories: {self.sex_categories}. "
            f"Localization categories: {self.localization_categories}."
        )

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Transforms tabular metadata into a standardized one-hot encoded tensor.

        Args:
            df (pd.DataFrame): Demographics dataframe containing 'age', 'sex', and 'localization'.

        Returns:
            torch.Tensor: Normalized feature tensor of shape [batch_size, feature_dim].
        
        Raises:
            ValueError: If preprocessor is not fitted before transforming.
        """
        if not self.is_fitted:
            raise ValueError("MetadataPreprocessor must be fitted on training data before transforming.")

        # 1. Process and normalize Age
        # Impute missing values with training mean
        ages = df["age"].fillna(self.age_mean).to_numpy(dtype=np.float32)
        # Apply standard scaling
        normalized_ages = (ages - self.age_mean) / self.age_std
        # Shape: [num_samples, 1]
        age_feature = np.expand_dims(normalized_ages, axis=1)

        # 2. Process and One-Hot encode Sex
        sex_series = df["sex"].fillna("unknown").astype(str).str.lower().str.strip()
        sex_features = []
        for cat in self.sex_categories:
            # Create a boolean column map for this category
            col = (sex_series == cat).astype(np.float32)
            sex_features.append(col.to_numpy())
        # Shape: [num_samples, num_sex_cats]
        sex_feature = np.column_stack(sex_features)

        # 3. Process and One-Hot encode Localization
        loc_series = df["localization"].fillna("unknown").astype(str).str.lower().str.strip()
        # Handle anatomical sites that aren't in training fit by mapping them to 'unknown'
        # First check and replace any unseen sites with 'unknown'
        unseen_mask = ~loc_series.isin(self.localization_categories)
        if unseen_mask.any():
            logger.warning(
                f"Found {unseen_mask.sum()} entries with anatomical sites not present in training categories. "
                "These will be mapped to 'unknown'."
            )
            loc_series[unseen_mask] = "unknown"

        loc_features = []
        for cat in self.localization_categories:
            col = (loc_series == cat).astype(np.float32)
            loc_features.append(col.to_numpy())
        # Shape: [num_samples, num_loc_cats]
        loc_feature = np.column_stack(loc_features)

        # 4. Concatenate all demographic feature vectors
        # Total size = 1 (age) + len(sex_categories) + len(localization_categories)
        features_concat = np.hstack([age_feature, sex_feature, loc_feature])
        
        return torch.tensor(features_concat, dtype=torch.float32)

    @property
    def feature_dim(self) -> int:
        """
        Returns the dimension of the concatenated output metadata feature vector.

        Returns:
            int: Dimension size.
        """
        if not self.is_fitted:
            return 0
        return 1 + len(self.sex_categories) + len(self.localization_categories)
