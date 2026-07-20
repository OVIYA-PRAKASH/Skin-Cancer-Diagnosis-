"""
Unified Multi-Modal PyTorch Dataset module.
Combines image loading/preprocessing with preprocessed demographic metadata tensors
to yield complete tuples for training, validation, and testing.
"""

import os
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, List

from dataset.preprocessing import ImagePreprocessor

logger = logging.getLogger("SkinCancerAI.Dataset")


class MultiModalDermoscopicDataset(Dataset):
    """
    Unified multi-modal dataset returning (image, metadata, label) tuples.
    Loads images on-demand from disk, applies augmentation/scaling transforms,
    and returns them alongside pre-computed demographic tensors.
    """

    def __init__(
        self,
        image_paths: List[str],
        metadata_tensor: torch.Tensor,
        labels: List[int],
        image_size: int,
        is_training: bool = True
    ):
        """
        Initializes the dataset.

        Args:
            image_paths (List[str]): Absolute disk paths of the dermoscopic images.
            metadata_tensor (torch.Tensor): Preprocessed demographic features tensor [num_samples, 19].
            labels (List[int]): Integer diagnostic target labels.
            image_size (int): Image target dimension for resizing.
            is_training (bool): If True, applies training data augmentations.
        """
        assert len(image_paths) == len(labels), "Number of image paths and labels must match."
        assert len(image_paths) == metadata_tensor.shape[0], "Number of image paths and metadata rows must match."

        self.image_paths = image_paths
        self.metadata_tensor = metadata_tensor
        self.labels = labels
        self.preprocessor = ImagePreprocessor(image_size)
        self.transform = self.preprocessor.get_transforms(is_training=is_training)

        logger.info(
            f"Initialized MultiModalDermoscopicDataset [Size: {len(self.image_paths)} samples | "
            f"Training mode: {is_training} | Image size: {image_size}]"
        )

    def __len__(self) -> int:
        """
        Returns total number of samples in the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single multi-modal data item.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (image_tensor, metadata_tensor, label_tensor)
        """
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Dermoscopic image not found at: {img_path}")

        # 1. Load image and transform
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            image_tensor = self.transform(img)

        # 2. Extract preprocessed metadata features
        metadata_tensor = self.metadata_tensor[idx]

        # 3. Extract label
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_tensor, metadata_tensor, label_tensor
