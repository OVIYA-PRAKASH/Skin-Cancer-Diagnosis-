"""
Phase 4: Image Preprocessing.
Defines medical image augmentation and preprocessing pipelines using torchvision transforms,
and implements a custom PyTorch Dataset class for loading and preprocessing dermoscopic images.
"""

import os
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple

logger = logging.getLogger("SkinCancerAI.ImagePreprocessing")


class ImagePreprocessor:
    """
    Constructs torchvision transform pipelines for dermoscopic images.
    Applies data augmentation during training to prevent overfitting and ensure rotation/scale invariance,
    and applies standard scaling/normalization during evaluation.
    """

    def __init__(self, image_size: int):
        """
        Initializes the preprocessor.

        Args:
            image_size (int): Resolution to resize images to (typically 224 for pretrained backbones).
        """
        self.image_size = image_size
        # Standard ImageNet normalization parameters used by pretrained torchvision backbones
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def get_transforms(self, is_training: bool = True) -> transforms.Compose:
        """
        Generates the transformation pipeline.

        Args:
            is_training (bool): If True, returns augmentation + normalizations.
                                If False, returns scaling + normalizations only.

        Returns:
            transforms.Compose: Composed torchvision transforms.
        """
        if is_training:
            # Training augmentations specialized for dermatoscopic skin lesions
            # Lesions have no fixed orientation (rotation/reflection invariant)
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Rotate by any angle since medical camera orientation is arbitrary
                transforms.RandomRotation(degrees=90),
                # Simulates differences in dermatoscopic lens lighting, focus, and camera sensors
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        else:
            # Evaluation/Testing pipeline: deterministic resizing and normalization only
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])


class DermoscopicImageDataset(Dataset):
    """
    A custom PyTorch Dataset that loads dermoscopic images from disk,
    applies preprocessor transformations, and maps diagnoses labels to integers.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        image_size: int,
        is_training: bool = True
    ):
        """
        Initializes the dataset.

        Args:
            image_paths (list[str]): Absolute disk paths of the images.
            labels (list[int]): Encoded integer label targets for each image.
            image_size (int): Target dimensions for scaling.
            is_training (bool): Whether this dataset is used for training.
        """
        assert len(image_paths) == len(labels), "Number of image paths and labels must be equal."
        
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = ImagePreprocessor(image_size)
        self.transform = self.preprocessor.get_transforms(is_training=is_training)

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads, transforms, and returns a single dataset item.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image_tensor, label_tensor)
        
        Raises:
            FileNotFoundError: If the image file does not exist on disk.
        """
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Dermoscopic image file not found at: {img_path}")
            
        # Load image and convert to RGB (some raw files might be grayscale or RGBA)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            # Apply composed transforms (Resize, Augmentations, Normalize, Tensor conversion)
            img_tensor = self.transform(img)

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor
