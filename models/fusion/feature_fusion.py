"""
Phase 8: Feature Fusion.
Defines the multi-modal feature fusion block. Projects CNN, ViT, and patient demographics
features into aligned coordinate spaces, concatenates them, and passes them through
a joint dense bottleneck layer. Includes beginner-friendly explanations.
"""

import logging
import torch
import torch.nn as nn
from typing import Tuple

logger = logging.getLogger("SkinCancerAI.FeatureFusion")


class MultiModalFeatureFusion(nn.Module):
    """
    Fuses visual features from CNN & ViT backbones with demographic patient metadata.
    Uses linear projection layers, batch normalization, and dropout for regularization.
    """

    def __init__(
        self,
        cnn_dim: int = 1280,
        vit_dim: int = 768,
        meta_dim: int = 19,
        visual_proj_dim: int = 512,
        meta_embed_dim: int = 128,
        joint_dim: int = 256,
        dropout_prob: float = 0.3
    ):
        """
        Initializes the feature fusion module.

        Args:
            cnn_dim (int): Input dimensions from CNN (1280 for EfficientNet-B0).
            vit_dim (int): Input dimensions from ViT (768 for ViT-B/16).
            meta_dim (int): Input dimensions from demographics (19 for HAM10000 preprocessed).
            visual_proj_dim (int): Aligned shared dimension size for CNN and ViT.
            meta_embed_dim (int): Higher-dimensional representation size for patient demographics.
            joint_dim (int): Output bottleneck dimensions of the combined features.
            dropout_prob (float): Percentage of neurons to randomly drop out.
        """
        super().__init__()
        
        logger.info(
            f"Initializing MultiModalFeatureFusion: CNN({cnn_dim}) | ViT({vit_dim}) | Meta({meta_dim}) "
            f"-> Shared Visual Proj: {visual_proj_dim} | Meta Embed: {meta_embed_dim} -> Joint Bottleneck: {joint_dim}"
        )

        # 1. CNN Feature Projection: Aligns CNN features to the shared visual space
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, visual_proj_dim),
            nn.BatchNorm1d(visual_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 2. ViT Feature Projection: Aligns ViT features to the shared visual space
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, visual_proj_dim),
            nn.BatchNorm1d(visual_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 3. Metadata Embedding: Scales demographics to prevent visual modality dominance
        self.meta_embed = nn.Sequential(
            nn.Linear(meta_dim, meta_embed_dim),
            nn.BatchNorm1d(meta_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 4. Joint Bottleneck Representation: Compresses the concatenated multi-modal vectors
        # Total concatenated dimension = CNN projection + ViT projection + Demographics Embedding
        concat_dim = (2 * visual_proj_dim) + meta_embed_dim
        self.joint_bottleneck = nn.Sequential(
            nn.Linear(concat_dim, joint_dim),
            nn.BatchNorm1d(joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        self.output_dim = joint_dim
        logger.info("MultiModalFeatureFusion block successfully initialized.")

    def forward(
        self,
        cnn_feats: torch.Tensor,
        vit_feats: torch.Tensor,
        meta_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs forward multi-modal feature fusion pass.

        Args:
            cnn_feats (torch.Tensor): CNN features of shape [batch_size, cnn_dim].
            vit_feats (torch.Tensor): ViT features of shape [batch_size, vit_dim].
            meta_feats (torch.Tensor): Metadata features of shape [batch_size, meta_dim].

        Returns:
            torch.Tensor: Aligned, joint bottleneck tensor of shape [batch_size, joint_dim].
        """
        # 1. Project individual features to their respective aligned spaces
        projected_cnn = self.cnn_proj(cnn_feats)       # Shape: [batch_size, visual_proj_dim]
        projected_vit = self.vit_proj(vit_feats)       # Shape: [batch_size, visual_proj_dim]
        embedded_meta = self.meta_embed(meta_feats)    # Shape: [batch_size, meta_embed_dim]

        # 2. Concatenate all modalities along the feature dimension
        # Output shape: [batch_size, (2 * visual_proj_dim) + meta_embed_dim]
        fused_vector = torch.cat((projected_cnn, projected_vit, embedded_meta), dim=1)

        # 3. Pass concatenated features through the joint bottleneck mapping
        # Output shape: [batch_size, joint_dim]
        joint_representation = self.joint_bottleneck(fused_vector)

        return joint_representation


# =====================================================================
# FEATURE FUSION BLOCK LAYERS EXPLANATION FOR BEGINNERS
# =====================================================================
#
# Our system uses images and demographics. Mixing these formats requires
# aligning their coordinates, balancing their weights, and regulating neurons:
#
# 1. Projection Layers (nn.Linear):
#    - Why it is needed: EfficientNet features (size 1280) and ViT CLS features (size 768)
#      are calculated using completely different algorithms. Concatenating them directly is
#      mathematically imbalanced and leads to optimization difficulties.
#    - What it does: Runs a matrix multiplication ($y = xW^T + b$) to project both visual
#      representations into a shared coordinate space of size 512. Think of this as translating
#      two different languages into a common dialogue.
#
# 2. Metadata Embedding MLP (`self.meta_embed`):
#    - Why it is needed: Tabular patient demographics contain only 19 binary/scaled numbers.
#      If mixed with 2048 visual values, the training backpropagation will completely ignore
#      the patient metadata (this is called "modality dominance").
#    - What it does: Embeds the 19 features into a dense 128-dimensional latent vector,
#      amplifying patient details so they can compete and integrate with the visual maps.
#
# 3. Batch Normalization (nn.BatchNorm1d):
#    - Why it is needed: As data flows through deep layers, the range of feature values can
#      shift widely batch-to-batch, destabilizing learning (internal covariate shift).
#    - What it does: Scales the outputs of each batch to have a mean of 0 and a standard
#      deviation of 1. This acts as a stabilizer, letting us use higher learning rates safely.
#
# 4. Rectified Linear Unit (nn.ReLU):
#    - What it does: Converts negative values to 0 while keeping positive values unchanged:
#      $f(x) = \max(0, x)$. This introduces non-linearity, allowing the network to capture
#      complex patterns.
#
# 5. Dropout (nn.Dropout):
#    - Why it is needed: Skin cancer images are relatively limited, making models prone to
#      memorizing training files (overfitting).
#    - What it does: Temporarily "deactivates" 30% of random neurons during each training
#      step. This forces the model to build redundant diagnostic pathways rather than relying
#      on a single combination of features.
#
# 6. Joint Bottleneck (`self.joint_bottleneck`):
#    - Combines the visual and demographic features ($512 + 512 + 128 = 1152$ dimensions) and
#      compresses them into a final dense vector of size 256. This represents the ultimate
#      clinical diagnostic state of the patient.
# =====================================================================
