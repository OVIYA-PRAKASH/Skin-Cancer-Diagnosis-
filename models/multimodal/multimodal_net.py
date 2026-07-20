"""
Phase 9: Multi-modal Network.
Defines the final integrated classifier network HybridMultiModalClassifier.
Combines CNN backbones, ViT backbones, demographics embedding, and feature fusion,
and maps the joint representation to the final 7-class diagnostic logits.
Includes beginner-friendly layer explanations.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from models.cnn.cnn_backbone import CNNFeatureExtractor
from models.transformer.vit_backbone import ViTFeatureExtractor
from models.fusion.feature_fusion import MultiModalFeatureFusion

logger = logging.getLogger("SkinCancerAI.MultiModalNet")


class HybridMultiModalClassifier(nn.Module):
    """
    The complete Hybrid CNN + Vision Transformer + Metadata Multi-Modal Classifier.
    Integrates visual feature extractors and demographic embeddings to output 
    lesion class logits. Supports ablation studies via configuration flags.
    """

    def __init__(
        self,
        num_classes: int = 7,
        cnn_backbone: str = "efficientnet_b0",
        vit_backbone: str = "vit_b_16",
        pretrained: bool = True,
        use_cnn: bool = True,
        use_vit: bool = True,
        use_metadata: bool = True,
        visual_proj_dim: int = 512,
        meta_embed_dim: int = 128,
        joint_dim: int = 256,
        dropout_prob: float = 0.3
    ):
        """
        Initializes the multi-modal classifier.

        Args:
            num_classes (int): Number of diagnostic target classes (7 for HAM10000).
            cnn_backbone (str): CNN architecture name.
            vit_backbone (str): ViT architecture name.
            pretrained (bool): If True, loads pre-trained ImageNet weights.
            use_cnn (bool): If False, disables and zero-masks CNN features (Ablation).
            use_vit (bool): If False, disables and zero-masks ViT features (Ablation).
            use_metadata (bool): If False, disables and zero-masks demographic features (Ablation).
            visual_proj_dim (int): Aligned shared dimension size for visual backbones.
            meta_embed_dim (int): Dimension size for patient demographics embedding.
            joint_dim (int): Bottleneck dimension of combined features.
            dropout_prob (float): Regularization dropout probability.
        """
        super().__init__()
        
        self.use_cnn = use_cnn
        self.use_vit = use_vit
        self.use_metadata = use_metadata
        self.num_classes = num_classes

        logger.info(
            f"Building HybridMultiModalClassifier [Ablation Flags -> Use CNN: {self.use_cnn} | "
            f"Use ViT: {self.use_vit} | Use Metadata: {self.use_metadata}]"
        )

        # 1. Instantiate CNN feature extractor if enabled
        if self.use_cnn:
            self.cnn_extractor = CNNFeatureExtractor(backbone_name=cnn_backbone, pretrained=pretrained)
            cnn_dim = self.cnn_extractor.feature_dim
        else:
            cnn_dim = 1280  # Default feature size for zero-masking shape matching

        # 2. Instantiate ViT feature extractor if enabled
        if self.use_vit:
            self.vit_extractor = ViTFeatureExtractor(backbone_name=vit_backbone, pretrained=pretrained)
            vit_dim = self.vit_extractor.feature_dim
        else:
            vit_dim = 768   # Default feature size for zero-masking shape matching

        # Demographics dimension size (HAM10000 standard is 19: 1 age + 3 sex + 15 locations)
        meta_dim = 19

        # 3. Instantiate Feature Fusion block
        self.fusion_block = MultiModalFeatureFusion(
            cnn_dim=cnn_dim,
            vit_dim=vit_dim,
            meta_dim=meta_dim,
            visual_proj_dim=visual_proj_dim,
            meta_embed_dim=meta_embed_dim,
            joint_dim=joint_dim,
            dropout_prob=dropout_prob
        )

        # 4. Final Classification Head: Maps joint representation to disease class logits
        self.classifier_head = nn.Linear(joint_dim, num_classes)
        
        logger.info(f"HybridMultiModalClassifier built successfully. Output logits count: {num_classes}")

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs forward multi-modal classification pass.

        Args:
            images (torch.Tensor): Input image tensor of shape [batch_size, 3, 224, 224].
            metadata (torch.Tensor): Patient demographics tensor of shape [batch_size, 19].

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes].
        """
        batch_size = images.shape[0]
        device = images.device

        # 1. Extract CNN features (or substitute with zeros if disabled for ablation study)
        if self.use_cnn:
            cnn_feats = self.cnn_extractor(images)
        else:
            cnn_feats = torch.zeros(batch_size, 1280, device=device)

        # 2. Extract ViT features (or substitute with zeros if disabled for ablation study)
        if self.use_vit:
            vit_feats = self.vit_extractor(images)
        else:
            vit_feats = torch.zeros(batch_size, 768, device=device)

        # 3. Handle patient demographics metadata (or substitute with zeros if disabled for ablation study)
        if self.use_metadata:
            meta_feats = metadata
        else:
            meta_feats = torch.zeros(batch_size, 19, device=device)

        # 4. Fuse all feature representations
        # Output shape: [batch_size, joint_dim] (typically 256 dimensions)
        joint_rep = self.fusion_block(cnn_feats, vit_feats, meta_feats)

        # 5. Project to final diagnostic disease scores (logits)
        # Output shape: [batch_size, num_classes] (typically 7 classes)
        logits = self.classifier_head(joint_rep)

        return logits


# =====================================================================
# MULTI-MODAL CLASSIFIER HEAD LAYERS EXPLANATION FOR BEGINNERS
# =====================================================================
#
# Here is how our final network integrates everything and performs diagnosis:
#
# 1. Image Forward Pass:
#    - The clinical image (224x224 RGB pixels) is passed in parallel to the CNN and ViT.
#    - The CNN (`self.cnn_extractor`) extracts local features (textures, lesion color).
#    - The ViT (`self.vit_extractor`) extracts global attention features (asymmetry, borders).
#
# 2. Demographic Forward Pass:
#    - Patient metadata (`metadata` tensor containing age, sex, and anatomical location)
#      is loaded.
#
# 3. Dynamic Feature Fusion:
#    - The three representations are sent to the `MultiModalFeatureFusion` block.
#    - If a feature is disabled (e.g. `use_metadata = False`), it is filled with 0s. This
#      lets us measure accuracy drops without needing different network models.
#    - The features are aligned to a shared dimension and concatenated into a 256-dimensional
#      joint vector summarizing the patient's condition.
#
# 4. Classification Head (`self.classifier_head`):
#    - Why it is needed: The 256-dimensional vector contains raw features. We must map these
#      features to the 7 skin cancer classes.
#    - What it does: A linear layer (fully connected layer) projects the 256 features to
#      7 score values (logits):
#      $$\text{logits} = xW^T + b$$
#      Each of the 7 values corresponds to one disease class:
#      [akiec, bcc, bkl, df, mel, nv, vasc]
#      The highest score determines the model's prediction. During training, we apply a
#      Softmax function to convert these logits into confidence percentages.
# =====================================================================
