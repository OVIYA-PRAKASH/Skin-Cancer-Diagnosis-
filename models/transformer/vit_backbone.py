"""
Phase 7: Vision Transformer.
Defines the Vision Transformer (ViT) feature extraction backbone wrapper using PyTorch.
Includes beginner-friendly structural and mathematical explanations of transformer layers.
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any

logger = logging.getLogger("SkinCancerAI.ViTModel")


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer feature extractor wrapper. Instantiates standard ViTs,
    removes their final classification heads, and outputs the raw 1D CLS token representation.
    """

    def __init__(self, backbone_name: str = "vit_b_16", pretrained: bool = True):
        """
        Initializes the ViT backbone.

        Args:
            backbone_name (str): Core architecture name ("vit_b_16", "vit_l_16").
            pretrained (bool): If True, loads pre-trained ImageNet weights.
        """
        super().__init__()
        self.backbone_name = backbone_name.lower().strip()
        self.pretrained = pretrained
        
        logger.info(f"Initializing ViT backbone: {self.backbone_name} (Pretrained: {self.pretrained})")

        # Select weights enum based on configuration
        vit_weights = "DEFAULT" if self.pretrained else None

        if self.backbone_name == "vit_b_16":
            # ViT-Base with 16x16 patch projection
            base_model = models.vit_b_16(weights=vit_weights)
            # ViT-Base CLS token hidden dimension size is 768
            self.feature_dim = 768
            
            # Extract standard layers from ViT
            self.conv_proj = base_model.conv_proj
            self.class_token = base_model.class_token
            self.encoder = base_model.encoder
            
        elif self.backbone_name == "vit_l_16":
            # ViT-Large with 16x16 patch projection
            base_model = models.vit_l_16(weights=vit_weights)
            # ViT-Large CLS token hidden dimension size is 1024
            self.feature_dim = 1024
            
            self.conv_proj = base_model.conv_proj
            self.class_token = base_model.class_token
            self.encoder = base_model.encoder
            
        else:
            raise ValueError(f"Unsupported ViT backbone architecture: {backbone_name}")

        logger.info(f"ViT feature extraction backbone initialized. Output dimension size: {self.feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward feature extraction pass.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, 3, height, width].

        Returns:
            torch.Tensor: Flattened CLS token feature representation of shape [batch_size, feature_dim].
        """
        # 1. Patchify and Project: Convert image into a grid of token embeddings
        # Input shape: [batch_size, 3, 224, 224]
        # output shape: [batch_size, feature_dim, 14, 14]
        x = self.conv_proj(x)
        
        # Flatten spatial dimensions: [batch_size, feature_dim, 196]
        # Transpose: [batch_size, 196, feature_dim] (196 visual tokens)
        x = x.flatten(2).transpose(1, 2)
        
        # 2. Prepend the learnable Class (CLS) token
        # Class token shape: [1, 1, feature_dim] -> repeat across the batch dimension
        batch_size = x.shape[0]
        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token with patch tokens: [batch_size, 197, feature_dim]
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add position embeddings and pass through self-attention layers
        # The encoder handles: Position Embeddings + Transformer Blocks + Layer Normalization
        x = self.encoder(x)
        
        # 4. Extract only the CLS token (index 0) representing the consolidated global image features
        # Output shape: [batch_size, feature_dim]
        cls_output = x[:, 0]
        
        return cls_output


# =====================================================================
# VISION TRANSFORMER (ViT-B/16) LAYERS EXPLANATION FOR BEGINNERS
# =====================================================================
#
# Unlike standard CNNs that look at small localized pixel neighborhoods,
# the Vision Transformer treats an image like a sentence of visual words:
#
# 1. Patch Projection (Patchification - `self.conv_proj`):
#    - Why it is needed: Self-attention calculates connections between every token.
#      If we did this at the pixel level (224x224 = 50,176 pixels), the memory calculation
#      would crash your GTX 1650 VRAM instantly.
#    - What it does: Breaks the 224x224 image into a grid of 14x14 = 196 small patches.
#      Each patch is 16x16 pixels with 3 color channels (size = 16x16x3 = 768 values).
#      A Conv2D layer with filter size 16x16 and stride 16 projects each patch directly
#      into a 768-dimensional vector, giving us 196 "visual words" (tokens).
#
# 2. Class Token (CLS Token - `self.class_token`):
#    - Why it is needed: Visual tokens only represent local patch information. We need
#      a single representative output that summarizes the entire skin lesion image.
#    - What it does: A learnable 768-dimensional vector is prepended to our 196 tokens.
#      As the data flows through attention blocks, this token "asks" all other tokens for
#      their details, aggregating the global clinical visual context.
#
# 3. Position Embeddings (Encoder Input Additions):
#    - Why it is needed: Transformers process all tokens simultaneously, meaning they
#      do not naturally know if a patch belongs to the top-left or bottom-right of the skin.
#    - What it does: Adds a unique set of learnable coordinates to each token.
#
# 4. Multi-Head Self-Attention (MHSA - Inside `self.encoder`):
#    - Why it is needed: To capture relationships across distant areas of the skin lesion
#      (e.g., comparing pigmentation on one side of a lesion with the edge on the other).
#    - What it does: For each visual token, it generates:
#        - Query (what information I'm looking for)
#        - Key (what information I contain)
#        - Value (my actual feature values)
#      By multiplying Queries with Keys, it draws attention scores (importance weights).
#      "Multi-Head" means this is run 12 times in parallel. One head may focus on lesion asymmetry,
#      another on boundary fuzziness, and a third on color variance.
#
# 5. Multi-Layer Perceptron (MLP) Blocks:
#    - Standard feed-forward fully-connected neural networks that process each token
#      individually after attention to refine their feature representation.
#
# 6. Global Features Output (`x[:, 0]`):
#    - We discard the 196 spatial patch tokens and extract only the first token (index 0).
#      This is the final CLS token vector of length 768, holding the global attention summary.
# =====================================================================
