"""
Phase 6: CNN Model.
Defines the CNN feature extraction backbone wrapper using PyTorch torchvision models.
Includes step-by-step explanations of the neural layers for beginners.
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any

logger = logging.getLogger("SkinCancerAI.CNNModel")


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor wrapper. Instantiates standard backbones,
    removes their final classification head, and exposes the global pooled feature vectors.
    """

    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True):
        """
        Initializes the CNN backbone.

        Args:
            backbone_name (str): Core architecture name ("efficientnet_b0", "resnet18", "resnet34").
            pretrained (bool): If True, loads pre-trained ImageNet weights.
        """
        super().__init__()
        self.backbone_name = backbone_name.lower().strip()
        self.pretrained = pretrained
        
        logger.info(f"Initializing CNN backbone: {self.backbone_name} (Pretrained: {self.pretrained})")

        # Select weights enum based on configuration
        # ImageNet1K_V1 is the standard pre-trained weight set for torchvision backbones
        cnn_weights = "DEFAULT" if self.pretrained else None

        if self.backbone_name == "efficientnet_b0":
            # 1. Load base EfficientNet model
            base_model = models.efficientnet_b0(weights=cnn_weights)
            # EfficientNet-B0 outputs feature maps with 1280 channels before classification
            self.feature_dim = 1280
            
            # 2. Extract feature extraction layers (exclude avgpool and classifier classifier)
            # base_model.features contains all convolutional layers and MBConv blocks
            self.features = base_model.features
            # 3. Add adaptive average pooling to compress 2D spatial feature maps into a 1D vector
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.backbone_name in ["resnet18", "resnet34"]:
            # ResNet utilizes a residual block architecture
            if self.backbone_name == "resnet18":
                base_model = models.resnet18(weights=cnn_weights)
                self.feature_dim = 512
            else:
                base_model = models.resnet34(weights=cnn_weights)
                self.feature_dim = 512
            
            # Extract layers from ResNet (exclude fc classification layer)
            # In PyTorch, we can group ResNet layers by removing the last elements
            self.features = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            self.pool = base_model.avgpool

        else:
            raise ValueError(f"Unsupported CNN backbone architecture: {backbone_name}")

        logger.info(f"CNN feature extraction backbone initialized. Output dimension size: {self.feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward feature extraction pass.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, 3, height, width].

        Returns:
            torch.Tensor: Flattened feature tensor of shape [batch_size, feature_dim].
        """
        # 1. Pass image through base convolutional layers (extract 2D feature maps)
        # Shape output: [batch_size, feature_dim, H_pooled, W_pooled]
        x = self.features(x)
        
        # 2. Average spatial dimensions to compile local properties
        # Shape output: [batch_size, feature_dim, 1, 1]
        x = self.pool(x)
        
        # 3. Flatten the spatial dimensions into a simple 1D vector per sample
        # Shape output: [batch_size, feature_dim]
        x = torch.flatten(x, start_dim=1)
        
        return x


# =====================================================================
# LAYERS EXPLANATION FOR BEGINNERS (EfficientNet-B0)
# =====================================================================
#
# If you are new to deep learning, here is how the convolutional neural network
# processes your skin lesion images layer-by-layer:
#
# 1. Input Image:
#    - A tensor of shape [Batch, 3, 224, 224], where 3 represents color channels (RGB)
#      and 224x224 represents the height and width of the image.
#
# 2. Stem Convolution Layer (Standard Conv2D + BatchNorm + Swish Activation):
#    - Why it is needed: The first layer acts as the "receptors" or "detectors".
#    - What it does: Slides small filters (size 3x3) over the pixels to detect basic
#      visual features like sharp borders, transitions from light to dark skin, and colors.
#    - Batch Normalization stabilizes training by scaling the pixel values of mini-batches.
#    - Swish (SiLU) is the activation function: it introduces non-linearity, allowing the
#      network to learn complex curves instead of simple straight relationships.
#
# 3. Mobile Inverted Bottleneck Conv blocks (MBConv Blocks):
#    - EfficientNet-B0 has 16 of these blocks stacked on top of each other.
#    - Depthwise Separable Convolution: Traditional convolutions look at all colors
#      and spatial pixels simultaneously, which takes high memory. MBConv splits this
#      into two steps: (a) Depthwise Conv filters each color channel independently, and
#      (b) Pointwise (1x1) Conv mixes these channels. This saves VRAM on your GTX 1650.
#    - Squeeze-and-Excitation (SE) Block: A mini-attention layer. It squeezes the
#      convolution channels to a single vector, decides which channels contain important diagnostic
#      clues (like pigment networks) and excites (multiplies) them, while muting unimportant ones.
#    - Residual Shortcuts: Adds the input of the block directly to the output. This forms
#      a highway that allows gradients to flow backwards without fading, preventing the vanishing
#      gradient problem during backpropagation.
#
# 4. Global Adaptive Average Pooling (nn.AdaptiveAvgPool2d):
#    - Why it is needed: The final MBConv block outputs 1280 feature maps, each of size 7x7.
#      We need to summarize these maps into a single vector.
#    - What it does: Computes the average value of all pixels in each 7x7 grid.
#    - Output: Compresses [1280, 7, 7] down to [1280, 1, 1], representing the strength
#      of all 1280 diagnostic feature elements.
#
# 5. Flatten (torch.flatten):
#    - Simply drops the extra dimensions, converting the shape [1280, 1, 1] into a simple
#      1D vector of length 1280 per patient image, ready to be joined with demographic features.
# =====================================================================
