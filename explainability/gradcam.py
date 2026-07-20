"""
Phase 14 & 15: Explainable AI (Grad-CAM and Confidence Scores).
Implements the Grad-CAM visualization engine to capture gradients and feature maps
from the final convolutional layer of the CNN backbone, producing heatmap overlays
that explain model predictions to clinicians.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("SkinCancerAI.GradCAM")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) engine.
    Applies hooks to capture spatial activations and backpropagated gradients
    to calculate activation heatmaps.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initializes Grad-CAM.

        Args:
            model (nn.Module): The integrated HybridMultiModalClassifier model.
            target_layer (nn.Module): Convolutional layer to hook (usually cnn_extractor.features[-1]).
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hook triggers
        # Forward hook saves convolutional feature maps
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        # Full backward hook saves backpropagated gradients
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

        logger.info("Grad-CAM hooks successfully registered on target layer.")

    def _save_activations(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Forward hook callback saving spatial feature activations."""
        self.activations = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
        """Backward hook callback saving gradients."""
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Cleans and removes hooks to prevent memory leaks in PyTorch runtime."""
        self.forward_hook.remove()
        self.backward_hook.remove()
        logger.info("Grad-CAM hooks removed.")

    def generate_heatmap(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Runs a forward/backward pass on a single sample, extracts activations,
        and generates a 2D clinical focus heatmap.

        Args:
            images (torch.Tensor): Preprocessed image tensor [1, 3, 224, 224].
            metadata (torch.Tensor): Patient demographics tensor [1, 19].
            class_idx (Optional[int]): Target class to explain. If None, uses the predicted class.

        Returns:
            Tuple[np.ndarray, int, float]: (heatmap_grid_float, predicted_class_idx, prediction_confidence)
        """
        self.model.eval()
        
        # 1. Forward pass
        logits = self.model(images, metadata)
        probs = torch.softmax(logits, dim=1)
        
        pred_idx = int(torch.argmax(logits, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

        if class_idx is None:
            class_idx = pred_idx

        # 2. Backward pass targeting the score of the selected class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients. Verify forward/backward pass execution.")

        # 3. Calculate channel weights
        # Spatially average the gradients for each channel (Global Average Pooling on gradients)
        # Shape activations: [1, Channels, Height, Width]
        # Shape gradients: [1, Channels, Height, Width]
        gradients = self.gradients[0]  # Remove batch dim -> [C, H, W]
        activations = self.activations[0]  # Remove batch dim -> [C, H, W]
        
        # Mean along H and W dimensions
        weights = torch.mean(gradients, dim=(1, 2))  # Shape: [C]

        # 4. Compute weighted sum of spatial activations
        # Multiply each activation map by its gradient weight
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # 5. Apply ReLU to isolate positive feature correlations
        # Convert to numpy array and scale to range [0, 1]
        cam = torch.clamp(cam, min=0.0)
        cam_np = cam.cpu().numpy()
        
        # Prevent division by zero if all values are zero
        max_val = np.max(cam_np)
        if max_val > 0.0:
            cam_np = cam_np / max_val

        # Resize the 2D heatmap back to match original image dimensions using Pillow
        cam_pil = Image.fromarray(cam_np)
        cam_pil_resized = cam_pil.resize((images.shape[3], images.shape[2]), resample=Image.BILINEAR)
        heatmap_resized = np.array(cam_pil_resized, dtype=np.float32)
        
        return heatmap_resized, pred_idx, confidence

    @staticmethod
    def overlay_heatmap(
        image_np: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap_name: str = "jet"
    ) -> np.ndarray:
        """
        Blends the 2D heatmap overlay with the original lesion image.

        Args:
            image_np (np.ndarray): Original image (RGB format, values [0, 255] or [0, 1]).
            heatmap (np.ndarray): Normalized 2D Grad-CAM heatmap [0, 1].
            alpha (float): Transparency blending multiplier for the overlay.
            colormap_name (str): Matplotlib colormap name. Defaults to "jet".

        Returns:
            np.ndarray: Blended RGB image.
        """
        # Ensure image is in float range [0.0, 1.0]
        if image_np.max() > 1.0:
            image_val = image_np.astype(np.float32) / 255.0
        else:
            image_val = image_np.astype(np.float32)

        # Get Matplotlib colormap
        colormap = plt.colormaps.get_cmap(colormap_name)
        # Apply colormap -> outputs RGBA float [0, 1]
        colored_heatmap_rgba = colormap(heatmap)
        # Extract RGB channels
        colored_heatmap = colored_heatmap_rgba[:, :, :3].astype(np.float32)

        # Blend original image and colored heatmap
        blended = (1.0 - alpha) * image_val + alpha * colored_heatmap
        
        # Scale back to [0, 255]
        blended_uint8 = (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        return blended_uint8
