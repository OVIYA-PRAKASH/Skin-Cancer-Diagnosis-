"""
Phases 1, 2, 3, 10: Multi-Stage Validation Pipeline.
Defines abstract PipelineStage interface and concrete implementation classes for
image validation, skin detection, and skin lesion presence validation.
"""

from abc import ABC, abstractmethod
import os
import numpy as np
import cv2
from PIL import Image
from utils.config import Config
from utils.loggers import get_validation_logger

# Initialize logger
logger = None

def _get_logger(config: Config):
    global logger
    if logger is None:
        logger = get_validation_logger(config)
    return logger


class PipelineStage(ABC):
    """Abstract base class representing a single validation or detection stage (Phase 10)."""

    @abstractmethod
    def process(self, image: Image.Image, config: Config) -> dict:
        """
        Executes stage check logic.

        Args:
            image (PIL.Image.Image): Input raw image context.
            config (Config): Configuration containing thresholds.

        Returns:
            dict: {
                "passed": bool,
                "message": str,
                "metrics": dict
            }
        """
        pass


class ImageValidator(PipelineStage):
    """Phase 1: Validates general image formats, resolutions, sharpness, and rules out screenshots/cartoons."""

    def process(self, image: Image.Image, config: Config) -> dict:
        log = _get_logger(config)
        log.info("Starting Phase 1: Image Validation")

        # 1. Format and readability check
        try:
            # Ensure we can convert to RGB
            rgb_img = image.convert("RGB")
            img_np = np.array(rgb_img)
        except Exception as e:
            msg = "Invalid image. Please upload a readable image file."
            log.warning(f"Validation FAILED - Read/Convert Error: {str(e)}")
            return {"passed": False, "message": msg, "metrics": {"error": str(e)}}

        # 2. Dimensions check
        width, height = image.size
        min_w, min_h = config.validation.min_width, config.validation.min_height
        max_w, max_h = config.validation.max_width, config.validation.max_height

        if width < min_w or height < min_h:
            msg = f"Image resolution too low ({width}x{height}). Minimum required: {min_w}x{min_h}."
            log.warning(f"Validation FAILED - Resolution Underflow: {width}x{height}")
            return {"passed": False, "message": msg, "metrics": {"width": width, "height": height}}

        if width > max_w or height > max_h:
            msg = f"Image resolution too high ({width}x{height}). Maximum allowed: {max_w}x{max_h}."
            log.warning(f"Validation FAILED - Resolution Overflow: {width}x{height}")
            return {"passed": False, "message": msg, "metrics": {"width": width, "height": height}}

        # 3. Solid Color checks (Completely black/white)
        # Compute standard deviation across all color channels
        channel_stds = [np.std(img_np[:, :, c]) for c in range(3)]
        mean_std = np.mean(channel_stds)
        std_thresh = config.validation.solid_color_std_threshold

        if mean_std < std_thresh:
            msg = "Invalid image. The image is completely uniform (solid color) and cannot be processed."
            log.warning(f"Validation FAILED - Solid Color: Standard deviation {mean_std:.2f} < {std_thresh}")
            return {"passed": False, "message": msg, "metrics": {"mean_std": float(mean_std)}}

        # 4. Blur Check (Laplacian Variance)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_thresh = config.validation.blur_threshold

        if laplacian_var < blur_thresh:
            msg = "The image is too blurry. Please upload a clear dermoscopic skin lesion image."
            log.warning(f"Validation FAILED - Blur Detected: Variance {laplacian_var:.2f} < {blur_thresh}")
            return {"passed": False, "message": msg, "metrics": {"laplacian_variance": float(laplacian_var)}}

        # 5. Screenshot / Document / Cartoon filter heuristics
        # A. Unique colors ratio (downscaled check using NumPy)
        small_img = image.resize((64, 64))
        unique_colors = len(np.unique(np.array(small_img).reshape(-1, 3), axis=0))
        
        # B. Edge orientation alignment check (Sobel gradients)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        strong_edges = magnitude > 50.0

        edge_ratio = 0.0
        if np.sum(strong_edges) > 50:
            angles = np.abs(np.arctan2(sobely, sobelx)) * 180.0 / np.pi
            # Counts vertical (around 90) and horizontal (around 0/180) edges
            horiz_vert = np.sum(strong_edges & ((angles < 15) | (angles > 165) | ((angles > 75) & (angles < 105))))
            other = np.sum(strong_edges & ~((angles < 15) | (angles > 165) | ((angles > 75) & (angles < 105))))
            
            if other > 0:
                edge_ratio = horiz_vert / other

        # Documents, screenshots, and cartoons have low color variance or highly aligned borders (high edge_ratio)
        is_cartoon_or_ui = (unique_colors < 120)
        is_screenshot_or_doc = (edge_ratio > 4.5)

        if is_cartoon_or_ui or is_screenshot_or_doc:
            msg = "Invalid image. Please upload a dermoscopic skin lesion image."
            log.warning(
                f"Validation FAILED - Heuristic Check: "
                f"Unique Colors (64x64): {unique_colors} (is_cartoon: {is_cartoon_or_ui}), "
                f"Edge Ratio (H+V/Other): {edge_ratio:.2f} (is_screenshot: {is_screenshot_or_doc})"
            )
            return {
                "passed": False, 
                "message": msg, 
                "metrics": {
                    "unique_colors_64": unique_colors,
                    "edge_ratio": float(edge_ratio)
                }
            }

        log.info(
            f"Phase 1: Image Validation PASSED. "
            f"Size: {width}x{height}, StdDev: {mean_std:.2f}, BlurVar: {laplacian_var:.2f}"
        )
        return {
            "passed": True,
            "message": "Image validation passed.",
            "metrics": {
                "width": width,
                "height": height,
                "mean_std": float(mean_std),
                "laplacian_variance": float(laplacian_var),
                "unique_colors_64": unique_colors,
                "edge_ratio": float(edge_ratio)
            }
        }


class SkinDetector(PipelineStage):
    """Phase 2: Detects whether human skin color ranges are present in the image."""

    def process(self, image: Image.Image, config: Config) -> dict:
        log = _get_logger(config)
        log.info("Starting Phase 2: Skin Detection")

        try:
            img_np = np.array(image.convert("RGB"))
            ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        except Exception as e:
            log.warning(f"Skin Detection FAILED - Conversion Error: {str(e)}")
            return {"passed": False, "message": "Failed to process skin colors.", "metrics": {"error": str(e)}}

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        # Standard skin detection ranges in YCrCb color space
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_pixels = np.sum(skin_mask)
        total_pixels = skin_mask.size
        skin_ratio = skin_pixels / total_pixels
        skin_thresh = config.validation.skin_threshold

        if skin_ratio < skin_thresh:
            msg = "No skin detected. Please upload an image containing visible skin area."
            log.warning(f"Skin Detection FAILED - Skin ratio {skin_ratio:.4f} < {skin_thresh}")
            return {"passed": False, "message": msg, "metrics": {"skin_ratio": float(skin_ratio)}}

        log.info(f"Phase 2: Skin Detection PASSED. Skin Ratio: {skin_ratio*100:.2f}%")
        return {
            "passed": True,
            "message": "Skin detected successfully.",
            "metrics": {
                "skin_ratio": float(skin_ratio)
            }
        }


class LesionDetector(PipelineStage):
    """Phase 3: Validates if a high-contrast skin lesion contour exists in the image context."""

    def process(self, image: Image.Image, config: Config) -> dict:
        log = _get_logger(config)
        log.info("Starting Phase 3: Lesion Detection")

        try:
            img_np = np.array(image.convert("RGB"))
            ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            log.warning(f"Lesion Detection FAILED - Conversion Error: {str(e)}")
            return {"passed": False, "message": "Failed to process lesion contours.", "metrics": {"error": str(e)}}

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        
        # 1. Regenerate skin mask
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_pixels_count = np.sum(skin_mask)

        if skin_pixels_count == 0:
            log.warning("Lesion Detection FAILED - No skin pixels found for masking")
            return {
                "passed": False, 
                "message": "No skin detected, cannot verify lesion presence.", 
                "metrics": {"skin_ratio": 0.0}
            }

        # 2. Local contrast thresholding within skin mask
        skin_pixels = gray[skin_mask]
        mean_skin = np.mean(skin_pixels)
        std_skin = np.std(skin_pixels)

        # Lesions are darker/pigmented spots compared to surrounding skin.
        # Find pixels inside the skin mask that are darker than the local skin average.
        thresh_val = mean_skin - max(0.7 * std_skin, 10.0)
        lesion_mask = skin_mask & (gray < thresh_val)

        # 3. Find contours
        contours, _ = cv2.findContours(lesion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area_ratio = 0.0
        width, height = image.size
        img_area = width * height

        for c in contours:
            area = cv2.contourArea(c)
            ratio = area / img_area
            max_area_ratio = max(max_area_ratio, ratio)

        lesion_thresh = config.validation.lesion_threshold

        if max_area_ratio < lesion_thresh:
            msg = "No visible skin lesion detected. Please upload an image containing a skin lesion."
            log.warning(f"Lesion Detection FAILED - Max lesion area ratio {max_area_ratio:.4f} < {lesion_thresh}")
            return {"passed": False, "message": msg, "metrics": {"max_lesion_area_ratio": float(max_area_ratio)}}

        log.info(f"Phase 3: Lesion Detection PASSED. Max Lesion Area Ratio: {max_area_ratio*100:.2f}%")
        return {
            "passed": True,
            "message": "Skin lesion contour detected successfully.",
            "metrics": {
                "max_lesion_area_ratio": float(max_area_ratio)
            }
        }

