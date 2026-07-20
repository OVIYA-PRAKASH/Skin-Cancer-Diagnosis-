"""
Unit tests for the multi-stage validation pipeline stages.
Verifies Phase 1 ImageValidator parameters and heuristics.
"""

import os
import unittest
import numpy as np
from PIL import Image, ImageDraw
from utils.config import Config
from utils.pipeline_stages import ImageValidator, SkinDetector, LesionDetector


class TestPipelineStages(unittest.TestCase):
    def setUp(self):
        # Create a mock config block matching default configs
        self.config = Config.load_from_yaml("configs/default_config.yaml")
        # Ensure thresholds are standard for testing
        self.config.validation.min_width = 100
        self.config.validation.min_height = 100
        self.config.validation.max_width = 4000
        self.config.validation.max_height = 4000
        self.config.validation.blur_threshold = 10.0  # low for test ease
        self.config.validation.solid_color_std_threshold = 5.0
        
        self.validator = ImageValidator()
        self.skin_detector = SkinDetector()
        self.lesion_detector = LesionDetector()

    def test_valid_image_passes(self):
        """Test that a standard organic-looking gradient image passes validation."""
        # Create a 200x200 image with a soft gradient
        img = Image.new("RGB", (200, 200), color=(150, 100, 80))
        draw = ImageDraw.Draw(img)
        # Draw some organic circular shapes to simulate a lesion texture
        draw.ellipse([50, 50, 150, 150], fill=(100, 50, 40), outline=(80, 40, 30))
        # Add noise to increase unique colors and blur variance
        img_np = np.array(img)
        noise = np.random.normal(0, 10, img_np.shape).astype(np.uint8)
        img = Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8))

        res = self.validator.process(img, self.config)
        self.assertTrue(res["passed"], f"Valid image failed: {res.get('message')}")

    def test_resolution_underflow_fails(self):
        """Test that a tiny image below minimum bounds fails."""
        img = Image.new("RGB", (50, 50), color=(150, 100, 80))
        res = self.validator.process(img, self.config)
        self.assertFalse(res["passed"])
        self.assertIn("resolution too low", res["message"])

    def test_resolution_overflow_fails(self):
        """Test that a giant image above maximum bounds fails."""
        img = Image.new("RGB", (5000, 5000), color=(150, 100, 80))
        res = self.validator.process(img, self.config)
        self.assertFalse(res["passed"])
        self.assertIn("resolution too high", res["message"])

    def test_solid_color_fails(self):
        """Test that a completely flat solid color image fails."""
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        res = self.validator.process(img, self.config)
        self.assertFalse(res["passed"])
        self.assertIn("solid color", res["message"])

    def test_screenshot_grid_fails(self):
        """Test that sharp orthogonal grids (simulating UI/screenshots) fail edge alignment check."""
        img = Image.new("RGB", (200, 200), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Draw highly aligned grid vertical and horizontal lines
        for i in range(0, 200, 20):
            draw.line([(i, 0), (i, 200)], fill=(0, 0, 0), width=2)
            draw.line([(0, i), (200, i)], fill=(0, 0, 0), width=2)
        
        res = self.validator.process(img, self.config)
        self.assertFalse(res["passed"])
        self.assertIn("Please upload a dermoscopic", res["message"])

    def test_cartoon_flat_colors_fails(self):
        """Test that simple cartoons with very few colors fail unique color count."""
        # Create a drawing with only 3 flat uniform solid zones
        img = Image.new("RGB", (200, 200), color=(255, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill=(0, 255, 0))
        draw.ellipse([80, 80, 120, 120], fill=(0, 0, 255))

        res = self.validator.process(img, self.config)
        self.assertFalse(res["passed"])
        self.assertIn("Please upload a dermoscopic", res["message"])

    def test_skin_detector_passes_on_skin(self):
        """Test skin detector successfully flags skin-toned pixels."""
        # Skin color in YCrCb: Cr=154.6, Cb=109.6 (RGB: 210, 160, 140)
        img = Image.new("RGB", (200, 200), color=(210, 160, 140))
        res = self.skin_detector.process(img, self.config)
        self.assertTrue(res["passed"], f"Skin image failed: {res.get('message')}")
        self.assertGreaterEqual(res["metrics"]["skin_ratio"], self.config.validation.skin_threshold)

    def test_skin_detector_fails_on_non_skin(self):
        """Test skin detector flags landscape or non-skin pixels."""
        # Deep blue sky color RGB: 50, 100, 200
        img = Image.new("RGB", (200, 200), color=(50, 100, 200))
        res = self.skin_detector.process(img, self.config)
        self.assertFalse(res["passed"])
        self.assertIn("No skin detected", res["message"])

    def test_lesion_detector_passes_on_lesion(self):
        """Test lesion detector spots pigmented lesion area inside skin boundaries."""
        # Create skin image with a dark circular spot (lesion)
        img = Image.new("RGB", (200, 200), color=(210, 160, 140))
        draw = ImageDraw.Draw(img)
        draw.ellipse([40, 40, 160, 160], fill=(80, 40, 30))  # dark pigmented spot
        res = self.lesion_detector.process(img, self.config)
        self.assertTrue(res["passed"], f"Lesion image failed: {res.get('message')}")
        self.assertGreaterEqual(res["metrics"]["max_lesion_area_ratio"], self.config.validation.lesion_threshold)

    def test_lesion_detector_fails_on_clean_skin(self):
        """Test lesion detector fails on plain, unblemished skin."""
        img = Image.new("RGB", (200, 200), color=(210, 160, 140))
        # Add subtle noise but no dark spots
        img_np = np.array(img)
        noise = np.random.normal(0, 2, img_np.shape).astype(np.uint8)
        img = Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8))
        
        res = self.lesion_detector.process(img, self.config)
        self.assertFalse(res["passed"])
        self.assertIn("No visible skin lesion", res["message"])


if __name__ == "__main__":
    unittest.main()
