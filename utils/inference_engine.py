"""
Phase 17: Inference Engine.
Implements the InferenceEngine class. Loads the serialized model weights and
demographics preprocessor, processes single-patient inputs, runs diagnostic classification,
computes Grad-CAM heatmaps, and saves self-contained HTML clinical summaries.
"""

import os
# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import logging
import pickle
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional

from utils.config import Config
from dataset.preprocessing import ImagePreprocessor
from dataset.metadata import MetadataPreprocessor
from models.multimodal.multimodal_net import HybridMultiModalClassifier
from explainability.gradcam import GradCAM
from reports.report_generator import ClinicalReportGenerator
from utils.pipeline_stages import ImageValidator, SkinDetector, LesionDetector
from utils.loggers import get_validation_logger, get_inference_logger, get_prediction_logger

# Define lazy loaded loggers for different channels
validation_logger = None
inference_logger = None
prediction_logger = None

def _get_val_log(config: Config):
    global validation_logger
    if validation_logger is None:
        validation_logger = get_validation_logger(config)
    return validation_logger

def _get_inf_log(config: Config):
    global inference_logger
    if inference_logger is None:
        inference_logger = get_inference_logger(config)
    return inference_logger

def _get_pred_log(config: Config):
    global prediction_logger
    if prediction_logger is None:
        prediction_logger = get_prediction_logger(config)
    return prediction_logger



class InferenceEngine:
    """
    Production-grade inference interface that loads pre-trained checkpoints
    and generates full multi-modal clinical diagnostic reports for single cases.
    """

    def __init__(
        self,
        config_path: str = "configs/default_config.yaml",
        model_path: str = "checkpoints/best_model.pth",
        preprocessor_path: str = "checkpoints/metadata_preprocessor.pkl",
        device: Optional[torch.device] = None
    ):
        """
        Initializes the inference engine.

        Args:
            config_path (str): Path to defaults configuration YAML.
            model_path (str): Path to saved model weights checkpoint.
            preprocessor_path (str): Path to serialized MetadataPreprocessor pickle file.
            device (Optional[torch.device]): Target device (defaults to auto-detect).
        """
        # 1. Load config settings
        self.config = Config.load_from_yaml(config_path)

        # 2. Select device context
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log = _get_inf_log(self.config)
        log.info(f"Inference Engine running on device: {self.device}")

        # Initialize modular validation pipeline stages (Phases 1-3, 10)
        self.validator = ImageValidator()
        self.skin_detector = SkinDetector()
        self.lesion_detector = LesionDetector()

        # 3. Load demographic preprocessor statistics
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Serialized metadata preprocessor not found at: {preprocessor_path}. "
                "Ensure training pipeline has completed successfully."
            )
        
        with open(preprocessor_path, "rb") as f:
            self.preprocessor: MetadataPreprocessor = pickle.load(f)
        log.info("Metadata preprocessor successfully loaded.")

        # Determine ablation modality usage flags based on configuration settings
        mode = self.config.ablation.mode.lower().strip()
        image_mode = self.config.ablation.image_mode.lower().strip()

        use_metadata = (mode in ["multimodal", "metadata_only"])
        if mode == "metadata_only":
            use_cnn = False
            use_vit = False
        else:
            use_cnn = (image_mode in ["hybrid", "cnn_only"])
            use_vit = (image_mode in ["hybrid", "vit_only"])

        # 4. Reconstruct and load model weights
        self.model = HybridMultiModalClassifier(
            num_classes=7,
            cnn_backbone=self.config.model.cnn_backbone,
            vit_backbone=self.config.model.transformer_backbone,
            pretrained=False,  # Weights will be loaded from checkpoint
            use_cnn=use_cnn,
            use_vit=use_vit,
            use_metadata=use_metadata,
            visual_proj_dim=512,  # Standard project dimension
            meta_embed_dim=self.config.model.metadata_embed_dim,
            joint_dim=self.config.model.fusion_dim,
            dropout_prob=self.config.model.dropout
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights checkpoint not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        log.info(f"Model weights successfully loaded from: {model_path}")

        # 5. Initialize image preprocessing transforms
        self.image_preprocessor = ImagePreprocessor(self.config.data.image_size)
        self.image_transform = self.image_preprocessor.get_transforms(is_training=False)

        # 6. Initialize Report Generator
        self.report_generator = ClinicalReportGenerator(output_dir=os.path.join(self.config.paths.output_dir, "reports"))
        
        # Mapping numerical indices back to clinical labels
        self.classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    def run_pipeline_validation(self, image_path: str) -> Tuple[bool, dict]:
        """
        Runs the multi-stage validation pipeline on the image.
        Returns a tuple: (is_valid, stages_results)
        """
        results = {
            "image_validation": {"passed": False, "message": "Pending", "metrics": {}},
            "skin_detection": {"passed": False, "message": "Pending", "metrics": {}},
            "lesion_detection": {"passed": False, "message": "Pending", "metrics": {}}
        }
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
            
        with Image.open(image_path) as pil_img:
            # 1. Image Validation
            res1 = self.validator.process(pil_img, self.config)
            results["image_validation"] = res1
            if not res1["passed"]:
                return False, results
                
            # 2. Skin Detection
            res2 = self.skin_detector.process(pil_img, self.config)
            results["skin_detection"] = res2
            if not res2["passed"]:
                return False, results
                
            # 3. Lesion Detection
            res3 = self.lesion_detector.process(pil_img, self.config)
            results["lesion_detection"] = res3
            if not res3["passed"]:
                return False, results
                
        return True, results

    def predict_and_explain(
        self,
        image_path: str,
        age: float,
        sex: str,
        localization: str,
        patient_id: str = "CASE_999",
        report_filename: str = "clinical_report.html"
    ) -> Tuple[str, Dict[str, float], str]:
        """
        Runs complete inference pass, generates Grad-CAM overlays, and compiles
        the clinical HTML diagnostic report.

        Args:
            image_path (str): Disk path to the raw dermoscopic image.
            age (float): Patient age in years.
            sex (str): Patient sex ("male", "female", "unknown").
            localization (str): Anatomical site (e.g. "back", "face").
            patient_id (str): Unique identifier for logging and reporting.
            report_filename (str): Name of the generated report file.

        Returns:
            Tuple[str, Dict[str, float], str]: (predicted_class_name, class_probabilities, html_report_path)
        """
        inf_log = _get_inf_log(self.config)
        pred_log = _get_pred_log(self.config)

        inf_log.info(f"Running diagnostics for case: {patient_id}...")

        # 1. Pipeline Validation Checks (Phases 1-3)
        is_valid, validation_results = self.run_pipeline_validation(image_path)
        if not is_valid:
            for stage_name, res in validation_results.items():
                if not res["passed"]:
                    inf_log.warning(f"Inference aborted: {stage_name} FAILED - {res['message']}")
                    raise ValueError(res["message"])

        # 2. Image preprocessing
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Lesion image not found at: {image_path}")
            
        with Image.open(image_path) as pil_img:
            raw_rgb = pil_img.convert("RGB")
            # Apply evaluation scaling & normalization
            img_tensor = self.image_transform(raw_rgb).unsqueeze(0).to(self.device)
            # Retain a numpy copy of the raw image for overlay plotting
            raw_np = np.array(raw_rgb.resize((self.config.data.image_size, self.config.data.image_size)))

        # 3. Tabular metadata preprocessing
        patient_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "localization": localization
        }])
        meta_tensor = self.preprocessor.transform(patient_df).to(self.device)

        # 4. Setup hooks and run Grad-CAM to extract heatmaps
        target_layer = self.model.cnn_extractor.features[-1]
        gradcam = GradCAM(model=self.model, target_layer=target_layer)

        try:
            # Generate heatmap
            heatmap, base_pred_idx, base_confidence = gradcam.generate_heatmap(
                images=img_tensor,
                metadata=meta_tensor,
                class_idx=None
            )

            # 5. Compute calibrated probability distribution (Phase 6)
            temp = self.config.inference.calibration_temperature
            with torch.no_grad():
                logits = self.model(img_tensor, meta_tensor)
                calibrated_logits = logits / temp
                probs = torch.softmax(calibrated_logits, dim=1).cpu().squeeze().numpy()
            
            predictions_map = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
            
            # Determine maximum class and calibrated confidence
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            # 6. Confidence Thresholding & Unknown class overrides (Phases 4-5)
            thresh = self.config.inference.confidence_threshold
            if confidence < thresh:
                pred_class = "unknown"
                pred_log.warning(
                    f"Low confidence prediction for case {patient_id}. "
                    f"Max calibrated probability {confidence*100:.2f}% < threshold {thresh*100:.2f}%. "
                    f"Prediction overridden to UNKNOWN."
                )
            else:
                pred_class = self.classes[pred_idx]
                pred_log.info(
                    f"Case {patient_id} result: {pred_class.upper()} | "
                    f"Confidence: {confidence*100:.2f}% (calibrated, temp={temp})"
                )

            # 7. Blend Grad-CAM activations onto the original image
            blended_np = GradCAM.overlay_heatmap(image_np=raw_np, heatmap=heatmap, alpha=0.45)
            blended_pil = Image.fromarray(blended_np)
            
            # 8. Generate clinical HTML summary report
            demographics = {
                "age": age,
                "sex": sex,
                "localization": localization
            }

            report_path = self.report_generator.generate_report(
                patient_id=patient_id,
                demographics=demographics,
                predictions=predictions_map,
                original_image=Image.fromarray(raw_np),
                heatmap_image=blended_pil,
                filename=report_filename,
                predicted_class=pred_class
            )

        finally:
            # Ensure hooks are detached even if an exception occurs
            gradcam.remove_hooks()

        return pred_class, predictions_map, report_path
