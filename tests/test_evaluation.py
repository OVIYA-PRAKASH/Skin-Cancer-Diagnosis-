"""
Verification script for Phase 12 & 13: Testing Pipeline and Metrics.
Simulates evaluating a multi-class skin lesion diagnostic classifier on a test dataset.
Generates metrics and writes output visualization plots to disk.
"""

import sys
import os

# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch
from torch.utils.data import DataLoader, TensorDataset

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed, get_device
from models.multimodal.multimodal_net import HybridMultiModalClassifier
from evaluation.evaluator import MultiModalEvaluator


def verify_evaluation_pipeline() -> None:
    """
    Runs evaluations using a mock classifier to test confusion matrix generation, 
    OvR ROC-AUC, and clinical metric output logs.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Evaluation Suite Verification...")

    # 2. Load settings and set seed
    config_path = "configs/default_config.yaml"
    config = Config.load_from_yaml(config_path)
    set_seed(config.data.seed)
    
    # 3. Detect and allocate device (GPU/CPU)
    device = get_device()
    batch_size = 4
    img_size = config.data.image_size

    # 4. Generate mock test data loader
    logger.info("Generating mock test dataset splits...")
    num_samples = 16
    
    mock_images = torch.randn(num_samples, 3, img_size, img_size)
    mock_metadata = torch.randn(num_samples, 19)
    # Generate labels containing at least a few different classes
    mock_labels = torch.randint(0, 7, (num_samples,))

    dataset = TensorDataset(mock_images, mock_metadata, mock_labels)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 5. Build Model wrapper
    logger.info("Building model wrapper...")
    model = HybridMultiModalClassifier(
        num_classes=7,
        pretrained=False,
        use_cnn=True,
        use_vit=False,       # Disable ViT for speed
        use_metadata=True
    ).to(device)

    # 6. Define 7 class names
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    # 7. Initialize Evaluator
    # Save test outputs in a test outputs folder to keep production clean
    test_output_dir = "tests/test_outputs"
    
    logger.info("Initializing MultiModalEvaluator...")
    evaluator = MultiModalEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=test_output_dir
    )

    # 8. Run evaluation
    logger.info("Executing evaluation cycle...")
    results = evaluator.evaluate()

    logger.info("Verifying metric results...")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Macro F1-Score: {results['macro_f1']:.4f}")
    logger.info(f"Macro ROC-AUC: {results['macro_auc']:.4f}")

    # Assertions
    assert "accuracy" in results, "Accuracy field is missing."
    assert "macro_f1" in results, "Macro F1 field is missing."
    assert "macro_auc" in results, "Macro AUC field is missing."
    assert "confusion_matrix" in results, "Confusion Matrix field is missing."

    # Verify that visual output PNG files are saved successfully on disk
    cm_plot_path = os.path.join(test_output_dir, "confusion_matrix.png")
    roc_plot_path = os.path.join(test_output_dir, "roc_curves.png")

    assert os.path.exists(cm_plot_path), "confusion_matrix.png was not generated."
    assert os.path.exists(roc_plot_path), "roc_curves.png was not generated."

    # Clean up test output directory
    os.remove(cm_plot_path)
    os.remove(roc_plot_path)
    os.rmdir(test_output_dir)
    logger.info("[OK] Visual Output Files check: PASSED.")

    logger.info("Evaluation Suite verification successful!")


if __name__ == "__main__":
    verify_evaluation_pipeline()
