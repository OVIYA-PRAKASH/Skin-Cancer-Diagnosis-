"""
Verification script for Phase 10 & 11: Training and Validation Pipeline.
Sets up a small dummy training batch sequence, runs a mock 2-epoch training cycle,
and verifies that loss functions, optimizers, and checkpoint saves execute successfully.
"""

import sys
import os

# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Append project root directory to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import Config
from utils.helpers import setup_logging, set_seed, get_device
from models.multimodal.multimodal_net import HybridMultiModalClassifier
from training.trainer import MultiModalTrainer


def verify_training_pipeline() -> None:
    """
    Simulates a training run to verify model optimization, scheduling, and validation checks.
    """
    # 1. Initialize logging
    logger = setup_logging(log_dir="logs", log_filename="test_run.log")
    logger.info("Initializing Training Engine Verification...")

    # 2. Load settings and set seed
    config_path = "configs/default_config.yaml"
    config = Config.load_from_yaml(config_path)
    set_seed(config.data.seed)
    
    # 3. Detect and allocate device (GPU if CUDA binds, else CPU)
    device = get_device()
    batch_size = 4
    img_size = config.data.image_size

    # 4. Generate small mock dataset loaders
    logger.info("Generating mock dataset splits...")
    num_samples = 12  # Small size for verification speed
    
    mock_images = torch.randn(num_samples, 3, img_size, img_size)
    mock_metadata = torch.randn(num_samples, 19)
    # 7 class indices [0-6]
    mock_labels = torch.randint(0, 7, (num_samples,))

    dataset = TensorDataset(mock_images, mock_metadata, mock_labels)
    
    # Create DataLoader wrappers
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Mock Train Loader count: {len(train_loader)} batches of size {batch_size}")
    logger.info(f"Mock Val Loader count: {len(val_loader)} batches of size {batch_size}")

    # 5. Instantiate Model
    logger.info("Building model wrapper...")
    model = HybridMultiModalClassifier(
        num_classes=7,
        pretrained=False,
        use_cnn=True,
        use_vit=False,       # Disable ViT for this verification test to save GPU VRAM/Time
        use_metadata=True
    ).to(device)

    # 6. Calculate class weights (mock example: equal weighting)
    class_weights = torch.ones(7, dtype=torch.float32)

    # 7. Initialize Trainer
    # We use a test checkpoints directory to keep production folders clean
    test_checkpoint_dir = "tests/test_checkpoints"
    
    logger.info("Initializing MultiModalTrainer...")
    trainer = MultiModalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-2,
        epochs=2,                # Run 2 epochs for quick check
        checkpoint_dir=test_checkpoint_dir,
        use_amp=True,
        max_grad_norm=1.0
    )

    # 8. Execute fit
    logger.info("Starting training engine fit cycle...")
    history = trainer.fit()

    logger.info("Verifying training outputs...")
    logger.info(f"Training Loss History: {history['train_loss']}")
    logger.info(f"Validation Loss History: {history['val_loss']}")
    logger.info(f"Validation Accuracy: {history['val_accuracy']}")
    logger.info(f"Validation Macro-F1: {history['val_f1']}")

    # Assertions
    assert len(history["train_loss"]) == 2, "Epoch loss count mismatch."
    assert len(history["val_loss"]) == 2, "Validation loss count mismatch."
    
    best_chkpt = os.path.join(test_checkpoint_dir, "best_model.pth")
    last_chkpt = os.path.join(test_checkpoint_dir, "last_model.pth")
    
    assert os.path.exists(best_chkpt), "best_model.pth was not saved."
    assert os.path.exists(last_chkpt), "last_model.pth was not saved."

    # Clean up test checkpoints to keep space tidy
    os.remove(best_chkpt)
    os.remove(last_chkpt)
    os.rmdir(test_checkpoint_dir)
    logger.info("[OK] Checkpoint Storage check: PASSED.")

    logger.info("Training Engine verification successful!")


if __name__ == "__main__":
    verify_training_pipeline()
