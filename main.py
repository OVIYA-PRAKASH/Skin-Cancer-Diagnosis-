"""
Main orchestration entry point for the Hybrid Multi-Modal Skin Cancer Classification pipeline.
Supports three modes:
1. 'train' - Runs end-to-end dataset partitioning, preprocessor serialization, and model optimization.
2. 'evaluate' - Computes multi-class confusion matrices, ROC-AUC curves, and metrics on the test dataset.
3. 'explain' - Generates single-patient diagnostics, Grad-CAM overlays, and clinician HTML reports.
"""

import argparse
import sys
import os
import pickle

# Configure OpenMP duplicate library handling to prevent Windows abort crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd  # Import pandas before torch to resolve Windows OpenMP runtime collision
_ = pd.__name__
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

from utils.config import Config
from utils.helpers import set_seed, setup_logging, get_device
from dataset.loader import MetadataLoader, LesionLevelSplitter
from dataset.metadata import MetadataPreprocessor
from dataset.dataset import MultiModalDermoscopicDataset
from models.multimodal.multimodal_net import HybridMultiModalClassifier
from training.trainer import MultiModalTrainer
from evaluation.evaluator import MultiModalEvaluator
from utils.inference_engine import InferenceEngine


def setup_cli() -> argparse.ArgumentParser:
    """
    Configures command-line arguments.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Hybrid Multi-Modal Skin Cancer Classification Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to the configuration YAML file.",
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    # Train sub-command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )

    # Evaluate sub-command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model on test set")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint model file to evaluate.",
    )

    # Explain sub-command
    explain_parser = subparsers.add_parser(
        "explain", help="Generate Grad-CAM overlays and clinical summaries"
    )
    explain_parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to a single image file for inference.",
    )
    explain_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    explain_parser.add_argument(
        "--age", type=float, default=50.0, help="Patient age metadata."
    )
    explain_parser.add_argument(
        "--sex", type=str, default="male", help="Patient sex metadata."
    )
    explain_parser.add_argument(
        "--localization",
        type=str,
        default="back",
        help="Lesion anatomical site localization.",
    )

    return parser


def run_training(config: Config, device: torch.device, logger, resume: bool = False) -> None:
    """Executes the full training pipeline."""
    logger.info("Starting Dataset Pipeline for Training...")

    # 1. Load metadata and check physical disk files
    loader = MetadataLoader(
        metadata_path=config.paths.metadata_path,
        image_dirs=config.paths.image_dirs
    )
    df = loader.load_metadata()

    # Map categorical diagnoses targets to numerical labels
    classes = sorted(df["dx"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    logger.info(f"Target diagnostic classes mapping: {class_to_idx}")
    df["label"] = df["dx"].map(class_to_idx)

    # 2. Partition dataset at the lesion level to prevent data leakage
    train_r, val_r, test_r = config.data.train_val_test_split
    splitter = LesionLevelSplitter(
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=config.data.seed
    )
    train_df, val_df, test_df = splitter.split(df)

    # 3. Fit Metadata Preprocessor on training split demographics only
    preprocessor = MetadataPreprocessor()
    preprocessor.fit(train_df)
    
    # Save the fitted preprocessor for inference serialization
    preprocessor_path = os.path.join(config.paths.checkpoint_dir, "metadata_preprocessor.pkl")
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Serialized metadata preprocessor saved to: {preprocessor_path}")

    # Transform demographic splits
    train_meta_tensor = preprocessor.transform(train_df)
    val_meta_tensor = preprocessor.transform(val_df)

    # 4. Construct PyTorch Datasets
    train_dataset = MultiModalDermoscopicDataset(
        image_paths=train_df["image_path"].tolist(),
        metadata_tensor=train_meta_tensor,
        labels=train_df["label"].tolist(),
        image_size=config.data.image_size,
        is_training=True  # Apply training augmentations
    )
    
    val_dataset = MultiModalDermoscopicDataset(
        image_paths=val_df["image_path"].tolist(),
        metadata_tensor=val_meta_tensor,
        labels=val_df["label"].tolist(),
        image_size=config.data.image_size,
        is_training=False  # Apply scaling/normalizations only
    )

    # 5. Construct PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    # 6. Calculate balanced class weights to resolve class imbalance
    # Formula: w_c = N / (C * n_c)
    counts = train_df["label"].value_counts().sort_index()
    class_counts = counts.to_numpy()
    total_samples = len(train_df)
    weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Calculated balanced loss class weights: {class_weights}")

    # 7. Reconstruct model based on ablation study configuration
    mode = config.ablation.mode.lower().strip()
    image_mode = config.ablation.image_mode.lower().strip()

    use_metadata = (mode in ["multimodal", "metadata_only"])
    if mode == "metadata_only":
        use_cnn = False
        use_vit = False
    else:
        use_cnn = (image_mode in ["hybrid", "cnn_only"])
        use_vit = (image_mode in ["hybrid", "vit_only"])

    model = HybridMultiModalClassifier(
        num_classes=len(classes),
        cnn_backbone=config.model.cnn_backbone,
        vit_backbone=config.model.transformer_backbone,
        pretrained=config.model.pretrained,
        use_cnn=use_cnn,
        use_vit=use_vit,
        use_metadata=use_metadata,
        visual_proj_dim=512,
        meta_embed_dim=config.model.metadata_embed_dim,
        joint_dim=config.model.fusion_dim,
        dropout_prob=config.model.dropout
    ).to(device)

    # 8. Launch training optimizer loop
    trainer = MultiModalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        epochs=config.training.epochs,
        checkpoint_dir=config.paths.checkpoint_dir,
        use_amp=config.training.amp,
        max_grad_norm=1.0
    )

    if resume:
        trainer.load_checkpoint(os.path.join(config.paths.checkpoint_dir, "last_model.pth"))

    trainer.fit()


def run_evaluation(config: Config, checkpoint_path: str, device: torch.device, logger) -> None:
    """Runs the post-training evaluation suite on the test split."""
    logger.info(f"Starting test split evaluation on checkpoint: {checkpoint_path}")

    # 1. Load metadata and split partitions
    loader = MetadataLoader(
        metadata_path=config.paths.metadata_path,
        image_dirs=config.paths.image_dirs
    )
    df = loader.load_metadata()

    classes = sorted(df["dx"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    df["label"] = df["dx"].map(class_to_idx)

    train_r, val_r, test_r = config.data.train_val_test_split
    splitter = LesionLevelSplitter(
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=config.data.seed
    )
    _, _, test_df = splitter.split(df)

    # 2. Reload demographic preprocessor statistics
    preprocessor_path = os.path.join(config.paths.checkpoint_dir, "metadata_preprocessor.pkl")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Serialized metadata preprocessor file not found at: {preprocessor_path}")
        
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    # Transform test demographics
    test_meta_tensor = preprocessor.transform(test_df)

    # 3. Create PyTorch Dataset & DataLoader
    test_dataset = MultiModalDermoscopicDataset(
        image_paths=test_df["image_path"].tolist(),
        metadata_tensor=test_meta_tensor,
        labels=test_df["label"].tolist(),
        image_size=config.data.image_size,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    # 4. Reconstruct model
    mode = config.ablation.mode.lower().strip()
    image_mode = config.ablation.image_mode.lower().strip()

    use_metadata = (mode in ["multimodal", "metadata_only"])
    if mode == "metadata_only":
        use_cnn = False
        use_vit = False
    else:
        use_cnn = (image_mode in ["hybrid", "cnn_only"])
        use_vit = (image_mode in ["hybrid", "vit_only"])

    model = HybridMultiModalClassifier(
        num_classes=len(classes),
        cnn_backbone=config.model.cnn_backbone,
        vit_backbone=config.model.transformer_backbone,
        pretrained=False,
        use_cnn=use_cnn,
        use_vit=use_vit,
        use_metadata=use_metadata,
        visual_proj_dim=512,
        meta_embed_dim=config.model.metadata_embed_dim,
        joint_dim=config.model.fusion_dim,
        dropout_prob=config.model.dropout
    ).to(device)

    # Load weight state checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained checkpoint weights not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Weights successfully loaded.")

    # 5. Launch evaluation suite
    evaluator = MultiModalEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=classes,
        device=device,
        output_dir=config.paths.output_dir
    )

    evaluator.evaluate()


def run_explainability(config: Config, parsed_args, device: torch.device, logger) -> None:
    """Runs prediction and generates clinical summaries for a single case."""
    logger.info(f"Running inference explainability for image: {parsed_args.image_path}")

    preprocessor_path = os.path.join(config.paths.checkpoint_dir, "metadata_preprocessor.pkl")
    
    # Instantiate InferenceEngine
    engine = InferenceEngine(
        config_path=config.yaml_path,  # Load default configurations
        model_path=parsed_args.checkpoint,
        preprocessor_path=preprocessor_path,
        device=device
    )

    # Run diagnostics
    pred_class, probs, report_path = engine.predict_and_explain(
        image_path=parsed_args.image_path,
        age=parsed_args.age,
        sex=parsed_args.sex,
        localization=parsed_args.localization,
        patient_id="CASE_CLI_999",
        report_filename="cli_clinical_report.html"
    )

    logger.info(f"Diagnosis Result: {pred_class.upper()}")
    logger.info(f"Probability Map: {probs}")
    logger.info(f"Portable clinical report generated at: {report_path}")


def main(args: List[str]) -> None:
    """
    Main orchestration entry point.

    Args:
        args (List[str]): Command-line arguments.
    """
    parser = setup_cli()
    parsed_args = parser.parse_args(args)

    if parsed_args.mode is None:
        parser.print_help()
        return

    # Load configuration settings
    try:
        config = Config.load_from_yaml(parsed_args.config)
        # Store absolute yaml path for inference engine loaders
        config.yaml_path = parsed_args.config
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize directories (checkpoints, logs, outputs)
    config.paths.make_dirs()

    # Set up dual logging to console and disk
    logger = setup_logging(config.paths.log_dir, log_filename="run.log")
    logger.info("Initializing SkinCancerAI Pipeline...")
    logger.info(f"Configuration loaded successfully from: {parsed_args.config}")

    # Set random seeds for research reproducibility
    logger.info(f"Setting global random seed to: {config.data.seed}")
    set_seed(config.data.seed)

    # Hardware resource initialization
    device = get_device()
    logger.info(f"Active compute engine device: {device}")

    # Sub-command dispatcher
    if parsed_args.mode == "train":
        logger.info(f"Starting training process. Resume flag: {parsed_args.resume}")
        run_training(config, device, logger, parsed_args.resume)
    elif parsed_args.mode == "evaluate":
        logger.info(f"Evaluating model checkpoint: {parsed_args.checkpoint}")
        run_evaluation(config, parsed_args.checkpoint, device, logger)
    elif parsed_args.mode == "explain":
        logger.info(f"Generating explanations for {parsed_args.image_path}")
        run_explainability(config, parsed_args, device, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
