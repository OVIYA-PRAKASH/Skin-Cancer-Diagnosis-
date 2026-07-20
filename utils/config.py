"""
Configuration module for the SkinCancerAI project.
Defines strongly-typed dataclasses for project parameters and helper functions to load from YAML.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PathsConfig:
    """Configuration paths for datasets, checkpoints, logs, and outputs."""
    metadata_path: str
    image_dirs: List[str]
    checkpoint_dir: str
    log_dir: str
    output_dir: str

    def make_dirs(self) -> None:
        """Create output, log, and checkpoint directories if they do not exist."""
        for directory in [self.checkpoint_dir, self.log_dir, self.output_dir]:
            if directory:
                os.makedirs(directory, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing and data loader configurations."""
    image_size: int
    batch_size: int
    num_workers: int
    train_val_test_split: List[float]
    seed: int


@dataclass
class ModelConfig:
    """Model architecture backbones and dimensions."""
    cnn_backbone: str
    transformer_backbone: str
    pretrained: bool
    dropout: float
    metadata_embed_dim: int
    fusion_dim: int


@dataclass
class TrainingConfig:
    """Optimization loops and parameters."""
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    amp: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    gradient_accumulation_steps: int


@dataclass
class AblationConfig:
    """Ablation configurations for scientific research."""
    mode: str
    image_mode: str


@dataclass
class ValidationConfig:
    """Image validation, skin detection, and lesion detection thresholds."""
    min_width: int
    min_height: int
    max_width: int
    max_height: int
    blur_threshold: float
    solid_color_std_threshold: float
    skin_threshold: float
    lesion_threshold: float


@dataclass
class InferenceConfig:
    """Inference parameters including confidence threshold and probability calibration."""
    confidence_threshold: float
    calibration_temperature: float


@dataclass
class Config:
    """Main configuration class that wraps all sub-configs."""
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    ablation: AblationConfig
    validation: ValidationConfig
    inference: InferenceConfig

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "Config":
        """
        Loads configuration settings from a YAML file.

        Args:
            yaml_path (str): Path to the yaml config file.

        Returns:
            Config: Strongly-typed Config object.
        
        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")

        with open(yaml_path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        return cls(
            paths=PathsConfig(**raw_cfg["paths"]),
            data=DataConfig(**raw_cfg["data"]),
            model=ModelConfig(**raw_cfg["model"]),
            training=TrainingConfig(**raw_cfg["training"]),
            ablation=AblationConfig(**raw_cfg["ablation"]),
            validation=ValidationConfig(**raw_cfg["validation"]),
            inference=InferenceConfig(**raw_cfg["inference"]),
        )
