"""
Phase 8: Logging Module.
Defines separate logger initializers for validation, inference, and predictions.
"""

import os
import logging
from utils.config import Config

def get_custom_logger(name: str, log_filename: str, log_dir: str = "logs") -> logging.Logger:
    """
    Creates and configures a unique logger writing to a specific log file.

    Args:
        name (str): The name of the logger.
        log_filename (str): The filename (e.g. 'validation.log').
        log_dir (str): The directory to store log files.

    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicates in standard stdout

    # If handlers already exist, return the logger to prevent duplicate writes
    if logger.handlers:
        return logger

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # File Handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler for real-time visibility in developer logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_validation_logger(config: Config) -> logging.Logger:
    """Logger for Phases 1-3 validation checks."""
    return get_custom_logger("SkinCancerAI.Validation", "validation.log", config.paths.log_dir)


def get_inference_logger(config: Config) -> logging.Logger:
    """Logger for Phase 4-6 model execution and CUDA/tensor logistics."""
    return get_custom_logger("SkinCancerAI.Inference", "inference.log", config.paths.log_dir)


def get_prediction_logger(config: Config) -> logging.Logger:
    """Logger for Phase 4-5 confidence metrics and class prediction overrides."""
    return get_custom_logger("SkinCancerAI.Prediction", "prediction.log", config.paths.log_dir)
