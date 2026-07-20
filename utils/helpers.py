"""
Helper utilities for the SkinCancerAI project.
Includes random seed settings for reproducibility, logging configurations, and device helpers.
"""

import os
# Configure OpenMP to allow multiple initialized library instances (standard Windows workaround)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
# Dummy access to satisfy static analysis linters (e.g. PyLance) without config suppressions
_ = pd.__name__

import numpy as np
import random
import logging
import sys
import torch


def set_seed(seed: int) -> None:
    """
    Sets random seeds for Python, NumPy, PyTorch, and CUDA to ensure reproducibility.

    Args:
        seed (int): Selected seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU reproducibility
    
    # Enforce deterministic behavior in PyTorch algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, log_filename: str = "run.log") -> logging.Logger:
    """
    Configures a dedicated "SkinCancerAI" logger that outputs messages to both
    the stdout terminal and a log file on disk.

    Args:
        log_dir (str): Directory where the log file will be saved.
        log_filename (str): Name of the log file. Defaults to "run.log".

    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # Get our parent application logger
    app_logger = logging.getLogger("SkinCancerAI")
    
    # Clear existing handlers on the application logger to prevent duplicate logs if re-called
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)

    # Set logging level for the application logger
    app_logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler for terminal output (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)
    app_logger.addHandler(console_handler)

    # Handler for writing to file
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    app_logger.addHandler(file_handler)

    # Prevent logs from propagating up to the root logger to avoid duplication in streams
    app_logger.propagate = False

    app_logger.info(f"Logging initialized. Output mirrors to: {log_path}")
    return app_logger


def get_device() -> torch.device:
    """
    Identifies device context: returns CUDA device if available, else CPU.
    Logs hardware information.

    Returns:
        torch.device: Active device.
    """
    logger = logging.getLogger("DeviceInit")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA GPU detected: {device_name}")
        # Log VRAM details if possible
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Available GPU VRAM: {vram_gb:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA is not available. Falling back to CPU training.")
        
    return device
