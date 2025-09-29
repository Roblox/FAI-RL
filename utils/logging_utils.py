import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    name: str = "FAI-RL",
    level: int = logging.INFO,
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for the training process.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory to save log files
        console_output: Whether to output to console
        file_output: Whether to output to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = log_path / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_filename}")

    return logger


def log_gpu_memory():
    """Log GPU memory usage if CUDA is available."""
    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            logging.info(f"GPU {i} - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    else:
        logging.info("CUDA not available")


def log_system_info():
    """Log system information."""
    import torch
    import platform

    logger = logging.getLogger("system_info")

    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"CPU count: {os.cpu_count()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available")


class TrainingLogger:
    """Enhanced logger for training metrics and progress."""

    def __init__(self, name: str = "training", log_dir: str = "logs"):
        self.logger = setup_logging(name, log_dir=log_dir)
        self.step = 0

    def log_step(self, metrics: dict):
        """Log metrics for a training step."""
        self.step += 1
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {self.step} | {metric_str}")

    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for an epoch."""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} | {metric_str}")

    def log_checkpoint(self, checkpoint_path: str):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def log_experiment_start(self, config: dict):
        """Log experiment configuration at start."""
        self.logger.info("="*50)
        self.logger.info("EXPERIMENT START")
        self.logger.info("="*50)

        for section, values in config.items():
            self.logger.info(f"{section.upper()}:")
            for k, v in values.items():
                self.logger.info(f"  {k}: {v}")
            self.logger.info("")

    def log_experiment_end(self, duration: float):
        """Log experiment end."""
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60

        self.logger.info("="*50)
        self.logger.info("EXPERIMENT END")
        self.logger.info(f"Total duration: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        self.logger.info("="*50)

