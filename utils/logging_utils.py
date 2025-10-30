import logging
import os
import sys
from datetime import datetime
from pathlib import Path


class RobustFileHandler(logging.FileHandler):
    """
    A file handler that gracefully handles stale file handle errors.
    
    This is particularly useful in distributed/networked file systems (NFS)
    where file handles can become stale during long-running training jobs.
    """
    
    def emit(self, record):
        """
        Emit a record, handling stale file handle errors gracefully.
        
        If a stale file handle error occurs (OSError errno 116), we:
        1. Try to reopen the file
        2. If that fails, suppress the error to prevent training crashes
        """
        try:
            super().emit(record)
        except OSError as e:
            # Errno 116 is "Stale file handle"
            # Errno 5 is "Input/output error" (also common with NFS)
            if e.errno in (5, 116):
                try:
                    # Try to reopen the file
                    self.close()
                    self.stream = self._open()
                    super().emit(record)
                except Exception:
                    # If reopening fails, silently continue to prevent training crash
                    # The log message is lost, but training continues
                    pass
            else:
                # Re-raise other OSErrors
                raise
    
    def flush(self):
        """
        Flush the stream, handling stale file handle errors gracefully.
        """
        try:
            super().flush()
        except OSError as e:
            # Suppress stale file handle errors during flush
            if e.errno in (5, 116):
                pass
            else:
                raise


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

    # File handler with robust error handling
    if file_output:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = log_path / f"{name}_{timestamp}.log"

        # Use RobustFileHandler instead of regular FileHandler
        file_handler = RobustFileHandler(log_filename)
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


class SafeLogger:
    """
    A wrapper around logging.Logger that catches and handles logging exceptions.
    
    This prevents logging errors from crashing the training process.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _safe_log(self, level, msg, *args, **kwargs):
        """Safely log a message, catching any exceptions."""
        try:
            getattr(self.logger, level)(msg, *args, **kwargs)
        except Exception:
            # Silently continue if logging fails
            # This prevents training from crashing due to logging issues
            pass
    
    def info(self, msg, *args, **kwargs):
        self._safe_log('info', msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self._safe_log('debug', msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._safe_log('warning', msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._safe_log('error', msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._safe_log('critical', msg, *args, **kwargs)
    
    def __getattr__(self, name):
        """Forward any other attributes to the underlying logger."""
        return getattr(self.logger, name)


class TrainingLogger:
    """Enhanced logger for training metrics and progress."""

    def __init__(self, name: str = "training", log_dir: str = "logs"):
        base_logger = setup_logging(name, log_dir=log_dir)
        self.logger = SafeLogger(base_logger)
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

