import os, sys
import logging
from abc import ABC, abstractmethod
from typing import Optional
import wandb
import torch

from .config import ExperimentConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from utils.logging_utils import setup_logging


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__)
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.is_main_process = self.local_rank == -1 or self.local_rank == 0

        # Set CUDA device for distributed training to avoid NCCL warnings
        if self.local_rank != -1 and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.logger.info(f"Set CUDA device to GPU {self.local_rank} for this process")

        # Initialize wandb if enabled and on main process
        if self.is_main_process and self.config.wandb.enabled:
            self.setup_wandb()

        # Set memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
            config={
                **self.config.model.to_dict(),
                **self.config.data.to_dict(),
                **self.config.training.to_dict(),
            }
        )
        self.logger.info("Wandb initialized")

    def cleanup_wandb(self):
        """Clean up wandb session."""
        if self.is_main_process and self.config.wandb.enabled:
            wandb.finish()
            self.logger.info("Wandb session finished")

    @abstractmethod
    def setup_model(self):
        """Setup model and tokenizer. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def setup_data(self):
        """Setup training data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def setup_trainer(self):
        """Setup the specific trainer. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def train(self):
        """Run training. Must be implemented by subclasses."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with wandb cleanup."""
        self.cleanup_wandb()
