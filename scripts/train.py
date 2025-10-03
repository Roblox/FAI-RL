import argparse
import time
from pathlib import Path
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from trainers.dpo_trainer import DPOTrainer
from trainers.grpo_trainer import GRPOTrainer
from trainers.gspo_trainer import GSPOTrainer
from trainers.ppo_trainer import PPOTrainer
from trainers.sft_trainer import SFTTrainer
from utils.logging_utils import TrainingLogger, log_system_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DPO, GRPO, GSPO, or PPO model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        required=True,
        help="Path to deepspeed configuration json file"
    )
    parser.add_argument(
        "--local_rank",
        type=str,
        default=0,
        help="ignore the parameter"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    
    config.training.deepspeed_config = args.deepspeed_config

    # Get algorithm from config
    algorithm = config.training.algorithm.lower()

    # Setup logging with algorithm-specific prefix
    training_logger = TrainingLogger(f"{algorithm}_training")

    # Log system information
    log_system_info()
    
    # Log experiment configuration
    training_logger.log_experiment_start({
        "algorithm": {"name": algorithm},
        "model": config.model.to_dict(),
        "data": config.data.to_dict(),
        "training": config.training.to_dict(),
        "wandb": config.wandb.to_dict(),
    })

    start_time = time.time()

    try:
        # Create trainer based on algorithm and run training
        if algorithm == "dpo":
            trainer_class = DPOTrainer
        elif algorithm == "grpo":
            trainer_class = GRPOTrainer
        elif algorithm == "gspo":
            trainer_class = GSPOTrainer
        elif algorithm == "ppo":
            trainer_class = PPOTrainer
        elif algorithm == "sft":
            trainer_class = SFTTrainer
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        with trainer_class(config) as trainer:
            trainer.train()

        training_logger.logger.info(f"{algorithm.upper()} training completed successfully!")

    except Exception as e:
        training_logger.logger.error(f"Training failed with error: {str(e)}")
        raise

    finally:
        # Log experiment end
        end_time = time.time()
        duration = end_time - start_time
        training_logger.log_experiment_end(duration)


if __name__ == "__main__":
    main()
