import argparse
import time
import sys
import os
import subprocess
import yaml

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
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training (default: 1)"
    )

    return parser.parse_args()


def check_uses_quantization(config_path):
    """Check if config uses quantization (QLoRA)."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model = config.get('model', {})
        return model.get('load_in_4bit', False) or model.get('load_in_8bit', False)
    except Exception:
        return False


def is_distributed_launch():
    """Check if already running under a distributed launcher."""
    return 'RANK' in os.environ or 'LOCAL_RANK' in os.environ or 'WORLD_SIZE' in os.environ


def launch_distributed_training(args):
    """Launch training with the appropriate distributed launcher."""
    script_path = os.path.abspath(__file__)
    
    # Build base command arguments  
    cmd_args = ["--config", args.config, "--num-gpus", str(args.num_gpus)]
    
    # Check if using quantization
    uses_quantization = check_uses_quantization(args.config)
    
    if uses_quantization:
        # QLoRA is incompatible with DeepSpeed, use torchrun
        print(f"Detected quantization (QLoRA) - using torchrun for {args.num_gpus} GPU(s)")
        cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
    else:
        # Auto-select deepspeed config
        deepspeed_config = os.path.join(project_root, f"configs/deepspeed/zero3_config_gpu{args.num_gpus}.json")
        if os.path.exists(deepspeed_config):
            print(f"Auto-selected deepspeed config: {deepspeed_config}")
            # Use deepspeed launcher
            print(f"Using deepspeed for {args.num_gpus} GPU(s)")
            cmd = ["deepspeed", f"--num_gpus={args.num_gpus}", "--deepspeed_config", deepspeed_config, script_path] + cmd_args
        else:
            print(f"Warning: DeepSpeed config for {args.num_gpus} GPU(s) not found, using torchrun")
            cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
    
    # Execute the command
    return subprocess.call(cmd)


def main():
    """Main training function."""
    args = parse_args()

    # If num_gpus > 1 and not already in distributed mode, launch distributed training
    if args.num_gpus > 1 and not is_distributed_launch():
        print(f"Launching distributed training with {args.num_gpus} GPUs...")
        return launch_distributed_training(args)
    
    # For single GPU or already in distributed mode, proceed with normal training
    if args.num_gpus == 1:
        print("Running single-GPU training...")
    else:
        print(f"Running as distributed process (rank: {os.environ.get('RANK', 'unknown')})...")

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    
    # Get deepspeed config from environment variable if set by deepspeed launcher
    if 'DEEPSPEED_CONFIG' in os.environ:
        config.training.deepspeed_config = os.environ['DEEPSPEED_CONFIG']
    else:
        config.training.deepspeed_config = None

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
