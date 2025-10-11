import argparse
import time
import sys
import os
import subprocess
import yaml
import ast
from typing import Any, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, WandbConfig, DatasetInfo
from trainers.dpo_trainer import DPOTrainer
from trainers.grpo_trainer import GRPOTrainer
from trainers.gspo_trainer import GSPOTrainer
from trainers.ppo_trainer import PPOTrainer
from trainers.sft_trainer import SFTTrainer
from utils.logging_utils import TrainingLogger, log_system_info


def parse_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    # Try to evaluate as Python literal (handles int, float, bool, list, dict, etc.)
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If it fails, return as string
        return value_str


def set_nested_value(config_dict: Dict, key_path: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot notation.
    
    Example: 
        set_nested_value(config, "model.base_model_name", "llama")
        sets config["model"]["base_model_name"] = "llama"
    """
    keys = key_path.split('.')
    current = config_dict
    
    # Navigate to the nested location
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DPO, GRPO, GSPO, PPO, or SFT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file:
  fai-rl-train --config configs/training/sft/llama3_3B_lora_recipe.yaml
  
  # Using command-line arguments only:
  fai-rl-train \\
    model.base_model_name="meta-llama/Llama-3.2-3B-Instruct" \\
    training.algorithm="sft" \\
    training.output_dir="./output" \\
    data.datasets="[{'name': 'nvidia/Aegis-AI-Content-Safety-Dataset-2.0', 'split': 'train'}]"
  
  # Mix config file with overrides:
  fai-rl-train --config config.yaml training.learning_rate=1e-5 training.num_train_epochs=3
  
Config parameters use dot notation for nested values:
  - model.base_model_name, model.torch_dtype, model.use_lora, model.lora_r, etc.
  - data.max_length, data.max_prompt_length, data.system_prompt, etc.
  - training.algorithm, training.learning_rate, training.num_train_epochs, etc.
  - wandb.enabled, wandb.project, wandb.name, etc.
"""
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (optional if using CLI arguments)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training (default: 1)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by launcher)"
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Path to DeepSpeed config file (automatically set by launcher)"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g., model.base_model_name='meta-llama/Llama-3.2-3B-Instruct')"
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
    
    # Build base command arguments (don't pass --num-gpus, launcher handles GPU allocation)
    cmd_args = []
    
    # Add config file if provided
    if args.config:
        cmd_args.extend(["--config", args.config])
    
    # Add overrides
    if args.overrides:
        cmd_args.extend(args.overrides)
    
    # Check if using quantization (only if config file is provided)
    uses_quantization = check_uses_quantization(args.config) if args.config else False
    
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
            cmd = ["deepspeed", f"--num_gpus={args.num_gpus}", script_path, "--deepspeed_config", deepspeed_config] + cmd_argsprint(cmd)
        else:
            print(f"Warning: DeepSpeed config for {args.num_gpus} GPU(s) not found, using torchrun")
            cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
    
    # Execute the command
    return subprocess.call(cmd)


def load_config_with_overrides(args) -> ExperimentConfig:
    """Load configuration from file and/or command-line arguments.
    
    Priority (highest to lowest):
    1. Command-line overrides
    2. Config file values
    3. Default values from dataclasses
    """
    # Start with an empty config dict
    config_dict = {}
    
    # Load from config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"Loaded base configuration from: {args.config}")
    else:
        # Initialize with empty sections
        config_dict = {
            'model': {},
            'data': {},
            'training': {},
            'wandb': {}
        }
        print("No config file provided, using defaults with CLI overrides")
    
    # Parse and apply command-line overrides
    if args.overrides:
        print("Applying command-line overrides:")
        for override in args.overrides:
            if '=' not in override:
                print(f"  Warning: Skipping invalid override '{override}' (expected key=value format)")
                continue
            
            key, value_str = override.split('=', 1)
            value = parse_value(value_str)
            set_nested_value(config_dict, key, value)
            print(f"  {key} = {value}")
    
    # Ensure required fields have at least some value
    if not config_dict.get('model', {}).get('base_model_name'):
        raise ValueError(
            "model.base_model_name is required. "
            "Provide it via config file or CLI: model.base_model_name='model-name'"
        )
    
    if not config_dict.get('training', {}).get('output_dir'):
        raise ValueError(
            "training.output_dir is required. "
            "Provide it via config file or CLI: training.output_dir='./output'"
        )
    
    if not config_dict.get('training', {}).get('algorithm'):
        raise ValueError(
            "training.algorithm is required. "
            "Provide it via config file or CLI: training.algorithm='sft' (options: sft, dpo, ppo, grpo, gspo)"
        )
    
    # Handle datasets configuration
    data_config = config_dict.get('data', {}).copy()
    if 'datasets' in data_config and data_config['datasets']:
        # Convert to DatasetInfo objects if they're dicts
        if isinstance(data_config['datasets'][0], dict):
            data_config['datasets'] = [
                DatasetInfo(**ds) for ds in data_config['datasets']
            ]
    else:
        # Default to empty list if no datasets specified
        data_config['datasets'] = []
    
    # Create config objects with defaults
    return ExperimentConfig(
        model=ModelConfig(**config_dict.get('model', {})),
        data=DataConfig(**data_config),
        training=TrainingConfig(**config_dict.get('training', {})),
        wandb=WandbConfig(**config_dict.get('wandb', {})),
    )


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

    # Load configuration from file and/or CLI arguments
    config = load_config_with_overrides(args)
    
    # Get deepspeed config from command-line arg, environment variable, or leave as None
    if args.deepspeed_config:
        config.training.deepspeed_config = args.deepspeed_config
    elif 'DEEPSPEED_CONFIG' in os.environ:
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
