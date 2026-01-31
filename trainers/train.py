import argparse
import datetime
import logging
import time
import sys
import os
import subprocess
import yaml
import warnings
from typing import Dict

# Suppress Pydantic warnings from dependencies (TRL/transformers)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")
warnings.filterwarnings("ignore", message=".*'repr' attribute.*has no effect.*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*has no effect.*")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, WandbConfig, DatasetInfo, RewardAPIConfig
from trainers.dpo_trainer import DPOTrainer
from trainers.grpo_trainer import GRPOTrainer
from trainers.gspo_trainer import GSPOTrainer
from trainers.ppo_trainer import PPOTrainer
from trainers.sft_trainer import SFTTrainer
from utils.logging_utils import TrainingLogger, log_system_info
from utils.recipe_overrides import apply_overrides_to_recipe, parse_value, set_nested_value, load_recipe_from_yaml
from utils.device_utils import get_device_type, supports_deepspeed, is_mps_available


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DPO, GRPO, GSPO, PPO, or SFT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using recipe file:
  fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml
  
  # Mix recipe file with overrides:
  fai-rl-train --recipe recipe.yaml training.learning_rate=1e-5 training.num_train_epochs=3
"""
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Path to recipe YAML file (optional if using CLI arguments)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training (default: 1)"
    )
    parser.add_argument(
        "--nohup",
        action="store_true",
        help="Run training in background with nohup (output redirected to nohup.out)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Recipe overrides in key=value format (e.g., model.base_model_name='meta-llama/Llama-3.2-3B-Instruct')"
    )

    # Use parse_known_args to allow distributed launchers to pass additional args like --local_rank
    args, unknown = parser.parse_known_args()
    
    # Add this check: if no arguments provided at all, show help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    return args


def check_uses_quantization(recipe_path):
    """Check if recipe uses quantization (QLoRA)."""
    try:
        with open(recipe_path, 'r') as f:
            recipe = yaml.safe_load(f)
        model = recipe.get('model', {})
        return model.get('load_in_4bit', False) or model.get('load_in_8bit', False)
    except Exception:
        return False


def is_distributed_launch():
    """Check if already running under a distributed launcher."""
    return 'RANK' in os.environ or 'LOCAL_RANK' in os.environ or 'WORLD_SIZE' in os.environ


def get_algorithm_from_recipe(recipe_path, overrides):
    """Get algorithm name from recipe file and overrides."""
    try:
        # Load recipe dict
        recipe_dict = load_recipe_from_yaml(recipe_path) if recipe_path else {}
        
        # Apply overrides to get the final algorithm value
        if overrides:
            recipe_dict = apply_overrides_to_recipe(recipe_dict, overrides)
        
        # Get algorithm from training section
        algorithm = recipe_dict.get('training', {}).get('algorithm', 'training')
        return algorithm.lower()
    except Exception:
        return 'training'


def launch_distributed_training(args):
    """Launch training with the appropriate distributed launcher."""
    script_path = os.path.abspath(__file__)
    device_type = get_device_type()
    
    # Build base command arguments (don't pass --num-gpus and --nohup, launcher handles GPU allocation)
    cmd_args = []
    
    # Add recipe file if provided
    if args.recipe:
        cmd_args.extend(["--recipe", args.recipe])
    
    # Add debug flag if provided
    if args.debug:
        cmd_args.append("--debug")
    
    # Add overrides
    if args.overrides:
        cmd_args.extend(args.overrides)
    
    # For single GPU/device with nohup, just use python directly (no launcher needed)
    if args.num_gpus == 1:
        cmd = [sys.executable, script_path] + cmd_args
    else:
        # Multi-GPU training - check platform support
        if is_mps_available():
            print("Warning: Multi-GPU training is not supported on Apple Silicon (MPS).")
            print("Running single-device training instead.")
            cmd = [sys.executable, script_path] + cmd_args
        else:
            # Check if using quantization (only if recipe file is provided)
            uses_quantization = check_uses_quantization(args.recipe) if args.recipe else False
            
            if uses_quantization:
                # QLoRA is incompatible with DeepSpeed, use torchrun
                print(f"Detected quantization (QLoRA) - using torchrun for {args.num_gpus} GPU(s)")
                cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
            elif supports_deepspeed():
                # Auto-select deepspeed config
                deepspeed_config = os.path.join(project_root, f"configs/deepspeed/zero3_config_gpu{args.num_gpus}.json")
                if os.path.exists(deepspeed_config):
                    print(f"Auto-selected deepspeed config: {deepspeed_config}")
                    # Set environment variable for deepspeed config
                    os.environ['DEEPSPEED_CONFIG'] = deepspeed_config
                    # Use deepspeed launcher
                    print(f"Using deepspeed for {args.num_gpus} GPU(s)")
                    cmd = ["deepspeed", f"--num_gpus={args.num_gpus}", script_path] + cmd_args
                else:
                    print(f"Warning: DeepSpeed config for {args.num_gpus} GPU(s) not found, using torchrun")
                    cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
            else:
                print("DeepSpeed not available, using torchrun for multi-GPU training")
                cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
    
    # Handle nohup mode
    if args.nohup:
        # Get algorithm name from recipe to create consistent log filename
        algorithm = get_algorithm_from_recipe(args.recipe, args.overrides)
        
        # Generate log filename with timestamp (matching TrainingLogger format)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/{algorithm}_training_{timestamp}.log"
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        print(f"Running in background with nohup. Output will be saved to: {log_file}")
        
        # Prepare environment with log file path
        env = os.environ.copy()
        env['TRAINING_LOG_FILE'] = log_file
        
        # Prepare nohup command: nohup <command> > log_file 2>&1 &
        # We'll use shell=True to handle the redirection and background execution
        cmd_str = " ".join(cmd) + f" > {log_file} 2>&1"
        full_cmd = f"nohup {cmd_str} &"
        
        print(f"Executing: {full_cmd}")
        
        # Execute with Popen to start in background without waiting
        subprocess.Popen(full_cmd, shell=True, env=env)
        
        print(f"Training started in background. Monitor progress with: tail -f {log_file}")
        
        return 0
    else:
        # Execute the command normally (foreground)
        return subprocess.call(cmd)


def load_recipe_with_overrides(args) -> ExperimentConfig:
    """Load recipe from file and/or command-line arguments.
    
    Priority (highest to lowest):
    1. Command-line overrides
    2. Recipe file values
    3. Default values from dataclasses
    """
    # Start with an empty recipe dict
    recipe_dict = {}
    
    # Load from recipe file if provided
    if args.recipe:
        recipe_dict = load_recipe_from_yaml(args.recipe)
    else:
        # Initialize with empty sections
        recipe_dict = {
            'model': {},
            'data': {},
            'training': {},
            'wandb': {}
        }
        print("No recipe file provided, using defaults with CLI overrides")
    
    # Apply command-line overrides using common utility
    if args.overrides:
        recipe_dict = apply_overrides_to_recipe(recipe_dict, args.overrides)
    
    # Ensure required fields have at least some value
    if not recipe_dict.get('model', {}).get('base_model_name'):
        raise ValueError(
            "model.base_model_name is required. "
            "Provide it via recipe file or CLI: model.base_model_name='model-name'"
        )
    
    if not recipe_dict.get('training', {}).get('output_dir'):
        raise ValueError(
            "training.output_dir is required. "
            "Provide it via recipe file or CLI: training.output_dir='./output'"
        )
    
    if not recipe_dict.get('training', {}).get('algorithm'):
        raise ValueError(
            "training.algorithm is required. "
            "Provide it via recipe file or CLI: training.algorithm='sft' (options: sft, dpo, ppo, grpo, gspo)"
        )
    
    # Handle datasets configuration
    data_config = recipe_dict.get('data', {}).copy()
    if 'datasets' in data_config and data_config['datasets']:
        # Convert to DatasetInfo objects if they're dicts
        if isinstance(data_config['datasets'][0], dict):
            data_config['datasets'] = [
                DatasetInfo(**ds) for ds in data_config['datasets']
            ]
    else:
        # Default to empty list if no datasets specified
        data_config['datasets'] = []
    
    # Handle reward_api configuration
    reward_api = None
    if 'reward_api' in recipe_dict and recipe_dict['reward_api']:
        reward_api = RewardAPIConfig(**recipe_dict['reward_api'])
    
    # Create config objects with defaults
    return ExperimentConfig(
        model=ModelConfig(**recipe_dict.get('model', {})),
        data=DataConfig(**data_config),
        training=TrainingConfig(**recipe_dict.get('training', {})),
        wandb=WandbConfig(**recipe_dict.get('wandb', {})),
        reward_api=reward_api,
    )


def main():
    """Main training function."""
    args = parse_args()

    # Handle nohup or multi-GPU launch (if not already in distributed mode)
    if not is_distributed_launch():
        # If nohup is requested OR multi-GPU training, use launcher
        if args.nohup or args.num_gpus > 1:
            if args.num_gpus > 1:
                print(f"Launching distributed training with {args.num_gpus} GPUs...")
            else:
                print("Launching single-GPU training with nohup...")
            return launch_distributed_training(args)
    
    # For single GPU or already in distributed mode, proceed with normal training
    if is_distributed_launch():
        rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', 'unknown'))
        world_size = os.environ.get('WORLD_SIZE', 'unknown')
        print(f"Running as distributed process (rank {rank}/{world_size})...")
    else:
        print("Running single-GPU training...")

    # Load recipe from file and/or CLI arguments
    config = load_recipe_with_overrides(args)
    
    # Enable debug logging if --debug flag is passed
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also set debug level for specific modules
        logging.getLogger('utils.api_utils').setLevel(logging.DEBUG)
        logging.getLogger('trainers.rewards.subjective_rewards').setLevel(logging.DEBUG)
        print("Debug mode enabled - verbose logging activated")
        config.debug = True
        config.training.debug = True
    else:
        # Explicitly keep non-debug logs at INFO level
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('utils.api_utils').setLevel(logging.INFO)
        logging.getLogger('trainers.rewards.subjective_rewards').setLevel(logging.INFO)
    
    # Get deepspeed config from environment variable (auto-set by launcher)
    if 'DEEPSPEED_CONFIG' in os.environ:
        config.training.deepspeed_config = os.environ['DEEPSPEED_CONFIG']
    else:
        config.training.deepspeed_config = None

    # Get algorithm from config
    algorithm = config.training.algorithm.lower()

    # Setup logging with algorithm-specific prefix
    # When running with nohup, stdout is already redirected to a file,
    # so we don't need a separate file handler (it would cause duplicates)
    log_filename = os.environ.get('TRAINING_LOG_FILE', None)
    if log_filename:
        # Running with nohup: use console output only (nohup handles file redirection)
        training_logger = TrainingLogger(f"{algorithm}_training", file_output=False)
        print(f"Nohup mode detected. Logging to: {log_filename}")
    else:
        # Running normally: use both console and file output
        training_logger = TrainingLogger(f"{algorithm}_training")

    # Log system information
    log_system_info()
    
    # Log experiment configuration
    log_dict = {
        "algorithm": {"name": algorithm},
        "model": config.model.to_dict(),
        "data": config.data.to_dict(),
        "training": config.training.to_dict(),
        "wandb": config.wandb.to_dict(),
    }
    
    training_logger.log_experiment_start(log_dict)

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
            
        # Pass the training_logger to the trainer to consolidate logging
        with trainer_class(config, logger=training_logger.logger.logger) as trainer:
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

