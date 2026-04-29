import argparse
import datetime
import time
import sys
import os
import subprocess
import yaml
import warnings
from typing import Dict, Optional

# Suppress Pydantic warnings from dependencies (TRL/transformers)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")
warnings.filterwarnings("ignore", message=".*'repr' attribute.*has no effect.*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*has no effect.*")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, WandbConfig, S3Config, DatasetInfo
from trainers.cpt_trainer import CPTTrainer
from trainers.dpo_trainer import DPOTrainer
from trainers.grpo_trainer import GRPOTrainer
from trainers.gspo_trainer import GSPOTrainer
from trainers.sft_trainer import SFTTrainer
# PPOTrainer is imported lazily — the legacy trl PPOConfig/PPOTrainer API was
# removed in trl >= 1.0, so the eager import would break every other algorithm.
# `trainers/__init__.py` provides a placeholder that errors only if PPO is used.
from trainers import PPOTrainer
from utils.logging_utils import TrainingLogger, log_system_info, rank_zero_print
from utils.recipe_overrides import apply_overrides_to_recipe, parse_value, set_nested_value, load_recipe_from_yaml
from utils.device_utils import get_device_type, supports_deepspeed, is_mps_available


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CPT, DPO, GRPO, GSPO, PPO, or SFT model",
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


def resolve_deepspeed_config(num_gpus: int) -> Optional[str]:
    """Resolve a DeepSpeed config path for the current world size.

    Lookup order (first hit wins):
      1. configs/deepspeed/zero3_config_gpu{N}.json  -- legacy per-GPU file, if
         someone hand-tuned one for a specific count.
      2. configs/deepspeed/zero3_config.json         -- GPU-count-agnostic
         default. Works for any world size because train_batch_size and
         gradient_accumulation_steps are 'auto' (DeepSpeed derives them).

    Returns the resolved path, or None if neither file exists.
    """
    candidates = [
        os.path.join(project_root, f"configs/deepspeed/zero3_config_gpu{num_gpus}.json"),
        os.path.join(project_root, "configs/deepspeed/zero3_config.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _peek_recipe(recipe_path: Optional[str], overrides: Optional[list]) -> Dict:
    """Cheap, side-effect-free read of recipe + CLI overrides into a plain dict.

    Quieter than the public `load_recipe_from_yaml` / `apply_overrides_to_recipe`
    pair (which print) so the launcher can call this multiple times without
    spamming logs.
    """
    recipe: Dict = {}
    if recipe_path:
        try:
            with open(recipe_path, 'r') as f:
                recipe = yaml.safe_load(f) or {}
        except Exception:
            recipe = {}
    if overrides:
        for override in overrides:
            if '=' not in override:
                continue
            key, value_str = override.split('=', 1)
            try:
                set_nested_value(recipe, key.strip(), parse_value(value_str))
            except Exception:
                pass
    return recipe


def resolve_parallelism(
    recipe_path: Optional[str], overrides: Optional[list]
) -> tuple:
    """Resolve (strategy_name, deepspeed_config_path_or_None) from recipe + overrides.

    Precedence:
      1. If `training.deepspeed_config` is set explicitly in the recipe/overrides,
         honor it as a custom DeepSpeed config (escape hatch for legacy zero3
         users or anyone with a hand-tuned config).
      2. Otherwise consult `training.parallelism_strategy`:
           - "auto"  -> "ddp" if model.use_lora else "zero1"
           - "ddp"   -> ("ddp", None)        # use torchrun, no DeepSpeed
           - "zero1" -> ("zero1", configs/deepspeed/zero1_config.json)

    ZeRO-3 is intentionally not auto-picked anymore: LoRA on MoE models
    deadlocks under ZeRO-3 because per-rank expert routing diverges and the
    _ALLGATHER_BASE collective never completes.
    """
    recipe = _peek_recipe(recipe_path, overrides)
    training = recipe.get("training", {}) or {}
    model = recipe.get("model", {}) or {}

    explicit_ds = training.get("deepspeed_config")
    if explicit_ds:
        ds_path = explicit_ds
        if not os.path.isabs(ds_path):
            ds_path = os.path.join(project_root, ds_path)
        return "custom", ds_path

    strategy = training.get("parallelism_strategy", "auto")
    use_lora = bool(model.get("use_lora", False))

    if strategy == "auto":
        strategy = "ddp" if use_lora else "zero1"

    if strategy == "ddp":
        return "ddp", None
    if strategy == "zero1":
        return "zero1", os.path.join(project_root, "configs/deepspeed/zero1_config.json")
    raise ValueError(
        f"Unknown training.parallelism_strategy={strategy!r}; "
        "supported values: auto, ddp, zero1"
    )


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
                # Drop any stale DEEPSPEED_CONFIG so children don't accidentally enable DeepSpeed.
                os.environ.pop('DEEPSPEED_CONFIG', None)
                cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
            else:
                strategy, ds_path = resolve_parallelism(args.recipe, args.overrides)
                if strategy == "ddp":
                    print(f"Resolved parallelism: ddp -> using torchrun for {args.num_gpus} GPU(s)")
                    os.environ.pop('DEEPSPEED_CONFIG', None)
                    cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
                elif not supports_deepspeed():
                    print(
                        f"Resolved parallelism: {strategy} but DeepSpeed not installed - "
                        f"falling back to torchrun (DDP) for {args.num_gpus} GPU(s)"
                    )
                    os.environ.pop('DEEPSPEED_CONFIG', None)
                    cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", script_path] + cmd_args
                else:
                    if not ds_path or not os.path.exists(ds_path):
                        raise FileNotFoundError(
                            f"DeepSpeed config for strategy={strategy!r} not found: {ds_path}"
                        )
                    print(
                        f"Resolved parallelism: {strategy} -> using deepspeed for "
                        f"{args.num_gpus} GPU(s) (config: {ds_path})"
                    )
                    os.environ['DEEPSPEED_CONFIG'] = ds_path
                    cmd = ["deepspeed", f"--num_gpus={args.num_gpus}", script_path] + cmd_args
    
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
            'wandb': {},
            's3': {}
        }
        rank_zero_print("No recipe file provided, using defaults with CLI overrides")
    
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
            "Provide it via recipe file or CLI: training.algorithm='sft' (options: cpt, sft, dpo, ppo, grpo, gspo)"
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
    
    # Create config objects with defaults
    return ExperimentConfig(
        model=ModelConfig(**recipe_dict.get('model', {})),
        data=DataConfig(**data_config),
        training=TrainingConfig(**recipe_dict.get('training', {})),
        wandb=WandbConfig(**recipe_dict.get('wandb', {})),
        s3=S3Config(**recipe_dict.get('s3', {})),
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
    else:
        # Already in distributed mode (e.g., Ray Train pre-set RANK/WORLD_SIZE
        # before launching this script). The parent launcher in
        # `launch_distributed_training` never ran here, so we have to resolve
        # the parallelism strategy ourselves and possibly set DEEPSPEED_CONFIG.
        if 'DEEPSPEED_CONFIG' not in os.environ:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            uses_quantization = check_uses_quantization(args.recipe) if args.recipe else False
            if world_size > 1 and not uses_quantization:
                try:
                    strategy, ds_path = resolve_parallelism(args.recipe, args.overrides)
                except Exception as e:
                    rank_zero_print(
                        f"Warning: failed to resolve parallelism strategy ({e}); "
                        "defaulting to DDP (no DeepSpeed)."
                    )
                    strategy, ds_path = "ddp", None

                if strategy == "ddp":
                    rank_zero_print(
                        f"Distributed environment using strategy=ddp "
                        f"(world_size={world_size}, no DeepSpeed)."
                    )
                elif not supports_deepspeed():
                    rank_zero_print(
                        f"Distributed environment requested strategy={strategy} but "
                        "DeepSpeed not installed - falling back to DDP."
                    )
                elif ds_path and os.path.exists(ds_path):
                    os.environ['DEEPSPEED_CONFIG'] = ds_path
                    rank_zero_print(
                        f"Auto-selected DeepSpeed config for distributed environment: "
                        f"{ds_path} (strategy={strategy})"
                    )
                else:
                    rank_zero_print(
                        f"Distributed environment requested strategy={strategy} but "
                        f"config not found at {ds_path} - falling back to DDP."
                    )

    # For single GPU or already in distributed mode, proceed with normal training.
    # NOTE: every torchrun/deepspeed worker re-parses argv without --num-gpus, so
    # `args.num_gpus` is the parser default (1) inside children even on an 8-GPU
    # job. Use is_distributed_launch() instead of args.num_gpus to decide what to
    # log here, otherwise every worker prints "Running single-GPU training...".
    if is_distributed_launch():
        rank_zero_print(
            f"Running as distributed process "
            f"(world_size={os.environ.get('WORLD_SIZE', '?')})..."
        )
    else:
        print("Running single-GPU training...")

    # Load recipe from file and/or CLI arguments
    config = load_recipe_with_overrides(args)
    
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
        "s3": config.s3.to_dict(),
    }
    
    training_logger.log_experiment_start(log_dict)

    start_time = time.time()

    try:
        # Create trainer based on algorithm and run training
        if algorithm == "cpt":
            trainer_class = CPTTrainer
        elif algorithm == "dpo":
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

