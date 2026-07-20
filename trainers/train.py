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
from trainers.sft_vlm_trainer import SFTVLMTrainer
from utils.logging_utils import TrainingLogger, log_system_info, setup_logging
from utils.recipe_overrides import apply_overrides_to_recipe, parse_value, set_nested_value, load_recipe_from_yaml
from utils.device_utils import get_device_type, supports_deepspeed, is_mps_available

# Module-level logger for the launcher / pre-trainer status messages.
# setup_logging() attaches a RankFilter, so INFO/DEBUG records are dropped
# on non-rank-0 workers automatically. file_output=False because the parent
# training log already captures stdout (a second file would double-log
# under nohup).
logger = setup_logging("FAI-RL.launcher", file_output=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CPT, DPO, GRPO, GSPO, SFT, or SFT_VLM (multimodal) model",
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


def _peek_deepspeed_config(
    recipe_path: Optional[str], overrides: Optional[list]
) -> Optional[str]:
    """Read training.deepspeed_config from recipe + CLI overrides, returning
    an absolute path (or None if the field isn't set).

    Quieter than the public `load_recipe_from_yaml` / `apply_overrides_to_recipe`
    pair (which print and validate strictly) so the launcher can call this
    multiple times without spamming logs or crashing on partial recipes.

    Recipes for full fine-tuning typically set this to
    "configs/deepspeed/zero1_config.json"; LoRA recipes leave it unset and
    fall through to plain DDP.
    """
    recipe: Dict = {}
    if recipe_path:
        try:
            with open(recipe_path, 'r') as f:
                recipe = yaml.safe_load(f) or {}
        except Exception:
            return None
    if overrides:
        for override in overrides:
            if '=' not in override:
                continue
            key, value_str = override.split('=', 1)
            try:
                set_nested_value(recipe, key.strip(), parse_value(value_str))
            except Exception:
                pass

    ds_path = (recipe.get("training") or {}).get("deepspeed_config")
    if not ds_path:
        return None
    if not os.path.isabs(ds_path):
        ds_path = os.path.join(project_root, ds_path)
    return ds_path


# Auto-default ZeRO-3: large non-quantized multi-GPU LoRA/full runs that don't
# set training.deepspeed_config fall through to plain DDP, where *every* rank
# loads a full copy of the model into host RAM. For ~30B+ models that OOM-kills
# the pod during model load (silent SIGKILL, no traceback) regardless of GPU
# count -- see BaseTrainer._maybe_enable_deepspeed_zero3_init. ZeRO-3 partitions
# parameters at from_pretrained time so each rank materializes only 1/world_size
# of the model. We use a *no-offload* ZeRO-3 config (sharding alone fixes the
# host-RAM OOM; CPU offload is unnecessary on large GPUs and has caused VLM
# issues), and only above a size threshold so small models keep the faster DDP
# path. Override the threshold with FAI_RL_ZERO3_MIN_PARAMS_B; set it to a huge
# value to effectively disable the auto-default. An explicit
# `deepspeed_config: null` in the recipe is honored as "force DDP".
AUTO_ZERO3_CONFIG = "configs/deepspeed/zero3_auto_config.json"


def _zero3_min_params_billions() -> float:
    """Model size (in billions of params) at/above which ZeRO-3 is auto-selected."""
    try:
        return float(os.environ.get("FAI_RL_ZERO3_MIN_PARAMS_B", "20"))
    except (TypeError, ValueError):
        return 20.0


def _recipe_with_overrides(recipe_path: Optional[str], overrides: Optional[list]) -> Dict:
    """Load recipe YAML + apply CLI overrides, quietly (no logging/validation).

    Mirrors _peek_deepspeed_config's tolerant parsing so the launcher can inspect
    recipe fields repeatedly without spamming logs or crashing on partial recipes.
    """
    recipe: Dict = {}
    if recipe_path:
        try:
            with open(recipe_path, 'r') as f:
                recipe = yaml.safe_load(f) or {}
        except Exception:
            return {}
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


def _deepspeed_config_key_present(
    recipe_path: Optional[str], overrides: Optional[list]
) -> bool:
    """True if training.deepspeed_config is explicitly set in the recipe/overrides.

    Distinguishes "key absent" (auto-decide) from "key present but null" (an
    explicit request for plain DDP, which we must honor rather than override).
    """
    training = _recipe_with_overrides(recipe_path, overrides).get("training") or {}
    return "deepspeed_config" in training


def _peek_model_name(
    recipe_path: Optional[str], overrides: Optional[list]
) -> Optional[str]:
    """Read model.base_model_name from recipe + CLI overrides (or None)."""
    model = _recipe_with_overrides(recipe_path, overrides).get("model") or {}
    return model.get("base_model_name")


def _estimate_model_params_billions(model_name: Optional[str]) -> Optional[float]:
    """Best-effort estimate of a model's total parameter count, in billions.

    Loads only the (tiny) HF config -- not the weights -- and estimates from the
    architecture. Handles dense decoders, MoE (counts *all* experts, since host
    RAM holds every expert), and VLMs (unwraps text_config). Returns None when
    the config can't be resolved (e.g. s3:// path, offline, or unknown fields),
    in which case the caller falls back to the existing DDP behavior. The
    estimate only needs to be accurate enough to clear a coarse size threshold.
    """
    if not model_name or model_name.startswith("s3://"):
        return None
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:
        return None

    text_cfg = getattr(cfg, "text_config", None) or cfg

    def field(*names, default=0):
        for src in (text_cfg, cfg):
            for name in names:
                val = getattr(src, name, None)
                if val:
                    return val
        return default

    hidden = field("hidden_size", "n_embd", "d_model")
    layers = field("num_hidden_layers", "n_layer", "num_layers")
    vocab = field("vocab_size")
    if not hidden or not layers:
        return None
    intermediate = field("intermediate_size", "ffn_dim", default=4 * hidden)

    # Attention (q/k/v/o); GQA makes this an over-estimate, which is fine for a
    # coarse threshold.
    attn = 4 * hidden * hidden
    experts = field("num_experts", "num_local_experts", "n_routed_experts")
    if experts:
        moe_inter = field("moe_intermediate_size", default=intermediate)
        mlp = 3 * hidden * moe_inter * experts
    else:
        mlp = 3 * hidden * intermediate
    total = layers * (attn + mlp) + 2 * vocab * hidden  # + input embed & lm_head
    return total / 1e9


def _auto_zero3_config_if_large(
    recipe_path: Optional[str], overrides: Optional[list]
) -> Optional[str]:
    """Return the auto ZeRO-3 config path when a large model should use it, else None.

    Returns None (keep DDP) when deepspeed_config is explicitly set (incl. null),
    when DeepSpeed isn't available, when the model is below the size threshold,
    when its size can't be estimated, or when the shipped config file is missing.
    Caller is responsible for the multi-GPU / not-quantized gating.
    """
    if _deepspeed_config_key_present(recipe_path, overrides):
        return None
    if not supports_deepspeed():
        return None
    model_name = _peek_model_name(recipe_path, overrides)
    est_billions = _estimate_model_params_billions(model_name)
    if est_billions is None:
        logger.info(
            "Could not estimate size of model '%s'; keeping DDP. Set "
            "training.deepspeed_config explicitly to force a strategy.",
            model_name,
        )
        return None
    threshold = _zero3_min_params_billions()
    if est_billions < threshold:
        return None
    auto_path = os.path.join(project_root, AUTO_ZERO3_CONFIG)
    if not os.path.exists(auto_path):
        return None
    logger.info(
        "Auto-selecting ZeRO-3 (%s) for large model '%s' (~%.0fB params >= %.0fB "
        "threshold): plain DDP would replicate the full model into host RAM on "
        "every rank and OOM-kill the pod. Set deepspeed_config: null to force DDP.",
        AUTO_ZERO3_CONFIG, model_name, est_billions, threshold,
    )
    return auto_path


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
            logger.warning("Multi-GPU training is not supported on Apple Silicon (MPS); running single-device instead.")
            cmd = [sys.executable, script_path] + cmd_args
        else:
            # Check if using quantization (only if recipe file is provided)
            uses_quantization = check_uses_quantization(args.recipe) if args.recipe else False

            ds_path = None if uses_quantization else _peek_deepspeed_config(args.recipe, args.overrides)
            if not uses_quantization and not ds_path:
                # No explicit deepspeed_config: auto-select ZeRO-3 for large
                # models (see _auto_zero3_config_if_large) to avoid the
                # full-model-per-rank host-RAM OOM under plain DDP.
                ds_path = _auto_zero3_config_if_large(args.recipe, args.overrides)
            if ds_path and supports_deepspeed():
                if not os.path.exists(ds_path):
                    raise FileNotFoundError(f"training.deepspeed_config not found: {ds_path}")
                logger.info("Using deepspeed for %d GPU(s) (config: %s)", args.num_gpus, ds_path)
                os.environ['DEEPSPEED_CONFIG'] = ds_path
                cmd = ["deepspeed", f"--num_gpus={args.num_gpus}", script_path] + cmd_args
            else:
                if uses_quantization:
                    logger.info("Detected quantization (QLoRA) - using torchrun for %d GPU(s)", args.num_gpus)
                elif ds_path and not supports_deepspeed():
                    logger.warning(
                        "Recipe sets deepspeed_config=%s but DeepSpeed is not installed - "
                        "falling back to torchrun (DDP) for %d GPU(s)",
                        ds_path, args.num_gpus,
                    )
                else:
                    logger.info("Using torchrun (DDP) for %d GPU(s)", args.num_gpus)
                # Drop any stale DEEPSPEED_CONFIG so children don't accidentally enable DeepSpeed.
                os.environ.pop('DEEPSPEED_CONFIG', None)
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
        
        logger.info("Running in background with nohup. Output will be saved to: %s", log_file)
        
        # Prepare environment with log file path
        env = os.environ.copy()
        env['TRAINING_LOG_FILE'] = log_file
        
        # Prepare nohup command: nohup <command> > log_file 2>&1 &
        # We'll use shell=True to handle the redirection and background execution
        cmd_str = " ".join(cmd) + f" > {log_file} 2>&1"
        full_cmd = f"nohup {cmd_str} &"
        
        logger.info("Executing: %s", full_cmd)
        
        # Execute with Popen to start in background without waiting
        subprocess.Popen(full_cmd, shell=True, env=env)
        
        logger.info("Training started in background. Monitor progress with: tail -f %s", log_file)
        
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
        logger.info("No recipe file provided, using defaults with CLI overrides")
    
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
            "Provide it via recipe file or CLI: training.algorithm='sft' (options: cpt, sft, sft_vlm, dpo, grpo, gspo)"
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
                logger.info("Launching distributed training with %d GPUs...", args.num_gpus)
            else:
                logger.info("Launching single-GPU training with nohup...")
            return launch_distributed_training(args)
    else:
        # Already in distributed mode (e.g. Ray Train pre-set RANK/WORLD_SIZE
        # before launching this script). The parent launcher in
        # `launch_distributed_training` never ran here, so honor the recipe's
        # training.deepspeed_config ourselves if it's set and supported.
        if 'DEEPSPEED_CONFIG' not in os.environ:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            uses_quantization = check_uses_quantization(args.recipe) if args.recipe else False
            if world_size > 1 and not uses_quantization:
                ds_path = _peek_deepspeed_config(args.recipe, args.overrides)
                if not ds_path:
                    # No explicit deepspeed_config: auto-select ZeRO-3 for large
                    # models so DDP doesn't replicate the full model into host
                    # RAM on every rank and OOM-kill the pod.
                    ds_path = _auto_zero3_config_if_large(args.recipe, args.overrides)
                if ds_path and os.path.exists(ds_path) and supports_deepspeed():
                    os.environ['DEEPSPEED_CONFIG'] = ds_path
                    logger.info(
                        "Distributed environment using DeepSpeed config: %s (world_size=%d)",
                        ds_path, world_size,
                    )
                else:
                    logger.info(
                        "Distributed environment using DDP (world_size=%d, no deepspeed_config)",
                        world_size,
                    )

    # For single GPU or already in distributed mode, proceed with normal training.
    # NOTE: every torchrun/deepspeed worker re-parses argv without --num-gpus, so
    # `args.num_gpus` is the parser default (1) inside children even on an 8-GPU
    # job. Use is_distributed_launch() instead of args.num_gpus to decide what to
    # log here, otherwise every worker prints "Running single-GPU training...".
    if is_distributed_launch():
        logger.info(
            "Running as distributed process (world_size=%s)",
            os.environ.get('WORLD_SIZE', '?'),
        )
    else:
        logger.info("Running single-GPU training...")

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
        logger.info("Nohup mode detected. Logging to: %s", log_filename)
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
        elif algorithm == "sft":
            trainer_class = SFTTrainer
        elif algorithm == "sft_vlm":
            trainer_class = SFTVLMTrainer
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


    return None


if __name__ == "__main__":
    main()

