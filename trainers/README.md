# FAI-RL Training

Training implementations supporting DPO (Direct Preference Optimization), GRPO (Group Relative Policy Optimization), and GSPO (Group Sequence Policy Optimization) methods.

## 🚀 Quick Start

**Run training**:
```bash
# Single-GPU Training
./scripts/run_training.sh --config configs/training/dpo/llama3_3B_recipe.yaml --num-gpus 1

# Multi-GPU Training (8 GPUs)
./scripts/run_training.sh --config configs/training/dpo/llama3_3B_recipe.yaml --num-gpus 8

# Background execution
./scripts/run_training.sh --config configs/training/dpo/llama3_3B_recipe.yaml --num-gpus 4 --nohup
```

**Script Usage:**
```bash
Usage: ./scripts/run_training.sh --config CONFIG_FILE [OPTIONS]

Options:
  --config CONFIG_FILE                   Path to configuration YAML file (required)
  --num-gpus NUM_GPUS                    Number of GPUs to use (required)
  --nohup                                Run in background with nohup
  -h, --help                             Show help message
```

## 🔧 Configuration

Create training configs in `../configs/training/`:

```yaml
# Model Configuration
# Defines the base model and its loading parameters
model:
  base_model_name: "meta-llama/Llama-3.2-3B-Instruct"   # HuggingFace model name or local path
  torch_dtype: "bfloat16"                               # Data type for model weights (bfloat16/float16/float32)
  low_cpu_mem_usage: true                               # Reduce CPU memory usage during model loading
  load_in_8bit: false                                   # Enable 8-bit quantization
  load_in_4bit: false                                   # Enable 4-bit quantization
  use_flash_attention: false                            # Use Flash Attention for faster training (if supported)

# Data Configuration
# Specifies datasets and preprocessing settings
data:
  datasets:
    # Supports multiple datasets. 
    # Add additional entries here for combined training across datasets.
    - name: "your-dataset"              # HuggingFace dataset name/path (required) e.g. Anthropic/hh-rlhf
      subset: "your-subset"             # Dataset subset/config name (optional)
      split: "train"                    # Dataset split to use (default: "train")
      prompt_column: "prompt"           # Name of prompt column (default: "prompt")
      chosen_column: "chosen"           # Name of chosen/preferred response column (default: "chosen")
      rejected_column: "rejected"       # Name of rejected response column (default: "rejected")
    - name: "your-dataset2"             # Optional: add multiple datasets for combined training
      split: "train"
      chosen_column: "chosen"
      rejected_column: "rejected"
  
  # Text processing settings
  max_length: 2048                      # Maximum sequence length for model input
  max_prompt_length: 1024               # Maximum length for prompts (rest reserved for responses)
  remove_unused_columns: false          # Keep all dataset columns (set true to save memory)

# Training Configuration  
# Controls the training process and optimization settings
training:
  algorithm: "dpo"                      # Training algorithm: dpo, grpo, gspo
  output_dir: "models/output"           # Directory to save trained model and checkpoints
  run_name: "my-experiment"             # Unique identifier for this training run
  
  # Core training hyperparameters
  per_device_train_batch_size: 1        # Batch size per GPU (adjust based on GPU memory)
  gradient_accumulation_steps: 16       # Steps to accumulate gradients (effective batch = batch_size × accum_steps × num_gpus)
  learning_rate: 1.0e-6                 # Learning rate (typical range: 1e-7 to 1e-5 for LLMs)
  num_train_epochs: 3                   # Number of complete passes through the dataset
  warmup_steps: 50                      # Linear warmup steps for learning rate scheduler
  
  # GSPO-specific parameters (only used when algorithm: "gspo")
  beta: 0                               # KL regularization strength (0 = no KL penalty)
  group_size: 4                         # Group size for sequence grouping
  epsilon: 3e-4                         # Policy exploration parameter (lower bound)
  epsilon_high: 4e-4                    # Policy exploration parameter (upper bound)  
  steps_per_generation: 4               # Minibatch partitioning for rollout data
  
  # Logging and checkpointing
  logging_steps: 5                      # Log training metrics every N steps
  save_steps: 100                       # Save model checkpoint every N steps
  
  # Memory and precision optimization
  bf16: true                            # Use bfloat16 precision (recommended for modern GPUs)
  fp16: false                           # Use float16 precision (alternative to bf16)
  gradient_checkpointing: true          # Trade compute for memory (enables larger models/batch sizes)

  # Data loading optimization
  dataloader_num_workers: 0             # Number of CPU workers for data loading (0 = main process only)
  dataloader_pin_memory: false          # Pin memory for faster GPU transfer (set true if sufficient RAM)
  dataloader_drop_last: true            # Drop last incomplete batch to ensure consistent batch sizes
  
  # Output and evaluation settings
  save_only_model: true                 # Save only model weights (not optimizer states) to reduce disk usage
  prediction_loss_only: true            # Only compute prediction loss during evaluation

# Weights & Biases Integration
# Optional experiment tracking and monitoring
wandb:  
  enabled: true                         # Enable W&B logging
  project: "your-project"               # W&B project name
  entity: "your-entity"                 # W&B username or team name
  name: "your-wandb-name"               # Experiment name in W&B (defaults to run_name if not specified)
  tags: ["your-tags"]                   # Tags for organizing experiments
```

### Configuration Parameters

**Configuration Checklist:**
Replace the following values for your specific use case:
- `data.datasets.name` → your HuggingFace dataset(s) (e.g., "Anthropic/hh-rlhf")
- `training.output_dir` → your desired model output directory  
- `training.run_name` → your unique run identifier
- `wandb.*` → your Weights & Biases configuration (or set `enabled: false` to disable)

**Memory Optimization Tips:**
- Reduce `per_device_train_batch_size` if you encounter OOM errors
- Enable `gradient_checkpointing` for larger models
- Use `load_in_8bit` or `load_in_4bit` for memory-constrained setups
- Set `dataloader_pin_memory: true` only if you have sufficient system RAM

## 📊 Training Progress

**Monitoring options:**
- Logs are stored in `../logs/`
- If Weights & Biases is enabled, follow real-time progress at wandb
- Final models are saved under `./models/`