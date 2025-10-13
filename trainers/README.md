# FAI-RL Training

Training implementations supporting SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), PPO (Proximal Policy Optimization), GRPO (Group Relative Policy Optimization), and GSPO (Group Sequence Policy Optimization) methods.

## 🚀 Quick Start

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
  load_in_4bit: false                                   # Enable 4-bit quantization (for QLoRA)
  use_flash_attention: false                            # Use Flash Attention for faster training (if supported)
  
  # PPO-specific model parameters (only used when algorithm: "ppo")
  value_model_name: "meta-llama/Llama-3.2-3B-Instruct" # Model for value function (typically same as base_model)
  
  # QLoRA-specific quantization parameters (only when load_in_4bit: true)
  # bnb_4bit_compute_dtype: "bfloat16"                  # Compute dtype for 4-bit quantization
  # bnb_4bit_quant_type: "nf4"                          # Quantization type: "nf4" or "fp4"
  # bnb_4bit_use_double_quant: true                     # Use double quantization for better accuracy
  
  # LoRA configuration (for parameter-efficient fine-tuning)
  # use_lora: true                                      # Enable LoRA
  # lora_r: 8                                           # LoRA rank (8 for LoRA, 16 for QLoRA)
  # lora_alpha: 16                                      # LoRA alpha scaling factor (typically 2x rank)
  # lora_dropout: 0.05                                  # LoRA dropout rate
  # lora_target_modules:                                # Target modules for LoRA
  #   - q_proj
  #   - v_proj
  #   - k_proj
  #   - o_proj
  #   - gate_proj
  #   - up_proj
  #   - down_proj
  # lora_bias: "none"                                   # Bias handling: "none", "all", or "lora_only"

# Data Configuration
# Specifies datasets and preprocessing settings
data:
  datasets:
    # Supports multiple datasets. 
    # Add additional entries here for combined training across datasets.
    - name: "your-dataset"              # HuggingFace dataset name/path (required) e.g. Anthropic/hh-rlhf
      subset: "your-subset"             # Dataset subset/config name (optional)
      split: "train"                    # Dataset split to use (default: "train")
      
      # For DPO/PPO (preference-based methods):
      prompt_column: "prompt"           # Name of prompt column (default: "prompt")
      chosen_column: "chosen"           # Name of chosen/preferred response column (default: "chosen")
      rejected_column: "rejected"       # Name of rejected response column (default: "rejected")
      
      # For SFT (supervised fine-tuning):
      # prompt_column: "question"       # Name of instruction/prompt column
      # answer_column: "answer"         # Name of answer/completion column (default: "answer")
      
      # For GRPO/GSPO (math/reasoning tasks):
      # prompt_column: "question"       # Name of question/prompt column
      # answer_column: "answer"         # Name of answer column
      
      # Optional: Custom dataset columns for special processing
      # dataset_columns: ["prompt", "response", "label"]  # Specify exact columns to keep
      
    - name: "your-dataset2"             # Optional: add multiple datasets for combined training
      split: "train"
      chosen_column: "chosen"
      rejected_column: "rejected"
  
  # Text processing settings
  max_length: 2048                      # Maximum sequence length for model input
  max_prompt_length: 1024               # Maximum length for prompts (rest reserved for responses)
  remove_unused_columns: false          # Keep all dataset columns (set true to save memory)
  dataset_num_proc: 1                   # Number of processes for dataset preprocessing (optional)
  
  # Optional: System prompt for special formatting (SFT with custom templates)
  # system_prompt: |
  #   Your custom system prompt here with placeholders like {prompt} and {response}

# Training Configuration  
# Controls the training process and optimization settings
training:
  algorithm: "dpo"                      # Training algorithm: sft, dpo, ppo, grpo, gspo
  output_dir: "models/output"           # Directory to save trained model and checkpoints
  
  # Core training hyperparameters
  per_device_train_batch_size: 1        # Batch size per GPU (adjust based on GPU memory)
  gradient_accumulation_steps: 16       # Steps to accumulate gradients (effective batch = batch_size × accum_steps × num_gpus)
  learning_rate: 1.0e-5                 # Learning rate (1e-5 for full/LoRA, 1e-4 to 2e-4 for QLoRA)
  num_train_epochs: 1                   # Number of complete passes through the dataset
  max_steps: -1                         # Maximum number of training steps (-1 = train for num_train_epochs)
  warmup_steps: 50                      # Linear warmup steps for learning rate scheduler
  
  # PPO-specific parameters (only used when algorithm: "ppo")
  # gamma: 1.0                          # Discount factor for rewards
  # lam: 0.95                           # GAE (Generalized Advantage Estimation) lambda
  # cliprange: 0.2                      # PPO clipping range for policy updates
  # cliprange_value: 0.2                # Value function clipping range
  # vf_coef: 0.1                        # Value function loss coefficient
  
  # GSPO-specific parameters (only used when algorithm: "gspo")
  # beta: 0                             # KL regularization strength (0 = no KL penalty)
  # group_size: 4                       # Group size for sequence grouping
  # epsilon: 3e-4                       # Policy exploration parameter (lower bound)
  # epsilon_high: 4e-4                  # Policy exploration parameter (upper bound)  
  # steps_per_generation: 4             # Minibatch partitioning for rollout data
  
  # Logging and checkpointing
  logging_steps: 5                      # Log training metrics every N steps
  save_steps: 100                       # Save model checkpoint every N steps
  eval_steps: 100                       # Evaluate model every N steps (optional, used by DPO/GRPO/GSPO/SFT)
  
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
  name: "your-wandb-name"               # Experiment name in W&B
  tags: ["your-tags"]                   # Tags for organizing experiments
```

### Configuration Parameters

**Configuration Checklist:**
Replace the following values for your specific use case:
- `data.datasets.name` → your HuggingFace dataset(s) (e.g., "Anthropic/hh-rlhf" for DPO/PPO, "openai/gsm8k" for GRPO/GSPO, "nvidia/Aegis-AI-Content-Safety-Dataset-2.0" for SFT)
- `data.datasets.prompt_column` / `answer_column` / `chosen_column` / `rejected_column` → adjust based on your dataset and algorithm
  - **SFT**: Use `prompt_column` and `answer_column`
  - **DPO/PPO**: Use `prompt_column`, `chosen_column`, and `rejected_column`
  - **GRPO/GSPO**: Use `prompt_column` and `answer_column`
- `training.algorithm` → choose from: `sft`, `dpo`, `ppo`, `grpo`, `gspo`
- `training.output_dir` → your desired model output directory  
- `wandb.*` → your Weights & Biases configuration (or set `enabled: false` to disable)

**Algorithm-Specific Notes:**
- **SFT**: Best for initial instruction tuning; requires `prompt_column` and `answer_column` in dataset
- **DPO**: Preference-based method; requires `prompt_column`, `chosen_column`, and `rejected_column`
- **PPO**: Requires `value_model_name` in model config and additional PPO hyperparameters in training config
- **GRPO/GSPO**: Math/reasoning task optimization; requires `prompt_column` and `answer_column`

**Memory Optimization Tips:**
- Reduce `per_device_train_batch_size` if you encounter OOM errors
- Enable `gradient_checkpointing` for larger models
- Use `load_in_4bit: true` with LoRA configuration for QLoRA (most memory-efficient)
- Use `load_in_8bit: true` for 8-bit quantization (moderate memory savings)
- Use `use_lora: true` for parameter-efficient fine-tuning (LoRA without quantization)
- Set `dataloader_pin_memory: true` only if you have sufficient system RAM

**Learning Rate Guidelines:**
- Full fine-tuning: `1.0e-5` to `1.0e-6`
- LoRA: `1.0e-4`
- QLoRA: `2.0e-4`

## 📊 Training Progress

**Monitoring options:**
- Logs are stored in `../logs/`
- If Weights & Biases is enabled, follow real-time progress at wandb
- Final models are saved under `./models/`