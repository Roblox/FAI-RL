# FAI-RL Training

Training implementations supporting CPT (Continuous Pre-Training), SFT (Supervised Fine-Tuning), SFT_VLM (multimodal vision-language SFT), DPO (Direct Preference Optimization), GRPO (Group Relative Policy Optimization), and GSPO (Group Sequence Policy Optimization) methods.

> **Multimodal (SFT_VLM):** image + text fine-tuning of vision-language models has its own recipes and dataset schema. See the [SFT_VLM recipe guide](../recipes/training/README.md#multimodal-datasets-sft_vlm).

## 🚀 Quick Start

### Basic Training

```bash
# Single GPU training with SFT
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 1

# Multi-GPU training with DPO (8 GPUs)
fai-rl-train --recipe recipes/training/dpo/llama3_3B_lora.yaml --num-gpus 8

# Run training in background with nohup
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 8 --nohup
```

> **Running with Local Code**: If running directly from the repository, use `python trainers/train.py` instead of `fai-rl-train`:
> ```bash
> python trainers/train.py --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 1
> ```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model and training parameters
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 4 \
  model.base_model_name=Qwen/Qwen3-4B-Instruct-2507 \
  training.num_train_epochs=3 \
  training.learning_rate=5.0e-5

# Override dataset and output directory
fai-rl-train --recipe recipes/training/dpo/llama3_3B_lora.yaml --num-gpus 8 --nohup \
  data.datasets[0].name=your-org/your-dataset \
  training.output_dir=models/my_custom_model
```

## 🔧 Configuration

All configuration files are located in `../recipes/training/` and include comprehensive inline documentation. Each config file is fully self-documenting with detailed comments explaining every parameter.

**Available Config Templates:**

HuggingFace Hub datasets:
- **CPT (Continuous Pre-Training)**: `recipes/training/cpt/qwen3_4B_qlora.yaml`
- **SFT (Supervised Fine-Tuning)**: `recipes/training/sft/llama3_3B_lora.yaml`
- **DPO (Direct Preference Optimization)**: `recipes/training/dpo/llama3_3B_lora.yaml`
- **GRPO (Group Relative Policy Optimization)**: `recipes/training/grpo/llama3_3B_lora.yaml`
- **GSPO (Group Sequence Policy Optimization)**: `recipes/training/gspo/llama3_3B_lora.yaml`

Local file datasets:
- **CPT**: `recipes/training/cpt/qwen3_4B_local_file.yaml`
- **SFT**: `recipes/training/sft/llama3_3B_local_file.yaml`
- **DPO**: `recipes/training/dpo/llama3_3B_local_file.yaml`
- **GRPO**: `recipes/training/grpo/llama3_3B_local_file.yaml`
- **GSPO**: `recipes/training/gspo/llama3_3B_local_file.yaml`

S3 datasets:
- **CPT**: `recipes/training/cpt/qwen3_4B_s3_file.yaml`
- **SFT**: `recipes/training/sft/llama3_3B_s3_file.yaml`
- **DPO**: `recipes/training/dpo/llama3_3B_s3_file.yaml`
- **GRPO**: `recipes/training/grpo/llama3_3B_s3_file.yaml`
- **GSPO**: `recipes/training/gspo/llama3_3B_s3_file.yaml`

Each config file contains four main sections:
1. **Model Configuration** - Base model, quantization, and LoRA settings
2. **Data Configuration** - Dataset names, columns, and preprocessing
3. **Training Configuration** - Hyperparameters, optimization, and logging
4. **Weights & Biases** - Experiment tracking (optional)

Open any config file to see detailed inline documentation for all available parameters.

### Configuration Parameters

**Configuration Checklist:**
Replace the following values for your specific use case:
- `data.datasets.name` → your HuggingFace dataset(s) (e.g., "Anthropic/hh-rlhf" for DPO, "openai/gsm8k" for GRPO/GSPO, "nvidia/Aegis-AI-Content-Safety-Dataset-2.0" for SFT), **a local file path** (e.g., `data/train.jsonl`), or **an S3 URI** (e.g., `s3://bucket/datasets/train.jsonl`) — see [Local File Datasets](#local-file-datasets) and [S3 Datasets](#s3-datasets) below
- `data.datasets.text_column` → column containing raw text (CPT only; default: `"text"`)
- `data.datasets.prompt_column` / `answer_column` / `chosen_column` / `rejected_column` → adjust based on your dataset and algorithm
  - **CPT**: Use `text_column` (raw text, no chat template)
  - **SFT**: Use `prompt_column` and `answer_column`
  - **DPO**: Use `prompt_column`, `chosen_column`, and `rejected_column`
  - **GRPO/GSPO**: Use `prompt_column` and `answer_column`
- `training.algorithm` → choose from: `cpt`, `sft`, `sft_vlm`, `dpo`, `grpo`, `gspo`
- `training.output_dir` → your desired model output directory
- `wandb.*` → your Weights & Biases configuration (or set `enabled: false` to disable)

**Algorithm-Specific Notes:**
- **CPT**: Domain adaptation via next-token prediction on raw text; requires a `text_column` in dataset; no system prompt or chat template applied
- **SFT**: Best for initial instruction tuning; requires `prompt_column` and `answer_column` in dataset
- **DPO**: Preference-based method; requires `prompt_column`, `chosen_column`, and `rejected_column`
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

## 📊 Output & Monitoring

### Directory Structure

After training, the following directories will be created:

```
FAI-RL/
├── models/                   # Trained model checkpoints
│   └── llama3_3B_Inst_SFT_lora_v1/
│       ├── checkpoint-50/    # Intermediate checkpoints
│       ├── checkpoint-100/
│       └── ...
└── logs/                     # Training logs (if generated)
    └── training_YYYYMMDD_HHMMSS.log
```

## Local File Datasets

All training algorithms can load datasets directly from local files. Set `data.datasets[n].name` to a file path — the extension selects the loader automatically.

**Supported formats**

| Extension | Format |
|-----------|--------|
| `.jsonl` | Newline-delimited JSON (recommended) |
| `.json` | JSON array |
| `.csv` | Comma-separated values |
| `.parquet` | Apache Parquet |

Relative paths are resolved from the directory where `fai-rl-train` is launched. Absolute paths are used as-is.

**Expected JSONL schema by algorithm**

| Algorithm | Required fields |
|-----------|----------------|
| **SFT** | `prompt`, `response` |
| **CPT** | `text` |
| **DPO** | `prompt`, `chosen`, `rejected` |
| **GRPO** | `prompt`, `answer` |
| **GSPO** | `prompt`, `answer` |

**Example — SFT from a local JSONL file**

```yaml
data:
  datasets:
    - name: "data/train.jsonl"
      prompt_column: "prompt"
      dataset_columns: ["prompt", "response"]
```

```bash
fai-rl-train --recipe recipes/training/sft/llama3_3B_local_file.yaml --num-gpus 1 \
  data.datasets[0].name=data/my_train.jsonl
```

Multiple local files, or a mix of local files and Hub datasets, can be listed under `data.datasets` and will be concatenated before training.

## S3 Datasets

All training algorithms can load datasets directly from S3 (or any S3-compatible store). Set `data.datasets[n].name` to an `s3://` URI — the file extension selects the loader automatically.

AWS credentials are resolved from the standard boto3 chain: IAM role, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables, or `~/.aws/credentials`.

**Supported formats**

| Extension | Format |
|-----------|--------|
| `.jsonl` | Newline-delimited JSON (recommended) |
| `.json` | JSON array |
| `.csv` | Comma-separated values |
| `.parquet` | Apache Parquet |

**Example — SFT from an S3 JSONL file**

```yaml
data:
  datasets:
    - name: "s3://my-bucket/datasets/train.jsonl"
      prompt_column: "prompt"
      dataset_columns: ["prompt", "response"]
      s3_region: null        # override AWS region if needed
      s3_endpoint_url: null  # set for MinIO or other S3-compatible stores
```

```bash
fai-rl-train --recipe recipes/training/sft/llama3_3B_s3_file.yaml --num-gpus 1 \
  data.datasets[0].name=s3://my-bucket/datasets/my_train.jsonl
```

The file is downloaded to a temporary path at training startup, loaded into the HuggingFace Arrow cache, then deleted. Multiple S3 datasets, or a mix of S3, local, and Hub datasets, can be listed under `data.datasets` and will be concatenated before training.

## 💡 Best Practices

### Memory Management
- Start with `per_device_train_batch_size: 1` and increase if memory allows
- Use `gradient_accumulation_steps` to achieve larger effective batch sizes
- Enable `gradient_checkpointing: true` for memory-constrained scenarios
- Consider QLoRA (`load_in_4bit: true` + LoRA) for training large models on limited hardware

### Learning Rate Selection
- **Full fine-tuning**: 1e-5 to 1e-6
- **LoRA**: 1e-4
- **QLoRA**: 2e-4

### Checkpoint Strategy
- Set `save_steps` based on dataset size (e.g., every 10% of total steps)
- Keep `save_only_model: true` to save disk space
- Use `eval_steps` to monitor validation performance periodically

### Dataset Preparation
- Ensure column names in config match your dataset
- Set `max_length` based on your typical sequence length
- Use `dataset_num_proc` > 1 to speed up preprocessing for large datasets

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce `per_device_train_batch_size`
2. Enable `gradient_checkpointing: true`
3. Switch to QLoRA: set `load_in_4bit: true` and configure LoRA
4. Reduce `max_length` or `max_prompt_length`

### Slow Training
1. Increase `dataloader_num_workers` (e.g., 4-8)
2. Set `dataloader_pin_memory: true` if sufficient RAM available
3. Verify `gradient_accumulation_steps` isn't unnecessarily high
4. Consider using DeepSpeed for multi-GPU setups
