# Training Recipes

YAML configuration files for all supported training algorithms. Each recipe is a self-contained, fully-commented template — copy one, fill in your model/dataset/output paths, and launch.

📖 **[Full training guide →](../../trainers/README.md)**

## Recipe Index

### CPT (Continuous Pre-Training)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `cpt/qwen3_4B_lora.yaml` | HuggingFace Hub | LoRA, Qwen3-4B |
| `cpt/qwen3_4B_qlora.yaml` | HuggingFace Hub | QLoRA (4-bit), Qwen3-4B |
| `cpt/qwen3_30B_A3B_lora.yaml` | HuggingFace Hub | LoRA, Qwen3-30B-A3B MoE |
| `cpt/qwen3_30B_A3B_qlora.yaml` | HuggingFace Hub | QLoRA (4-bit), Qwen3-30B-A3B MoE |
| `cpt/qwen3_4B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `cpt/qwen3_4B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

### SFT (Supervised Fine-Tuning)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `sft/llama3_3B_lora.yaml` | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `sft/llama3_3B_qlora.yaml` | HuggingFace Hub | QLoRA (4-bit), Llama-3.2-3B |
| `sft/llama3_3B_full.yaml` | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |
| `sft/llama3_3B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `sft/llama3_3B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

### DPO (Direct Preference Optimization)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `dpo/llama3_3B_lora.yaml` | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `dpo/llama3_3B_qlora.yaml` | HuggingFace Hub | QLoRA (4-bit), Llama-3.2-3B |
| `dpo/llama3_3B_full.yaml` | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |
| `dpo/llama3_3B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `dpo/llama3_3B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

### PPO (Proximal Policy Optimization)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `ppo/llama3_3B_lora.yaml` | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `ppo/llama3_3B_qlora.yaml` | HuggingFace Hub | QLoRA (4-bit), Llama-3.2-3B |
| `ppo/llama3_3B_full.yaml` | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |

### GRPO (Group Relative Policy Optimization)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `grpo/llama3_3B_lora.yaml` | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `grpo/llama3_3B_full.yaml` | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |
| `grpo/llama3_3B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `grpo/llama3_3B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

### GSPO (Group Sequence Policy Optimization)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `gspo/llama3_3B_lora.yaml` | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `gspo/llama3_3B_full.yaml` | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |
| `gspo/llama3_3B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `gspo/llama3_3B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

## Dataset Sources

All algorithms support three dataset sources. The source is selected by the value of `data.datasets[n].name`:

| Source | `name` value | Example |
|--------|-------------|---------|
| HuggingFace Hub | Dataset name (no extension) | `"openai/gsm8k"` |
| Local file | File path ending in `.jsonl`, `.json`, `.csv`, or `.parquet` | `"data/train.jsonl"` |
| S3 | `s3://` URI ending in a supported extension | `"s3://my-bucket/datasets/train.jsonl"` |

### Required JSONL fields by algorithm

| Algorithm | Required fields |
|-----------|----------------|
| CPT | `text` |
| SFT | `prompt`, `response` |
| DPO | `prompt`, `chosen`, `rejected` |
| PPO | `prompt`, `chosen`, `rejected` |
| GRPO | `prompt`, `answer` |
| GSPO | `prompt`, `answer` |

Column names are configurable — use `text_column`, `prompt_column`, `answer_column`, `chosen_column`, `rejected_column` in the dataset entry to remap them.

## Quick Start

```bash
# Pick a recipe and launch
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 1

# Override dataset and output dir at the command line
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 4 \
  data.datasets[0].name=your-org/your-dataset \
  training.output_dir=models/my_model

# Use a local file
fai-rl-train --recipe recipes/training/sft/llama3_3B_local_file.yaml --num-gpus 1 \
  data.datasets[0].name=data/train.jsonl

# Use an S3 file
fai-rl-train --recipe recipes/training/sft/llama3_3B_s3_file.yaml --num-gpus 1 \
  data.datasets[0].name=s3://my-bucket/datasets/train.jsonl
```

> Running from the repo directly: replace `fai-rl-train` with `python trainers/train.py`.

## S3 Dataset Loading

Set `name` to an `s3://` URI. AWS credentials are resolved from the standard boto3 chain (IAM role, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars, `~/.aws/credentials`).

```yaml
data:
  datasets:
    - name: "s3://my-bucket/datasets/train.jsonl"
      prompt_column: "prompt"
      dataset_columns: ["prompt", "response"]
      s3_region: null        # override AWS region if needed
      s3_endpoint_url: null  # set for MinIO or other S3-compatible stores
```

The file is downloaded to a temporary path at training startup, loaded into the HuggingFace Arrow cache, and then deleted. Multiple S3 datasets (or a mix of S3, local, and Hub) can be listed and will be concatenated before training.

## Naming Convention

```
<algorithm>/<model>_<variant>.yaml

Variants:
  lora          LoRA fine-tuning (recommended starting point)
  qlora         QLoRA (4-bit quantization + LoRA)
  full          Full parameter fine-tuning
  local_file    Local dataset file (.jsonl / .json / .csv / .parquet)
  s3_file       S3 dataset file (s3://bucket/key.ext)
```
