# FAI-RL: Foundation AI - Reinforcement Learning Library

<div align="center" style="line-height: 1;">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>
</div>

A production-ready framework for training, inference, evaluation using advanced reinforcement learning techniques. Built for researchers and practitioners who need a flexible, scalable solution for LLM fine-tuning.

## Overview

FAI-RL provides a unified, extensible framework for fine-tuning language models with the state-of-the-art algorithms:

- 🎯 **Supports Multiple RL Algorithms**: DPO, PPO, GRPO, GSPO implementations as well as support for Supervised Fine-Tuning and Continuous Pre-Training.
- 🚀 **Production Ready**: Validated on AWS p4d instances with 8x A100 GPUs
- 📦 **Simple Configuration**: YAML-based configs with CLI override support
- ⚡ **Memory Efficient**: Full support for LoRA, QLoRA, and DeepSpeed ZeRO-3
- 🔧 **Highly Extensible**: Custom reward functions, dataset templates, and API integrations

## Table of Contents

- [Installation](#-installation)
- [Authentication & Setup](#-authentication--setup)
- [Quick Start](#-quick-start)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Supported Methods](#supported-methods)
- [Key Features](#key-features)
- [Project Structure](#-project-structure)
- [S3 Checkpoint Upload](#-s3-checkpoint-upload)
- [Memory Optimization](#memory-optimization)
- [System Requirements](#-system-requirements)
- [License](#-license)

## 📦 Installation

### Install the Package

We assume the user environment already has the necessary ML libraries
installed (notably `torch`, with a CUDA build matching the host).

**For Linux/Windows with NVIDIA GPUs (CUDA):**

```bash
pip install FAI-RL[cuda]
```

**For macOS (Apple Silicon or Intel):**

```bash
pip install FAI-RL==0.1.19
```

### Clone the Repository for Configuration Recipes

```bash
git clone https://github.com/Roblox/FAI-RL.git
cd FAI-RL
```

> **Package**: [https://pypi.org/project/FAI-RL/](https://pypi.org/project/FAI-RL/)

## 🔑 Authentication & Setup

Before training or using models, you'll need to authenticate with HuggingFace and optionally set up experiment tracking with Weights & Biases.

### HuggingFace Authentication

Login to HuggingFace to access models and datasets:

```bash
huggingface-cli login
```

You'll be prompted to enter your HuggingFace access token. You can create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**What this enables:**
- Access gated models (if you have permission)


### Weights & Biases (Optional)

Login to Weights & Biases for experiment tracking and visualization:

```bash
wandb login
```

You'll be prompted to enter your W&B API key. Get your API key at [https://wandb.ai/authorize](https://wandb.ai/authorize).

For self-hosted or private W&B deployments, set `WANDB_BASE_URL` before training:

```bash
export WANDB_BASE_URL="https://your-wandb-instance.com"
```

The default value (`https://api.wandb.ai`) points to the public W&B cloud. This can also be set directly in the recipe under `wandb.base_url`.

> **Note**: W&B integration is optional. If not logged in, training will proceed without experiment tracking.

## 🚀 Quick Start

### Training

Train a model using any of the supported algorithms (CPT, SFT, DPO, PPO, GRPO, GSPO):

```bash
# Single GPU training with LoRA
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 1

# Multi-GPU training with DeepSpeed
fai-rl-train --recipe recipes/training/dpo/llama3_3B_lora.yaml --num-gpus 8

# Override parameters from CLI
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 4 \
  training.learning_rate=5e-5 \
  training.num_train_epochs=3
```

📖 **[Complete Training Guide →](./trainers/README.md)**

### Inference

Generate text completions from trained or base models:

```bash
# Run inference on a trained model
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml

# Use debug mode for detailed logging
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml --debug
```

📖 **[Complete Inference Guide →](./inference/README.md)**

### Evaluation

Evaluate model performance on academic benchmarks (MMLU, GSM8K):

```bash
# Evaluate on MMLU benchmark
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml --debug
```

📖 **[Complete Evaluation Guide →](./evaluations/README.md)**

## Supported Algorithms

FAI-RL supports six training algorithms for language model fine-tuning:

| Algorithm | Full Name | Description | Best For |
|-----------|-----------|-------------|----------|
| **CPT** | Continuous Pre-Training | Next-token prediction on raw text; no chat template | Domain adaptation, corpus ingestion |
| **SFT** | Supervised Fine-Tuning | Direct supervised learning from labeled examples | Instruction fine-tuning and foundational model fine-tuning |
| **DPO** | Direct Preference Optimization | Alignment via preference learning without explicit reward models | Human preference alignment, chat model training |
| **PPO** | Proximal Policy Optimization | Policy gradient method with value function and reward model | Complex reward functions, multi-objective optimization |
| **GRPO** | Group Relative Policy Optimization | Efficient preference learning with group-based comparison | Reasoning tasks, competitive response generation |
| **GSPO** | Group Sequence Policy Optimization | Advanced sequence-level policy optimization | Complex multi-step reasoning, mathematical problem-solving |

### Training Configurations

All algorithms support three efficiency modes:

| Mode | Memory Usage | Training Speed | Best For |
|------|-------------|---------------|----------|
| **Full Fine-tuning** | High (baseline) | Fastest | Small models (<3B params), maximum performance |
| **LoRA** | Low (~10% of full) | Fast | Most use cases, balanced efficiency |
| **QLoRA** | Very Low (~3-4GB for 7B model) | Moderate | Large models on consumer GPUs |

Additional features supported across all algorithms:
- ✅ Multi-GPU training with DeepSpeed ZeRO-3
- ✅ Gradient checkpointing for memory efficiency
- ✅ Custom reward functions and dataset templates
- ✅ Weights & Biases integration for experiment tracking
- ✅ Automatic S3 checkpoint upload (supports S3-compatible stores)

## Key Features

### 🎯 Flexible Configuration System
- **YAML-based recipes** with comprehensive inline documentation for all parameters
- **CLI overrides** for runtime parameter changes without editing files
- **Pre-configured templates** for popular models (Llama 3, Qwen 3, etc.)
- **Easy experimentation** with hyperparameter tuning

### 🔧 Extensible Architecture

**Custom Reward Functions:**
- `exact_match_reward_func` - Accuracy-based rewards for verifiable tasks
- `structured_xml_reward_func` - Format-based rewards for structured outputs
- Easy to add your custom reward function

**Dataset Templates:**
- `GSM8KTemplate` - Math problem formatting with chain-of-thought
- `OpenMathInstructTemplate` - Mathematical instruction formatting

**Pluggable Components:**
- Extensible trainer base classes for new algorithms
- HuggingFace Transformers and TRL integration
- Custom dataset processing pipelines

### 🌐 Multi-Provider API Support

Native support for commercial LLM APIs with automatic provider detection for inference and evaluation:

**Supported Providers:**
- 🤖 **OpenAI** (GPT-5, GPT-4.5, GPT-4.1, etc.)
- 🧠 **Google** (Gemini Pro, Gemini Flash)
- 💬 **Anthropic** (Claude 4.5 Sonnet, Opus, etc.)
- 🏠 **Hosted LLM** (self-hosted or custom endpoints)

**Configuration Example:**

```yaml
# OpenAI ChatGPT - provider detected from endpoint URL
inference:
  api_endpoint: "https://api.openai.com/v1/chat/completions"
  api_key: "sk-..."
  model: "gpt-4.1"  # Just the model name, no prefix needed!

# Google Gemini - provider detected from endpoint URL
inference:
  api_endpoint: "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
  api_key: "AIza..."
  model: "gemini-2.5-pro"

# Anthropic Claude - provider detected from endpoint URL
inference:
  api_endpoint: "https://api.anthropic.com/v1/messages"
  api_key: "sk-ant-..."
  model: "claude-sonnet-4-5-20250929"

# Hosted LLM - any custom or self-hosted model endpoint
inference:
  api_endpoint: "https://your-hosted-endpoint.com/v1/chat"
  api_key: "your-api-key"
  model: "your-model-name"
```

**Customization for Custom APIs:**

If your hosted LLM uses a non-OpenAI format, customize `utils/hosted_llm_config.py`:
- `build_hosted_llm_request()` - Modify request payload format
- `parse_hosted_llm_response()` - Customize response parsing
- `build_hosted_llm_headers()` - Adjust authentication headers

Each function includes detailed examples and inline documentation.


## 📁 Project Structure

```
FAI-RL/
├── core/                      # Core framework components
├── trainers/                  # Algorithm implementations
│   ├── rewards/               # Custom reward functions
│   │   ├── accuracy_rewards.py
│   │   └── format_rewards.py
│   └── templates/             # Dataset formatting templates
│       ├── gsm8k_template.py
│       └── openmathinstruct_template.py
├── inference/                 # Inference system
├── evaluations/               # Evaluation system
│   └── eval_datasets/         # Dataset-specific evaluation logic
│       ├── mmlu.py
│       └── gsm8k.py
├── recipes/                   # YAML configuration files
│   ├── training/              # Training recipes (cpt/, sft/, dpo/, ppo/, grpo/, gspo/)
│   ├── inference/             # Inference recipes
│   └── evaluation/            # Evaluation recipes (mmlu/, gsm8k/)
├── configs/                   # DeepSpeed configurations
│   └── deepspeed/             # ZeRO-3 configs for 1/2/4/8 GPUs
├── utils/                     # Shared utilities
│   ├── s3_utils.py            # S3 checkpoint upload callback
│   └── hosted_llm_config.py   # Custom API endpoint configuration
└── [auto-generated]
    ├── models/                # Trained model checkpoints
    ├── outputs/               # Inference and evaluation results
    └── logs/                  # Training logs
```

## ☁️ S3 Checkpoint Upload

FAI-RL can automatically upload checkpoints and the final fine-tuned model to Amazon S3 (or any S3-compatible store such as MinIO). Uploads run in background threads so they never block training.

### Prerequisites

Configure AWS credentials using any standard method (environment variables, `~/.aws/credentials`, IAM role, etc.):

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# Option 2: AWS CLI
aws configure
```

### Configuration

Add an `s3` section to your training recipe YAML:

```yaml
s3:
  enabled: true                                          # Enable S3 upload
  bucket: "your-s3-bucket"                               # S3 bucket name
  prefix: "your-s3-prefix"                               # Key prefix (folder path inside bucket)
  region: null                                           # AWS region (null = use default)
  endpoint_url: null                                     # Custom S3-compatible endpoint (e.g. MinIO)
  upload_checkpoints: true                               # Upload intermediate checkpoints (at every save_steps)
  upload_final_model: true                               # Upload the final model at end of training
  delete_local_after_upload: false                       # Delete local files after successful upload
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Master switch for the S3 upload feature |
| `bucket` | string | `""` | Target S3 bucket name (required when enabled) |
| `prefix` | string | `""` | Key prefix under which all uploads are stored |
| `region` | string | `null` | AWS region; falls back to `AWS_DEFAULT_REGION` or boto3 default |
| `endpoint_url` | string | `null` | Custom endpoint for S3-compatible stores (e.g. `http://minio:9000`) |
| `upload_checkpoints` | bool | `true` | Upload each intermediate checkpoint saved at `save_steps` intervals |
| `upload_final_model` | bool | `true` | Upload the final model directory at the end of training |
| `delete_local_after_upload` | bool | `false` | Remove local checkpoint directory after a successful upload |

### How It Works

1. **Intermediate checkpoints** -- When the trainer saves a checkpoint (every `training.save_steps` steps), the S3 callback uploads the entire checkpoint directory to `s3://<bucket>/<prefix>/checkpoint-<step>/` in a background thread.
2. **Final model** -- At the end of training, the output directory is uploaded to `s3://<bucket>/<prefix>/final/`.
3. **Non-blocking** -- All uploads happen on daemon threads. Training continues while files are being transferred. At the end of training, the callback waits for any remaining uploads to finish before the process exits.

### S3 Upload Structure

Given the example config above, the resulting S3 layout would be:

```
s3://your-s3-bucket/
└── checkpoints/qwen3-4B-inst-dpo-lora-150k/
    ├── checkpoint-100/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── ...
    ├── checkpoint-200/
    │   └── ...
    └── final/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...
```

## Memory Optimization

FAI-RL provides multiple techniques for efficient training of large models on limited hardware:

### Optimization Techniques

| Technique | Memory Savings | Speed Impact | Configuration |
|-----------|---------------|--------------|---------------|
| **LoRA** | ~90% reduction | Minimal | `use_lora: true` + LoRA params |
| **QLoRA** | ~95% reduction | Moderate | `load_in_4bit: true` + LoRA params |
| **8-bit Quantization** | ~50% reduction | Minimal | `load_in_8bit: true` |
| **Gradient Checkpointing** | ~30-50% reduction | 20% slower | `gradient_checkpointing: true` |
| **DeepSpeed ZeRO-3** | Distributed across GPUs | Varies | Auto-enabled for multi-GPU |


### Optimization Strategy

1. **Start with QLoRA** if GPU memory is limited (<16GB)
2. **Use LoRA** for balanced efficiency on mid-range GPUs (16-40GB)
3. **Full fine-tuning** only for small models or high-end GPUs (80GB+)
4. **Enable gradient checkpointing** if still encountering OOM errors
5. **Use DeepSpeed ZeRO-3** for multi-GPU setups to distribute memory load

## 🧪 System Requirements

### Validated on Hardware

This framework has been validated on:

* **Instance:** AWS EC2 p4d.24xlarge
* **GPUs:** 8 x NVIDIA A100-SXM4-80GB (80GB VRAM each)
* **CPU:** 96 vCPUs
* **Memory:** 1152 GiB
* **Storage:** 8TB NVMe SSD
* **Network:** 400 Gbps

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## For Maintainers

<details>
<summary>Publishing a New Release</summary>

1. **Update version** in `pyproject.toml`:
```toml
[project]
name = "FAI-RL"
version = "X.Y.Z"  # Increment version
```

2. **Build and publish**:
```bash
# Install build tools
pip install --upgrade pip build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# Upload to PyPI (requires credentials)
python -m twine upload dist/*

# Or upload to test PyPi (requires credentials)
python -m twine upload --repository testpypi dist/*
```

</details>
