# FAI-RL: Foundation AI - Reinforcement Learning Library

<div align="center" style="line-height: 1;">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>
</div>

A production-ready framework for training, inference, evaluation using advanced reinforcement learning techniques. Built for researchers and practitioners who need a flexible, scalable solution for LLM fine-tuning.

## Overview

FAI-RL provides a unified, extensible framework for fine-tuning language models with the state-of-the-art algorithms:

- 🎯 **Supports Multiple RL Algorithms**: DPO, PPO, GRPO, GSPO implementations as well as support for Supervised Fine-Tuning.
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
- [Memory Optimization](#memory-optimization)
- [System Requirements](#-system-requirements)
- [License](#-license)

## 📦 Installation

### Install the Package

```bash 
pip install --extra-index-url https://download.pytorch.org/whl/cu118 FAI-RL
```

### Clone the Repository for Configuration Recipes

```bash
git clone https://github.com/Roblox/FAI-RL.git
cd FAI-RL
```

> **Package**: [https://pypi.org/project/FAI-RL/](https://pypi.org/project/FAI-RL/)  
> **Note**: The `--extra-index-url` flag ensures PyTorch is installed with CUDA 11.8 support.

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


> **Note**: W&B integration is optional. If not logged in, training will proceed without experiment tracking.

## 🚀 Quick Start

### Training

Train a model using any of the supported algorithms (DPO, PPO, GRPO, GSPO, SFT):

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

FAI-RL implements five state-of-the-art reinforcement learning algorithms for language model fine-tuning:

| Algorithm | Full Name | Description | Best For |
|-----------|-----------|-------------|----------|
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
│   ├── training/              # Training recipes (sft/, dpo/, ppo/, grpo/, gspo/)
│   ├── inference/             # Inference recipes
│   └── evaluation/            # Evaluation recipes (mmlu/, gsm8k/)
├── configs/                   # DeepSpeed configurations
│   └── deepspeed/             # ZeRO-3 configs for 1/2/4/8 GPUs
├── utils/                     # Shared utilities
│   └── hosted_llm_config.py   # Custom API endpoint configuration
└── [auto-generated]
    ├── models/                # Trained model checkpoints
    ├── outputs/               # Inference and evaluation results
    └── logs/                  # Training logs
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
```

</details>
