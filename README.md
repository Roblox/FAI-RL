# FAI-RL: Foundation AI - Reinforcement Learning

**FAI-RL** is a modular, production-ready library for training, inference, and evaluation of large language models using state-of-the-art reinforcement learning methods.

## Overview

FAI-RL provides a unified framework for fine-tuning language models with multiple RL algorithms, featuring:

- 🎯 **Multiple RL Algorithms**: SFT, DPO, PPO, GRPO, GSPO
- 🚀 **Production Ready**: Battle-tested on large-scale deployments
- 📦 **Easy to Use**: Simple YAML configuration and CLI interface
- ⚡ **Memory Efficient**: LoRA, QLoRA, and DeepSpeed ZeRO-3 support
- 🔧 **Modular Design**: Extensible architecture for custom implementations

## Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Supported Methods](#supported-methods)
- [Key Features](#key-features)
- [Project Structure](#-project-structure)
- [Memory Optimization](#memory-optimization)
- [System Requirements](#-system-requirements)

## 📦 Installation

Install FAI-RL from PyPI:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 FAI-RL
```

For development installation:

```bash
git clone https://github.com/your-org/FAI-RL-OSS.git
cd FAI-RL-OSS
pip install -e .
```

**PyPI Package**: [https://pypi.org/project/FAI-RL/](https://pypi.org/project/FAI-RL/)

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## 🚀 Quick Start

### Training

Train a model using SFT, DPO, PPO, GRPO, or GSPO:

```bash
# Single GPU training
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 1
```

📖 **[See detailed Training Guide →](./trainers/README.md)**

### Inference

Generate responses from your trained models:

```bash
# Run inference with debug mode
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml --debug
```

📖 **[See detailed Inference Guide →](./inference/README.md)**

### Evaluation

Evaluate model performance on benchmarks:

```bash
# Evaluate with debug output
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml --debug
```

📖 **[See detailed Evaluation Guide →](./evaluations/README.md)**

## Supported Methods

FAI-RL implements state-of-the-art reinforcement learning algorithms for language model fine-tuning:

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Supervised Fine-Tuning | Initial instruction tuning on high-quality datasets |
| **DPO** | Direct Preference Optimization | Align models with human preferences without reward models |
| **PPO** | Proximal Policy Optimization | Classic RL approach with value functions and rewards |
| **GRPO** | Group Relative Preference Optimization | Efficient preference learning with group-based sampling |
| **GSPO** | Group Sequence Policy Optimization | Advanced sequence-level optimization |

Each method supports:
- ✅ Full fine-tuning
- ✅ LoRA (Low-Rank Adaptation)
- ✅ QLoRA (4-bit Quantized LoRA)
- ✅ Multi-GPU training with DeepSpeed

## Key Features

### 🎯 **Flexible Configuration System**
- YAML-based configuration for all training parameters
- Pre-configured recipes for popular models (Llama, Qwen, etc.)
- Easy hyperparameter tuning and experimentation

### 🔧 **Modular Architecture**
- Extensible trainer base classes
- Custom reward functions
- Pluggable dataset templates
- Easy integration with HuggingFace ecosystem


## 📁 Project Structure

```
FAI-RL/
├── core/                      # Core framework components
├── trainers/                  # Training method implementations
├── inference/                 # Inference components
├── evaluations/               # Evaluation system
├── recipes/                   # Recipe configuration files
│   ├── training/              # Training recipes
│   ├── inference/             # Inference recipes
│   └── evaluation/            # Evaluation recipes
├── configs/                   # Core configuration files
│   └── deepspeed/             # DeepSpeed ZeRO configurations
├── utils/                     # Utility modules
├── logs/                      # Training logs (auto-generated)
└── outputs/                   # Inference output (auto-generated)
```

## Memory Optimization

FAI-RL supports various techniques to train large models efficiently:

| Technique | Memory Usage | Speed | Best For |
|-----------|-------------|-------|----------|
| **Full Fine-tuning** | High (100%) | Fastest | Small models, ample GPU memory |
| **LoRA** | Low (~10%) | Fast | Most use cases, balanced efficiency |
| **QLoRA** | Very Low (~25% of LoRA) | Medium | Large models (7B+) on consumer GPUs |
| **DeepSpeed ZeRO-3** | Distributed | Variable | Models exceeding single GPU capacity |

### Example Memory Requirements

- **Llama-3 8B Full**: ~32GB VRAM
- **Llama-3 8B LoRA**: ~12GB VRAM
- **Llama-3 8B QLoRA**: ~6GB VRAM

## 🧪 System Requirements

### Validated on Hardware

This framework has been validated on:

* **Instance:** AWS EC2 p4d.24xlarge
* **GPUs:** 8 x NVIDIA A100-SXM4-80GB (80GB VRAM each)
* **CPU:** 96 vCPUs
* **Memory:** 1152 GiB
* **Storage:** 8TB NVMe SSD
* **Network:** 400 Gbps

## ⭐ For Maintainers

<details>

### Publishing a New Release

1. Update version in `pyproject.toml`:
```toml
[project]
name = "FAI-RL"
version = "X.Y.Z"  # Update version here
```

2. Build and publish:
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
