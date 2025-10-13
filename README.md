# FAI-RL: Foundation of AI - Reinforcement learning Library

A modular, production-ready library designed for **easy training, inference, and evaluation** of language models using reinforcement learning methods. Currently supports: 
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Preference Optimization)
- GSPO (Group Sequence Policy Optimization)

## 🚀 Quick Start

Get started with installation, training, inference, and evaluation in just a few commands:

### 📦 Installation

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 FAI-RL
```


### Training

Train a model using SFT, DPO, PPO, GRPO, or GSPO:

```bash
# Single GPU training
fai-rl-train --config configs/training/sft/llama3_3B_lora_recipe.yaml --num-gpus 1

# Multi-GPU training in background (8 GPUs)
fai-rl-train --config configs/training/sft/llama3_3B_lora_recipe.yaml --num-gpus 8 --nohup

# Runtime parameter overrides
fai-rl-train --config configs/training/sft/llama3_3B_lora_recipe.yaml --num-gpus 8 --nohup \
model.base_model_name=Qwen/Qwen3-4B-Instruct-2507 \
training.num_train_epochs=3
```

### Inference

Generate responses from your trained models:

```bash
# Run inference on trained model
fai-rl-inference --config configs/inference/llama3_3B_inference.yaml

# Run inference with debug mode
fai-rl-inference --config configs/inference/llama3_3B_inference.yaml --debug
```

### Evaluation

Evaluate model performance on benchmarks:

```bash
# Evaluate on MMLU benchmark
fai-rl-eval --config configs/evaluation/mmlu/llama3_3B_recipe.yaml

# Evaluate with debug output
fai-rl-eval --config configs/evaluation/mmlu/llama3_3B_recipe.yaml --debug
```

-----

## Flexible Configuration System
* YAML-based configuration for all training parameters
* Pre-configured recipes for popular models
* DeepSpeed ZeRO-3 integration for distributed training


## 📁 Project Structure

```
FAI-RL/
├── core/                      # Core framework components
├── trainers/                  # Training method implementations
├── inference/                 # Inference components
├── evaluations/               # Evaluation system
├── configs/                   # Configuration files
│   ├── training/              # Training configurations
│   ├── inference/             # Inference configurations
│   ├── evaluation/            # Evaluation configurations
│   └── deepspeed/             # DeepSpeed ZeRO configurations
├── utils/                     # Utility modules
├── scripts/                   # Scripts
├── logs/                      # Training logs (auto-generated)
└── outputs/                   # Inference output (auto-generated)
```

-----

## 🔗 Quick Links

* **[Training Guide](./trainers/README.md)** - Comprehensive guide to configuring and running model training with detailed parameter explanations
* **[Inference Guide](./inference/README.md)** - Running model inference and text generation
* **[Evaluation Guide](./evaluations/README.md)** - Evaluating model performance on standard benchmarks

## Memory Optimization

FAI-RL supports various techniques to train large models efficiently:

* **Full Fine-tuning:** Train all model parameters (requires most memory)
* **LoRA:** Parameter-efficient training (~10% memory of full fine-tuning)
* **QLoRA:** 4-bit quantized LoRA (train 7B+ models on single consumer GPU)
* **DeepSpeed ZeRO-3:** Distributed training for models that don't fit on single GPU

## 🧪 Tested Environment

This framework has been validated on:

* **Instance:** AWS EC2 p4d.24xlarge
* **GPUs:** 8 x NVIDIA A100-SXM4-80GB (80GB VRAM each)
* **CPU:** 96 vCPUs
* **Memory:** 1152 GiB
* **Storage:** 8TB NVMe SSD
* **Network:** 400 Gbps

🛠 For Maintainers

To release a new version of FAI-RL:

1. Update version in pyproject.toml:
```bash
[project]
name = "FAI-RL"
version = "__NEW_VERSION__"
```

2. Build and upload the package:
```bash
# Upgrade pip and build tools
pip install --upgrade pip
pip install build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```