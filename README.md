# FAI-RL: Foundation of AI - Reinforcement learning Library

A modular, production-ready library designed for **easy training, inference, and evaluation** of language models using reinforcement learning methods. Currently supports: 
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Preference Optimization)
- GSPO (Group Sequence Policy Optimization)

### Flexible Configuration System
* YAML-based configuration for all training parameters
* Pre-configured recipes for popular models
* DeepSpeed ZeRO-3 integration for distributed training

## 🚀 Quick Start

Get started with installation, training, inference, and evaluation in just a few commands:

### 📦 Installation

#### Option 1: Install from PyPI (Recommended)

```bash
pip install FAI-RL
```

#### Option 2: Install from source

```bash
# Clone the repository
git clone https://github.com/Roblox/FAI-RL.git
cd FAI-RL

# Install in development mode
pip install -e .
```

#### Option 3: Manual setup with virtual environment

```bash
# Clone the repository
git clone https://github.com/Roblox/FAI-RL.git
cd FAI-RL

# Create virtual environment
python -m venv venv_fai_rl
source venv_fai_rl/bin/activate

# Install the package
pip install -e .
```

### Training

Train a model using SFT, DPO, PPO, GRPO, or GSPO:

```bash
# Single GPU training
./scripts/run_training.sh \
    --config configs/training/dpo/llama3_3B_recipe.yaml \
    --num-gpus 1

# Multi-GPU training (8 GPUs)
./scripts/run_training.sh \
    --config configs/training/dpo/llama3_3B_recipe.yaml \
    --num-gpus 8 \
    --nohup  # Run in background
```

### Inference

Generate responses from your trained models:

```bash
# Run inference on trained model
./scripts/run_inference.sh \
    --config configs/inference/llama3_3B_recipe.yaml

# Run inference with debug mode
./scripts/run_inference.sh \
    --config configs/inference/llama3_3B_recipe.yaml \
    --debug
```

### Evaluation

Evaluate model performance on benchmarks:

```bash
# Evaluate on MMLU benchmark
./scripts/run_evaluation.sh \
    --config configs/evaluation/mmlu/llama3_3B_recipe.yaml

# Evaluate with debug output
./scripts/run_evaluation.sh \
    --config configs/evaluation/mmlu/llama3_3B_recipe.yaml \
    --debug
```

-----

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

## Algorithm Selection Guide

Choose the right algorithm for your use case:

| Algorithm | Best For | Requirements | Key Benefits |
|-----------|----------|--------------|--------------|
| **SFT** | Initial instruction tuning, domain adaptation | Prompt-response pairs | Simple, fast, establishes baseline |
| **DPO** | Aligning to human preferences | Preference pairs (chosen/rejected) | No reward model needed, stable training |
| **PPO** | Complex sequential tasks, agentic workflows | Preference pairs + reward model | Most flexible, handles multi-turn interactions |
| **GRPO** | Math reasoning, efficiency-focused tasks | Question-answer pairs | No critic model, faster training |
| **GSPO** | Multi-turn RL, stable sequence-level optimization | Question-answer pairs | Better stability than GRPO |

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