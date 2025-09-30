# FAI-RL: Framework for Aligned and Interactive Reinforcement Learning

A modular, production-ready library designed for **easy training, inference, and evaluation** of language models using reinforcement learning methods. Currently supports 
- DPO (Direct Preference Optimization),
- GRPO (Group Relative Preference Optimization), and
- GSPO (Group Sequence Policy Optimization).

## 🚀 Quick Start

Get started with installation, training, inference, and evaluation in just a few commands:

### 📦 Installation

Follow these steps to set up the FAI-RL library on your local machine. The first time, you'll need to clone the repository, create a virtual environment, and install the required dependencies. For subsequent runs, you only need to activate the virtual environment.

#### Setup (first time only)

```bash
# Clone the repository
git clone https://github.com/Roblox/FAI-RL.git
cd FAI-RL

# Create virtual environment
python -m venv venv_fai_rl
source venv_rl_library/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Next runs

```bash
source venv_fai_rl/bin/activate
```

### Training

Train a model using DPO, GRPO, or GSPO:

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

- **[Training Guide](./trainers/README.md)** - Configure and run model training
- **[Inference Guide](./inference/README.md)** - Run model inference and generation  
- **[Evaluation Guide](./evaluations/README.md)** - Evaluate model performance on benchmarks
