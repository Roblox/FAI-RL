# FAI-RL Evaluation

Comprehensive model evaluation system for assessing language model performance on academic benchmarks and custom datasets. Supports automatic answer extraction, accuracy calculation, and detailed result analysis.

## 🚀 Quick Start

### Basic Evaluation

```bash
# Evaluate on MMLU benchmark
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml

# Evaluate with debug mode for detailed logging
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml --debug

# Run evaluation in background with nohup
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml --nohup
```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model path and output file
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml \
  evaluation.model_path=models/my_custom_model/checkpoint-100 \
  evaluation.output_file=outputs/my_eval_results.csv

# Override dataset subset and generation parameters
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml \
  evaluation.dataset_subset=college_mathematics \
  evaluation.temperature=0.0 \
  evaluation.do_sample=false
```

## 📊 Output

### Output Files

Evaluation generates a detailed CSV file at the specified `output_file` path:

```
outputs/
└── llama3_3B_Inst_SFT_lora_v1_checkpoint100_evaluation.csv
```

## 🔬 Supported Benchmarks

### MMLU (Massive Multitask Language Understanding)
- **Dataset**: `cais/mmlu`
- **Task Type**: Multiple choice questions across 57 academic subjects
- **Subsets**: 57 subjects (e.g., `abstract_algebra`, `college_biology`, `high_school_physics`)
- **Evaluation**: Automatic JSON answer extraction and accuracy calculation
- **Example Config**: `recipes/evaluation/mmlu/llama3_3B.yaml`

### GSM8K (Grade School Math 8K)
- **Dataset**: `openai/gsm8k`
- **Task Type**: Grade school math word problems requiring multi-step reasoning
- **Subsets**: `main` (8.5K problems: 7,473 train, 1,319 test)
- **Evaluation**: Automatic numeric answer extraction and accuracy calculation
- **Example Config**: `recipes/evaluation/gsm8k/llama3_8B_vanilla.yaml`