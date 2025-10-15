# FAI-RL Evaluation

Comprehensive model evaluation system for assessing language model performance on academic benchmarks and custom datasets. Supports automatic answer extraction, accuracy calculation, and detailed result analysis.

## 🚀 Quick Start

### Basic Evaluation

```bash
# Evaluate on MMLU benchmark
fai-rl-eval --config recipes/evaluation/mmlu/llama3_3B.yaml

# Evaluate with debug mode for detailed logging
fai-rl-eval --config recipes/evaluation/mmlu/llama3_3B.yaml --debug
```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model path and output file
fai-rl-eval --config recipes/evaluation/mmlu/llama3_3B.yaml \
  evaluation.model_path=models/my_custom_model/checkpoint-100 \
  evaluation.output_file=outputs/my_eval_results.csv

# Override dataset subset and generation parameters
fai-rl-eval --config recipes/evaluation/mmlu/llama3_3B.yaml \
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
- **Splits**: `test`, `validation`, `dev`
- **Subsets**: 57 subjects (e.g., `abstract_algebra`, `college_biology`, `high_school_physics`)
- **Evaluation**: Automatic JSON answer extraction and accuracy calculation
- **Example Config**: `recipes/evaluation/mmlu/llama3_3B.yaml`
