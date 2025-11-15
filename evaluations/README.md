# FAI-RL Evaluation

Comprehensive model evaluation system that leverages the inference pipeline to assess language model performance on academic benchmarks. Supports automatic answer extraction using sophisticated regex patterns, accuracy calculation with detailed metrics, and per-example correctness analysis. Works with local fine-tuned models, vanilla HuggingFace models, and API-based inference.

## 🚀 Quick Start

### Basic Evaluation

```bash
# Evaluate on MMLU benchmark
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml

# Evaluate on GSM8K benchmark (math word problems)
fai-rl-eval --recipe recipes/evaluation/gsm8k/qwen3_4B.yaml

# Evaluate multiple checkpoints (batch evaluation)
fai-rl-eval --recipe recipes/evaluation/gsm8k/qwen3_4B_multi_ckpt.yaml

# Evaluate with debug mode for detailed logging
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml --debug

# Run evaluation in background with nohup
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml --nohup
```

> **Running with Local Code**: If running directly from the repository, use `python evaluations/eval.py` instead of `fai-rl-eval`:
> ```bash
> python evaluations/eval.py --recipe recipes/evaluation/mmlu/llama3_3B.yaml
> ```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model paths and output file
fai-rl-eval --recipe recipes/evaluation/mmlu/llama3_3B.yaml \
  'evaluation.model_paths=["models/my_custom_model/checkpoint-100"]' \
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

### Multi-Checkpoint Evaluation

When evaluating multiple checkpoints, the system:
- Runs inference on all checkpoints sequentially
- Generates a single CSV with a `checkpoint` column identifying the source checkpoint
- Calculates separate accuracy metrics for each checkpoint
- Prints a summary comparison of all checkpoints

**Example Output:**
```
EVALUATION SUMMARY (All Checkpoints):
============================================================
models/checkpoint-100:
  Accuracy: 0.7543
models/checkpoint-200:
  Accuracy: 0.7821
models/checkpoint-300:
  Accuracy: 0.7965
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