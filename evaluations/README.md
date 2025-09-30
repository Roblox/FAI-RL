# FAI-RL Evaluation

Comprehensive model evaluation system that leverages the inference pipeline to assess model performance on various benchmarks and datasets.

## 🚀 Quick Start

**Run evaluation**:
```bash
# Basic evaluation
./scripts/run_evaluation.sh --config configs/evaluation/mmlu/llama3_3B_recipe.yaml

# With debug logging
./scripts/run_evaluation.sh --config configs/evaluation/mmlu/llama3_3B_recipe.yaml --debug
```

## 🔧 Configuration

Create evaluation configs in `../configs/evaluation/`:

```yaml
evaluation:
  model_path: "your-local-model-path"        # Path to the trained model to evaluate
  output_file: "your-output.csv"             # Where to save evaluation results
  
  # Dataset configuration
  # Specifies which dataset and subset to evaluate on
  dataset_name: "cais/mmlu"                  # HuggingFace dataset identifier
  dataset_subset: "college_biology"          # Specific subset of the dataset (optional)
  output_type: "multiple_choice"             # Type of evaluation task ("multiple_choice" supported)
  dataset_split: "test"                      # Which split to evaluate on (test/validation/dev)
  dataset_columns: ["question", "choices", "answer"]  # List of dataset columns to include in evaluation
  ground_truth_column: "answer"              # Column containing the correct answers
  
  # System prompt template with placeholders
  # Template for evaluation prompts (supports variable substitution with {variable})
  system_prompt: |
    Question: {question}
    Choose the best option and respond only with the letter of your choice.
    
    {choices}
    
    Please respond **only in valid JSON format** with the following keys:
    {{
      "answer": "<the letter of the chosen option, e.g., A, B, C, D>"
    }}
    
    Let's think step by step.
  
  # Generation parameters
  # Controls how the model generates responses during evaluation
  temperature: 1.0                           # Sampling temperature for response generation (higher = more random)
  top_p: 0.9                                # Nucleus sampling parameter (probability threshold for token selection)
  max_new_tokens: 100                       # Maximum tokens to generate per response
  do_sample: true                           # Whether to use sampling for generation (false = greedy decoding)
```

## 📊 Output

**Results CSV**: Detailed per-example results with predictions, ground truth, and correctness
**Metrics Summary**: Overall accuracy, valid prediction accuracy, and extraction success rates
**Evaluation Log**: Detailed logging of the evaluation process

## 🔬 Supported Benchmarks

### MMLU (Massive Multitask Language Understanding)
- **Dataset**: `cais/mmlu`
- **Task Type**: Multiple choice questions across 57 academic subjects
- **Splits**: test, validation, dev
- **Evaluation**: Automatic answer extraction and accuracy calculation
