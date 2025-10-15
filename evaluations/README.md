# FAI-RL Evaluation

Comprehensive model evaluation system for assessing language model performance on academic benchmarks and custom datasets. Supports automatic answer extraction, accuracy calculation, and detailed result analysis.

## 🚀 Quick Start

### Basic Evaluation

```bash
# Evaluate on MMLU benchmark
fai-rl-eval --config configs/evaluation/mmlu/llama3_3B_recipe.yaml

# Evaluate with debug mode for detailed logging
fai-rl-eval --config configs/evaluation/mmlu/llama3_3B_recipe.yaml --debug
```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model path and output file
fai-rl-eval --config configs/evaluation/mmlu/llama3_3B_recipe.yaml \
  evaluation.model_path=models/my_custom_model/checkpoint-100 \
  evaluation.output_file=outputs/my_eval_results.csv

# Override dataset subset and generation parameters
fai-rl-eval --config configs/evaluation/mmlu/llama3_3B_recipe.yaml \
  evaluation.dataset_subset=college_mathematics \
  evaluation.temperature=0.0 \
  evaluation.do_sample=false
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
  response_column: "response"                # Name of column to store/read model responses (default: "response")
  
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
- **Example Config**: `configs/evaluation/mmlu/llama3_3B_recipe.yaml`
