# FAI-RL Inference

High-performance inference system for generating text completions from trained language models. Supports batch processing, custom prompts, and flexible configuration.

## 🚀 Quick Start

### Basic Inference

```bash
# Run inference on a local fine-tuned model
fai-rl-inference --config configs/inference/llama3_3B_recipe.yaml

# Run inference on a vanilla HuggingFace model
fai-rl-inference --config configs/inference/llama3_vanilla_3B_recipe.yaml

# Run inference with debug mode for detailed logging
fai-rl-inference --config configs/inference/llama3_3B_recipe.yaml --debug
```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model path and output file
fai-rl-inference --config configs/inference/llama3_3B_recipe.yaml \
  inference.model_path=models/my_custom_model/checkpoint-100 \
  inference.output_file=outputs/your-output.csv

# Override generation parameters
fai-rl-inference --config configs/inference/llama3_3B_recipe.yaml \
  inference.temperature=0.7 \
  inference.max_new_tokens=512 \
  inference.do_sample=false
```

## 🔧 Configuration

Create inference configs in `../configs/inference/`:

### Three Inference Modes

**1. Local Fine-tuned Model (using `model_path`):**
```yaml
inference:
  model_path: "models/your-local-model-path"        # Path to local fine-tuned model checkpoint
  output_file: "your-output.csv"
  # ... rest of config
```

**2. Vanilla HuggingFace Model (using `model` without `api_key`):**
```yaml
inference:
  model: "meta-llama/Llama-3.2-3B-Instruct"         # HuggingFace model identifier
  output_file: "your-output.csv"
  # ... rest of config
```

**3. API-based Inference (using `model` + `api_key`):**
```yaml
inference:
  model: "openai/gpt-4"                             # API model identifier
  api_key: "your-api-key"                           # API authentication key
  api_endpoint: "https://<YOUR_API_ENDPOINT>"       # custom endpoint
  output_file: "your-output.csv"
  # ... rest of config
```

### Full Configuration Example

```yaml
# Inference Configuration
# Defines model source and inference settings
inference:
  # Model Configuration - Choose ONE of the following:
  # Option A: model_path (for local fine-tuned models)
  # Option B: model (for vanilla HuggingFace models)
  # Option C: model + api_key (for API-based inference)
  
  model_path: "models/your-local-model-path"        # OR
  model: "meta-llama/Llama-3.2-3B-Instruct"         # OR
  api_key: "your-api-key"                           # (Only for API inference)
  
  output_file: "your-output.csv"                    # Path to save inference results (CSV format)

  # Dataset Configuration
  # Specifies which dataset to run inference on
  dataset_name: "your-huggingface-dataset"          # HuggingFace dataset identifier (e.g., "Anthropic/hh-rlhf")
  dataset_split: "test"                             # Dataset split to use: train, test, validation
  dataset_columns: ["persona", "prompt"]            # List of columns to concatenate as model input
  response_column: "response"                       # Name of column to store model responses (default: "response")

  # System Prompt
  # Provides context and instructions to the model
  system_prompt: |
    your inference prompt...                        # Multi-line system message for generation context
  
  # Generation Parameters
  # Controls the randomness and quality of generated text
  temperature: 1.0                                  # Sampling temperature (0.0 = deterministic, 2.0 = very random)
  top_p: 0.9                                        # Nucleus sampling threshold (0.0-1.0, lower = more focused)
  max_new_tokens: 1000                              # Maximum number of tokens to generate per response
  do_sample: true                                   # Enable sampling (false = greedy decoding, true = stochastic sampling)
```

### Configuration Parameters

**Model Selection Tips:**
- **Use `model_path`** for local fine-tuned models you've trained with this framework
- **Use `model`** (without `api_key`) to test vanilla HuggingFace models before fine-tuning
- **Use `model` + `api_key`** for API-based inference with commercial models

**Generation Tips:**
- **For consistent results**: Set `temperature: 0.0` and `do_sample: false`
- **For creative generation**: Use `temperature: 0.8-1.2` with `top_p: 0.9`
- **Memory considerations**: Reduce `max_new_tokens` if encountering memory issues
- **Prompt engineering**: Use `system_prompt` to improve response quality

## 📊 Output

### Output Files

Inference generates a CSV file at the specified `output_file` path:

```
outputs/
└── llama3_3B_Inst_SFT_lora_v1_checkpoint100_inference.csv
```

### Output Format

The CSV file contains the following columns:
- **Input columns**: All columns specified in `dataset_columns` (e.g., `persona`, `prompt`)
- **Response column**: The model's generated response (column name specified by `response_column`, default is `response`)
- **Metadata**: Generation parameters used (temperature, top_p, max_new_tokens)

## 🐛 Troubleshooting

### Slow Inference
- Reduce `max_new_tokens` if not needed
- Ensure model is loaded on GPU (not CPU)
- Consider using smaller models for faster generation

### Out of Memory
- Reduce batch size (processed internally)
- Use a smaller model
- Reduce `max_new_tokens`

### Poor Quality Outputs
- Adjust `temperature` (try lower values for more focused outputs)
- Refine `system_prompt` to provide better context
- Ensure model is properly trained for the task
- Try different `top_p` values

### Missing Outputs
- Check `output_file` path is writable
- Verify `dataset_columns` match your dataset
- Enable `--debug` flag for detailed error messages