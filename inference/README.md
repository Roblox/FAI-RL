# FAI-RL Inference

Inference system supporting both local model inference.

## 🚀 Quick Start

**Run inference**:
```bash
# Basic inference
./scripts/run_inference.sh --config configs/inference/llama3_3B_recipe.yaml

# Background execution
./scripts/run_inference.sh --config configs/inference/llama3_3B_recipe.yaml --nohup

# With debug logging
./scripts/run_inference.sh --config configs/inference/llama3_3B_recipe.yaml --debug
```

## 🔧 Configuration

Create inference configs in `../configs/inference/`:

```yaml
# Inference Configuration
# Defines model source and inference settings
inference:
  # Model Configuration - Choose ONE of the following options:
  model_path: "models/your-local-model-path"        # Local model path for local inference
  output_file: "your-output.csv"                    # Path to save inference results (CSV format)

  # Dataset Configuration
  # Specifies which dataset to run inference on
  dataset_name: "your-huggingface-dataset"          # HuggingFace dataset identifier (e.g., "Anthropic/hh-rlhf")
  dataset_split: "test"                             # Dataset split to use: train, test, validation
  dataset_columns: ["persona", "prompt"]            # List of columns to concatenate as model input

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

**Configuration Tips:**
- **For consistent results**: Set `temperature: 0.0` and `do_sample: false`
- **For creative generation**: Use `temperature: 0.8-1.2` with `top_p: 0.9`
- **Memory considerations**: Reduce `max_new_tokens` if encountering memory issues
- **Prompt engineering**: Use `system_prompt` to improve response quality

## 📊 Output

**Results file**: Contains all generated responses
**Summary file**: Contains inference statistics and configuration