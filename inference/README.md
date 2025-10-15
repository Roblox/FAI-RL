# FAI-RL Inference

High-performance inference system for generating text completions from trained language models. Supports batch processing, custom prompts, and flexible configuration.

## 🚀 Quick Start

### Basic Inference

```bash
# Run inference on a trained model
fai-rl-inference --config configs/inference/llama3_3B_recipe.yaml

# Run inference with debug mode for detailed logging
fai-rl-inference --config configs/inference/llama3_3B_recipe.yaml --debug
```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model path and output file
fai-rl-inference --config configs/inference/llama3_3B_recipe.yaml \
  inference.model_path=models/my_custom_model/checkpoint-100 \
  inference.output_file=outputs/my_inference_results.csv

# Override generation parameters
fai-rl-inference --config configs/inference/llama3_3B_recipe.yaml \
  inference.temperature=0.7 \
  inference.max_new_tokens=512 \
  inference.do_sample=false
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

### Output Files

Inference generates a CSV file at the specified `output_file` path:

```
outputs/
└── llama3_3B_Inst_SFT_lora_v1_checkpoint100_inference.csv
```

### Output Format

The CSV file contains the following columns:
- **Input columns**: All columns specified in `dataset_columns` (e.g., `persona`, `prompt`)
- **`generated_text`**: The model's generated response
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