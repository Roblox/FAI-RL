# FAI-RL Inference

High-performance inference system for generating text completions from trained language models. Supports batch processing, custom prompts, and flexible configuration.

## 🚀 Quick Start

### Basic Inference

```bash
# Run inference on a local fine-tuned model
fai-rl-inference --config recipes/inference/llama3_3B.yaml

# Run inference on a vanilla HuggingFace model
fai-rl-inference --config recipes/inference/llama3_vanilla_3B.yaml

# Run inference with debug mode for detailed logging
fai-rl-inference --config recipes/inference/llama3_3B.yaml --debug

# Run inference in background with nohup
fai-rl-inference --config recipes/inference/llama3_3B.yaml --nohup
```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model path and output file
fai-rl-inference --config recipes/inference/llama3_3B.yaml \
  inference.model_path=models/my_custom_model/checkpoint-100 \
  inference.output_file=outputs/your-output.csv

# Override generation parameters
fai-rl-inference --config recipes/inference/llama3_3B.yaml \
  inference.temperature=0.7 \
  inference.max_new_tokens=512 \
  inference.do_sample=false
```

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