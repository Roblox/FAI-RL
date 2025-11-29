# FAI-RL Inference

High-performance inference system for generating text completions from language models. Supports three inference modes: local fine-tuned models, vanilla HuggingFace models, and API-based inference. Features include automatic PEFT/LoRA checkpoint handling, template-based prompts with variable substitution, and flexible configuration.

## 🚀 Quick Start

### Basic Inference

```bash
# Run inference on a local fine-tuned model (including PEFT/LoRA checkpoints)
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml

# Run inference on multiple checkpoints (batch inference)
fai-rl-inference --recipe recipes/inference/llama3_3B_multi_ckpt.yaml

# Run inference on a vanilla HuggingFace model
fai-rl-inference --recipe recipes/inference/llama3_vanilla_3B.yaml

# Run inference using an API endpoint (OpenAI, hosted LLM, etc.)
fai-rl-inference --recipe recipes/inference/llama3_3B_api.yaml

# Run inference with a local CSV file as the dataset
fai-rl-inference --recipe recipes/inference/llama3_3B_local_csv.yaml

# Run inference with debug mode for detailed logging
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml --debug

# Run inference in background with nohup
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml --nohup
```

> **Running with Local Code**: If running directly from the repository, use `python inference/inference.py` instead of `fai-rl-inference`:
> ```bash
> python inference/inference.py --recipe recipes/inference/llama3_3B.yaml
> ```

### Runtime Parameter Overrides

Override configuration parameters directly from command line:

```bash
# Override model paths and output file
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml \
  'inference.model_paths=["models/my_custom_model/checkpoint-100"]' \
  inference.output_file=outputs/your-output.csv

# Override generation parameters
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml \
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
- **Checkpoint column** (multi-checkpoint only): Identifies which checkpoint generated each response (column name specified by `checkpoint_column`, default is `checkpoint`)
- **Response column**: The model's generated response (column name specified by `response_column`, default is `response`)
- **Metadata**: Generation parameters used (temperature, top_p, max_new_tokens)

### Multi-Checkpoint Inference

When running inference on multiple checkpoints, all results are combined into a single CSV file with an additional `checkpoint` column:

```csv
persona,prompt,checkpoint,response
"helpful assistant","What is AI?","models/checkpoint-100","AI is artificial intelligence..."
"helpful assistant","What is AI?","models/checkpoint-200","AI stands for artificial..."
"helpful assistant","What is AI?","models/checkpoint-300","Artificial Intelligence is..."
```

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