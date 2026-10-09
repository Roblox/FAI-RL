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

# Run inference with a local file as the dataset (.jsonl / .json / .csv / .parquet)
fai-rl-inference --recipe recipes/inference/llama3_3B_local_csv.yaml

# Run inference with debug mode for detailed logging
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml --debug

# Run inference in background with nohup
fai-rl-inference --recipe recipes/inference/llama3_3B.yaml --nohup

# Run multimodal (image + text) inference on a fine-tuned VLM checkpoint
CUDA_VISIBLE_DEVICES=0 fai-rl-inference --recipe recipes/inference/qwen2_5_vl_3b.yaml

# Run data-parallel inference with one model replica per GPU
fai-rl-inference --recipe recipes/inference/qwen3_vl_30b_a3b.yaml --num-gpus 8
```

### Multi-GPU Inference

Use `--num-gpus N` when the model fits on one GPU and you want higher dataset
throughput. The launcher starts `N` torchrun workers, pins one complete model
replica to each GPU, and dynamically claims dataset rows from a shared atomic
work queue so faster workers never sit idle while slower workers finish
variable-latency generations. Temporary rank outputs are merged by rank 0 into
the configured `output_file` in exact original row order. The job must allocate
and expose at least `N` CUDA devices.

Workers claim 1 row at a time by default. Set `FAI_RL_INFERENCE_CHUNK_SIZE` to a
positive integer to claim larger batches if desired.

Ranks publish their temporary outputs atomically and may finish at different
times; rank 0 waits for every completion marker before merging. Process-group
setup and this result rendezvous both default to a one-hour timeout. Set
`FAI_RL_DISTRIBUTED_TIMEOUT_SECONDS` to a positive number of seconds to override
that limit for unusually long or variable-latency workloads.

This is data parallelism, not tensor parallelism: each GPU must have enough
memory for the complete model. Without `--num-gpus`, inference remains a
single-process run and `device_map="auto"` may shard a model across visible GPUs
only when needed for model capacity.

### Multimodal (VLM) Inference

To run inference on a vision-language model fine-tuned with the `sft_vlm` algorithm, set **`image_columns`** in the recipe. Its presence switches inference into VLM mode: for each row, the image URL/path in those columns is fetched into a PIL image and fed to the model alongside the templated text prompt. See `recipes/inference/qwen2_5_vl_3b.yaml`.

Key VLM recipe fields (under `inference:`):
- `image_columns` — list of dataset columns, each holding an image URL / `s3://` URI / local path (or a list of them). Every image found across these columns (in order) is fed to the model, so a row can carry **multiple images**. **Required to enable VLM mode.** Mirrors the `sft_vlm` trainer's `image_columns`.
- `image_cache_dir`, `image_fetch_timeout`, `image_fetch_retries`, `max_image_pixels` — image-fetch settings (mirror the training recipe).
- `system_prompt` — prompt template (filled per row from `dataset_columns`); becomes the user text shown with the image(s).

For multiple images per row, list more than one column (e.g. `image_columns: ["image_a", "image_b"]`) — the processor receives one image placeholder per fetched image, in column order. See `recipes/inference/qwen2_5_vl_3b_multi_image.yaml`.

VLM mode loads the model as `AutoModelForImageTextToText` + `AutoProcessor`, automatically detects and merges PEFT/LoRA adapters, and supports the same multi-checkpoint, CSV-output, and S3-upload workflow as text models. It is **local-model only** — API endpoints are not supported for VLMs.

### Chat (Split) Mode

By default, `system_prompt` is a single **flat** template that becomes the entire prompt fed to the model. Setting **`user_prompt`** switches inference into **chat mode**, structuring each row as proper conversation turns instead:

- `system_prompt` (optional) → a **system-role** turn.
- `user_prompt` (required for chat mode) → a **user-role** turn.
- The model generates the **assistant** turn — so, unlike the `sft` trainer's split mode, there is **no `assistant_prompt`**.

Both are `str.format()` templates keyed by `dataset_columns` (e.g. `user_prompt: "{prompt}"`), exactly like the flat `system_prompt`. In chat mode the model's chat template is applied with `add_generation_prompt=True`, so system/user roles are honored (rather than concatenated as raw text). This works across all three paths: local text, VLM (system turn prepended before the image+text user turn), and API (OpenAI/default → `system` message; Anthropic → top-level `system` field; Gemini → `system_instruction`).

Leave `user_prompt` unset to keep the legacy flat `system_prompt` behavior — existing recipes are unaffected. See `recipes/inference/llama3_3B_chat.yaml`.

### Dataset sources

Inference uses the same dataset loader as training. Set `inference.dataset_name` to a HuggingFace Hub id, a local file path, or an `s3://` URI — the file extension selects the loader.

**Supported file formats**

| Extension | Format |
|-----------|--------|
| `.jsonl` | Newline-delimited JSON (recommended) |
| `.json` | JSON array |
| `.csv` | Comma-separated values |
| `.parquet` | Apache Parquet |

Relative paths are resolved from the directory where `fai-rl-inference` is launched. A missing local file raises `FileNotFoundError` (it is not treated as a Hub id). Hub datasets still use `dataset_name` plus `dataset_split` (and optional `dataset_subset`).

```yaml
inference:
  dataset_name: "data/eval.jsonl"   # or s3://bucket/eval.jsonl, or org/hub-dataset
  dataset_split: "test"             # used for Hub ids; ignored for local/S3 files
  dataset_columns: ["question", "response"]
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

Hybrid reasoning models can explicitly enable or disable thinking while
rendering local text or VLM chat prompts:

```yaml
inference:
  user_prompt: "{question}"
  enable_thinking: false
```

Omit `enable_thinking` to preserve the model chat template's default. Templates
that do not implement this option ignore it.

### Structured (JSON) Output

Set `json_schema` (a mapping or a JSON string) to constrain local text and VLM
generation to JSON that matches the schema. Requires `pip install "FAI-RL[structured]"`.

```yaml
inference:
  json_schema: {type: object, properties: {decision: {enum: [allow, block]}}, required: [decision]}
```

Each row gets `__parse_ok` and `__schema_error` columns; rows whose generation fails
are kept and flagged, and only `__parse_ok` rows count as successful in the summary.
Thinking follows `enable_thinking` as it does without a schema; when it is unset,
the chat template's default applies (Qwen3 thinks by default), and flat prompts
without `user_prompt` never think. A thinking model thinks first, the text before
`</think>` goes to a `__reasoning` column, and only the JSON after it is validated.
`enable_thinking: true` needs a `<think>…</think>` model such as Qwen3 and chat
mode; others fail at startup. Thinking has no separate budget, so output that runs out of
`max_new_tokens` (while thinking or in the JSON) is flagged `__parse_ok=False`. API
inference is not supported.
See `recipes/inference/qwen3_4b_json_schema.yaml`.

Instead of `json_schema`, set `json_schema_from_column: <column>` to infer the schema at
startup from JSON objects in that dataset column (for example, reference outputs). The
first 200 non-empty cells are sampled (```json fences are stripped) and at least half must
be JSON objects. Keys found in every row are `required`, values seen as both integers and
decimals become `number`, mixed types become `anyOf`, and strings with at most 20 distinct
values that repeat on average become an `enum`. The inferred schema is printed and saved
in the summary JSON, so you can copy it into `json_schema` and adjust it.

## 📊 Output

### Output Files

Inference generates a CSV file at the specified `output_file` path:

```
outputs/
└── llama3_3B_Inst_SFT_lora_v1_checkpoint100_inference.csv
```

### Output Format

Generated columns start with `__` so they never overwrite a dataset column of the same name (before 0.2.24 they were `response` and `confidence`).

The CSV file contains the following columns:
- **Input columns**: All columns specified in `dataset_columns` (e.g., `persona`, `prompt`)
- **Checkpoint column** (multi-checkpoint only): Identifies which checkpoint generated each response (column name specified by `checkpoint_column`, default is `checkpoint`)
- **Response column**: The model's generated response (column name specified by `response_column`, default is `__response`)
- **Confidence column**: Geometric mean of the generated-token probabilities, from 0 to 1 (column name specified by `confidence_column`, default is `__confidence`). API inference leaves this value blank when the endpoint does not return token probabilities.
- **`__parse_ok` / `__schema_error` / `__reasoning`** (only with `json_schema`): see [Structured (JSON) Output](#structured-json-output)
- **Metadata**: Generation parameters used (temperature, top_p, max_new_tokens)

### Multi-Checkpoint Inference

When running inference on multiple checkpoints, all results are combined into a single CSV file with an additional `checkpoint` column:

```csv
persona,prompt,checkpoint,__response,__confidence
"helpful assistant","What is AI?","models/checkpoint-100","AI is artificial intelligence...",0.87
"helpful assistant","What is AI?","models/checkpoint-200","AI stands for artificial...",0.81
"helpful assistant","What is AI?","models/checkpoint-300","Artificial Intelligence is...",0.79
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