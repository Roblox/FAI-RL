# Training Recipes

YAML configuration files for all supported training algorithms. Each recipe is a self-contained, fully-commented template — copy one, fill in your model/dataset/output paths, and launch.

📖 **[Full training guide →](../../trainers/README.md)**

## Recipe Index

### CPT (Continuous Pre-Training)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `cpt/qwen3_4B_lora.yaml` | HuggingFace Hub | LoRA, Qwen3-4B |
| `cpt/qwen3_4B_qlora.yaml` | HuggingFace Hub | QLoRA (4-bit), Qwen3-4B |
| `cpt/qwen3_30B_A3B_lora.yaml` | HuggingFace Hub | LoRA, Qwen3-30B-A3B MoE |
| `cpt/qwen3_30B_A3B_qlora.yaml` | HuggingFace Hub | QLoRA (4-bit), Qwen3-30B-A3B MoE |
| `cpt/qwen3_4B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `cpt/qwen3_4B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

### SFT (Supervised Fine-Tuning)

| Recipe | Dataset source | Model source | Notes |
|--------|---------------|--------------|-------|
| `sft/llama3_3B_lora.yaml` | HuggingFace Hub | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `sft/llama3_3B_qlora.yaml` | HuggingFace Hub | HuggingFace Hub | QLoRA (4-bit), Llama-3.2-3B |
| `sft/llama3_3B_full.yaml` | HuggingFace Hub | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |
| `sft/llama3_3B_local_file.yaml` | Local file | HuggingFace Hub | `.jsonl/.json/.csv/.parquet` |
| `sft/llama3_3B_s3_file.yaml` | S3 | HuggingFace Hub | `s3://bucket/key.jsonl` |
| `sft/llama3_3B_s3_model.yaml` | S3 | S3 | Continue training from fine-tuned model in S3 |

### SFT_VLM (Multimodal Supervised Fine-Tuning — image + text)

| Recipe | Dataset source | Model source | Notes |
|--------|---------------|--------------|-------|
| `sft_vlm/qwen2_5_vl_3b_lora.yaml` | Local file (image URLs) | HuggingFace Hub | LoRA, Qwen2.5-VL-3B — small, for validating the pipeline |
| `sft_vlm/qwen2_5_vl_3b_lora_image_bytes.yaml` | Local file (local image paths / raw bytes) | HuggingFace Hub | LoRA, Qwen2.5-VL-3B — images loaded from local `.bin` byte files |
| `sft_vlm/qwen2_5_vl_3b_parquet.yaml` | Local Parquet (embedded images) | HuggingFace Hub | LoRA, Qwen2.5-VL-3B — HF-style `images: List[Image]` parquet, images decoded in-memory |
| `sft_vlm/qwen2_5_vl_3b_lora_s3_file.yaml` | S3 file (image URLs) | HuggingFace Hub | LoRA, Qwen2.5-VL-3B — dataset file loaded from S3 |
| `sft_vlm/qwen2_5_vl_3b_lora_s3_image_bytes.yaml` | S3 file (`s3://` image URIs) | HuggingFace Hub | LoRA, Qwen2.5-VL-3B — images fetched from S3 |
| `sft_vlm/qwen3_vl_30b_a3b_lora.yaml` | Local file (image URLs) | HuggingFace Hub | LoRA, Qwen3-VL-30B-A3B MoE VLM |
| `sft_vlm/qwen3_vl_30b_a3b_qlora.yaml` | Local file (image URLs) | HuggingFace Hub | QLoRA (4-bit), Qwen3-VL-30B-A3B MoE VLM |
| `sft_vlm/qwen3_vl_30b_a3b_lora_s3_file.yaml` | S3 file (image URLs) | HuggingFace Hub | LoRA, Qwen3-VL-30B-A3B MoE VLM — dataset file loaded from S3 |

Fine-tunes a vision-language model on `(image, text) -> response` data, and supports **multiple images per row**. Images can be supplied as **HTTP(S) URLs**, local paths, `s3://` URIs, raw bytes, or **embedded in a Parquet file** (HuggingFace `images: List[Image]` schema) — all fetched/decoded into PIL images at data-loading time. See [Multimodal datasets](#multimodal-datasets-sft_vlm) below.

### DPO (Direct Preference Optimization)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `dpo/llama3_3B_lora.yaml` | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `dpo/llama3_3B_qlora.yaml` | HuggingFace Hub | QLoRA (4-bit), Llama-3.2-3B |
| `dpo/llama3_3B_full.yaml` | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |
| `dpo/llama3_3B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `dpo/llama3_3B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

### GRPO (Group Relative Policy Optimization)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `grpo/llama3_3B_lora.yaml` | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `grpo/llama3_3B_full.yaml` | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |
| `grpo/llama3_3B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `grpo/llama3_3B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

### GSPO (Group Sequence Policy Optimization)

| Recipe | Dataset source | Notes |
|--------|---------------|-------|
| `gspo/llama3_3B_lora.yaml` | HuggingFace Hub | LoRA, Llama-3.2-3B |
| `gspo/llama3_3B_full.yaml` | HuggingFace Hub | Full fine-tune, Llama-3.2-3B |
| `gspo/llama3_3B_local_file.yaml` | Local file | `.jsonl/.json/.csv/.parquet` |
| `gspo/llama3_3B_s3_file.yaml` | S3 | `s3://bucket/key.jsonl` |

## Model Sources

`model.base_model_name` accepts three formats. The source is selected automatically by its prefix:

| Source | `base_model_name` value | Example |
|--------|------------------------|---------|
| HuggingFace Hub | Hub model ID | `"meta-llama/Llama-3.2-3B-Instruct"` |
| Local directory | Absolute or relative path | `"/checkpoints/my-model"` |
| S3 | `s3://` URI pointing to a model directory | `"s3://my-bucket/checkpoints/run-v1/final"` |

When an `s3://` URI is given, the trainer downloads the model directory to a local cache before calling `from_pretrained()`. The cache is keyed on the URI so repeated runs on the same node skip the download.

## Dataset Sources

All algorithms support three dataset sources. The source is selected by the value of `data.datasets[n].name`:

| Source | `name` value | Example |
|--------|-------------|---------|
| HuggingFace Hub | Dataset name (no extension) | `"openai/gsm8k"` |
| Local file | File path ending in `.jsonl`, `.json`, `.csv`, or `.parquet` | `"data/train.jsonl"` |
| S3 | `s3://` URI ending in a supported extension | `"s3://my-bucket/datasets/train.jsonl"` |

### Required JSONL fields by algorithm

| Algorithm | Required fields |
|-----------|----------------|
| CPT | `text` |
| SFT | `prompt`, `response` |
| DPO | `prompt`, `chosen`, `rejected` |
| GRPO | `prompt`, `answer` |
| GSPO | `prompt`, `answer` |
| SFT_VLM | one or more image columns + one or more text columns (see below) |

Column names are configurable — use `text_column`, `prompt_column`, `answer_column`, `chosen_column`, `rejected_column` in the dataset entry to remap them. For **SFT_VLM**, images and text columns are named explicitly via `image_columns` and `dataset_columns` (see [Multimodal datasets](#multimodal-datasets-sft_vlm)).

### Multimodal datasets (SFT_VLM)

The `sft_vlm` algorithm trains vision-language models on image + text. Each dataset row needs at least one image column plus one or more text columns. Columns are named via **`image_columns`** (a list) and **`dataset_columns`** (a list); a row can carry **multiple images**. The shipped example is a CSV (any of `.csv` / `.jsonl` / `.json` / `.parquet`, HF Hub, or S3 works):

```csv
image,question,response
https://example.com/cat.jpg,What is in this image?,A cat sitting on a couch.
```

Configure the columns in the dataset entry, and the image-fetch behavior under `data`:

```yaml
data:
  datasets:
    - name: "data/sft_vlm/example_training_image_url.csv"   # HF Hub / local / S3
      image_columns: ["image"]              # one or more columns; each cell may hold an
                                            # HTTP(S) URL, local path, s3:// URI, raw bytes,
                                            # an embedded PIL image, or a list of any of these.
                                            # List several columns for multi-image rows.
      dataset_columns: ["question", "response"]  # text columns that fill the system_prompt template
  image_cache_dir: "data/vlm_image_cache"   # cache downloads so they aren't re-fetched each epoch
  image_fetch_timeout: 15
  image_fetch_retries: 3
  max_image_pixels: 1048576          # optional: downscale large images to bound vision tokens / memory
  image_s3_region: null              # override AWS region when an image value is an s3:// URI
  image_s3_endpoint_url: null        # set for MinIO or other S3-compatible stores
  # system_prompt is a .format() template (like the text SFT recipes), keyed by the
  # dataset_columns names: {question} and {response} are filled per row. {response}
  # leaks the target, so use it only for labeling/eval-style data. When no
  # system_prompt is set, the dataset_columns values are concatenated instead.
  system_prompt: |
    You are a helpful multimodal assistant. Answer the user's question based on the image.
    Question: {question}
    Response: {response}
```

Notes:
- `system_prompt` is a `str.format()` template whose placeholders are the `dataset_columns` names (e.g. `{question}`, `{response}`), mirroring the text-SFT templating above. Unknown placeholders fall back to the literal text. The rendered text is a single training turn (no prompt/completion masking).
- **Multiple images per row:** list more than one column in `image_columns` (e.g. `["image_a", "image_b"]`), or put a list of image sources in a single cell. Every image found across the listed columns (in order) is attached to the row.
- **Image sources:** each cell may be an HTTP(S) URL, a local path, an `s3://` URI, raw image bytes, or an embedded PIL image. Parquet files using the HuggingFace `images: List[Image]` schema (embedded PNG bytes) are decoded in-memory — no fetch happens. See `sft_vlm/qwen2_5_vl_3b_parquet.yaml`.
- Images are fetched/decoded at startup; rows whose images can't be fetched/decoded (or that render empty text) are dropped, with a logged count.
- Example datasets ship under `data/sft_vlm/`: `example_training_image_url.csv` (public image URLs), `example_training_data.csv` (local image paths), `example_training_data_bytes.csv` (local byte files), and `example_training_data.parquet` (embedded images). Any HF/local/S3 dataset with an image column works — just point `name` at it.
- `data.max_length` is forced to `None` internally for VLMs so image placeholder tokens are never truncated.
- **Network:** the training environment must be able to reach the image hosts **and** the HuggingFace Hub (for the model). If image hosts are blocked, pre-download images and use local paths (or an embedded-image parquet) in `image_columns`.

## Quick Start

```bash
# Pick a recipe and launch
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 1

# Override dataset and output dir at the command line
fai-rl-train --recipe recipes/training/sft/llama3_3B_lora.yaml --num-gpus 4 \
  data.datasets[0].name=your-org/your-dataset \
  training.output_dir=models/my_model

# Use a local file
fai-rl-train --recipe recipes/training/sft/llama3_3B_local_file.yaml --num-gpus 1 \
  data.datasets[0].name=data/train.jsonl

# Use an S3 file
fai-rl-train --recipe recipes/training/sft/llama3_3B_s3_file.yaml --num-gpus 1 \
  data.datasets[0].name=s3://my-bucket/datasets/train.jsonl

# Continue training from a fine-tuned model stored in S3
fai-rl-train --recipe recipes/training/sft/llama3_3B_s3_model.yaml --num-gpus 4 \
  model.base_model_name=s3://my-bucket/checkpoints/run-v1/final \
  data.datasets[0].name=s3://my-bucket/datasets/train-v2.jsonl \
  training.output_dir=models/my_model_v2
```

> Running from the repo directly: replace `fai-rl-train` with `python trainers/train.py`.

## S3 Model Loading

Set `base_model_name` to an `s3://` URI that points to a model directory — typically the `final/` prefix written by the S3 upload callback after a previous training run. AWS credentials are resolved from the standard boto3 chain (IAM role, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars, `~/.aws/credentials`).

```yaml
model:
  base_model_name: "s3://my-bucket/checkpoints/run-v1/final"
  s3_region: null        # override AWS region if needed
  s3_endpoint_url: null  # set for MinIO or other S3-compatible stores
```

**Download behavior:**
- On each node, local rank 0 downloads the model directory to `/tmp/fai-rl-model-<hash>/` and writes a sentinel file when done. Non-zero local ranks wait on the sentinel, then all ranks load from the shared local path.
- The cache is keyed on the S3 URI, so subsequent runs on the same node skip the download.
- Both s5cmd (fast, ~1.25 GiB/s) and boto3 (fallback) are supported — s5cmd is used automatically if it is on `$PATH`.

**Continuing from a LoRA checkpoint:**

The S3 callback saves LoRA adapter files (`adapter_config.json`, `adapter_model.safetensors`) when `save_only_model: true`. These can be loaded as `base_model_name` for the next training round — HuggingFace PEFT merges the adapter automatically on load.

## S3 Dataset Loading

Set `name` to an `s3://` URI. AWS credentials are resolved from the standard boto3 chain (IAM role, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars, `~/.aws/credentials`).

```yaml
data:
  datasets:
    - name: "s3://my-bucket/datasets/train.jsonl"
      prompt_column: "prompt"
      dataset_columns: ["prompt", "response"]
      s3_region: null        # override AWS region if needed
      s3_endpoint_url: null  # set for MinIO or other S3-compatible stores
```

The file is downloaded to a temporary path at training startup, loaded into the HuggingFace Arrow cache, and then deleted. Multiple S3 datasets (or a mix of S3, local, and Hub) can be listed and will be concatenated before training.

## Naming Convention

```
<algorithm>/<model>_<variant>.yaml

Variants:
  lora          LoRA fine-tuning (recommended starting point)
  qlora         QLoRA (4-bit quantization + LoRA)
  full          Full parameter fine-tuning
  local_file    Local dataset file (.jsonl / .json / .csv / .parquet)
  s3_file       S3 dataset file (s3://bucket/key.ext)
  s3_model      S3 base model + S3 dataset (continue from fine-tuned checkpoint)
```
