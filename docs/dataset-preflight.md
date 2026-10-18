# Dataset preflight

## Debugging Training Failures

Training preflight loads each raw dataset before trainer construction and checks
required columns, emptiness, missing values across all rows, rendered text
usability for templated datasets, and up to 32 chat samples per dataset.
The trainer reuses these raw datasets. A parent
distributed launcher and its workers each perform preflight; normal HuggingFace
dataset caching still applies. For large datasets, the full missing-value scan
adds CPU startup time.

Chat samples are rendered with a tokenizer/processor before model weights load.
S3-backed models use the existing model download first. VLM checks cover text,
roles, and media placeholders; fetching/decoding images and video and actual
collator execution still happen in the training path. Sample contents are not
logged. Sampling does not guarantee that every later row is well formatted.


## Common Problems & Fixes

- Missing columns: fix `data.datasets` source mappings using the expected/found list.
- DPO requires chosen/rejected columns; a missing prompt remains supported.
- GRPO/GSPO require a usable prompt but no ground-truth answer column.
- Missing values are summarized; the existing trainer row filtering is preserved.
- Match prompt placeholders to `dataset_columns`; escape literal braces as `{{` and `}}`.
- Check model access and supported chat roles/media when a chat template fails.

```bash
python -m pytest -q tests/test_dataset_validation.py
```

Tests use in-memory rows and mocked tokenizers. Project dependencies are needed
for the existing package imports, but no GPU, token, or model download is needed.
