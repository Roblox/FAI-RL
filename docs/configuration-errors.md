# Configuration errors

Recipe validation reports the full path of unknown keys, likely replacements,
available fields, missing required fields, and wrong value types. YAML syntax
errors identify the source file and line/column. CLI overrides are checked after
they are applied, without changing precedence or coercing recipe values.

For example, `training.lr` now suggests `training.learning_rate`. Dataset entry
errors use paths such as `data.datasets[0].name`. The training launcher and the
existing YAML config loaders use these checks.

## Common Problems & Fixes

- Check indentation, colons, and quotes when a YAML location is reported.
- Use the suggested full key path and available keys to correct misspellings.
- Use YAML booleans rather than quoted strings for boolean settings.
- Keep the existing `data.datasets` list of mappings format.

```bash
python -m pytest -q tests/test_config_validation.py
```

These tests run on CPU with project dependencies installed and require no model
downloads or credentials. They also validate the shipped training recipes.
