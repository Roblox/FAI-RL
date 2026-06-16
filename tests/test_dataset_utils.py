import importlib.util
from pathlib import Path


def _load_dataset_utils():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "dataset_utils.py"
    spec = importlib.util.spec_from_file_location("dataset_utils", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_utils = _load_dataset_utils()


def test_format_multiple_choice_parses_stringified_list():
    choices = "['red', 'green', 'blue']"

    assert dataset_utils.format_multiple_choice_for_inference(choices) == (
        "A. red\nB. green\nC. blue"
    )


def test_format_multiple_choice_does_not_execute_stringified_input(tmp_path):
    marker = tmp_path / "executed"
    choices = f"[__import__('pathlib').Path(r'{marker}').touch()]"

    formatted = dataset_utils.format_multiple_choice_for_inference(choices)

    assert formatted == f"A. {choices}"
    assert not marker.exists()


def test_format_multiple_choice_splits_comma_separated_string():
    assert dataset_utils.format_multiple_choice_for_inference("red, green, blue") == (
        "A. red\nB. green\nC. blue"
    )
