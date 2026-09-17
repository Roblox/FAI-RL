"""Static memory estimates use fake parameters; no GPU or model download is needed."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utils.debug_callback import DebugCallback


@pytest.mark.parametrize("partitioned", [False, True])
def test_estimate_counts_frozen_weights_and_only_trainable_states(partitioned):
    count = 2**28
    trainable = SimpleNamespace(
        numel=lambda: 0 if partitioned else count,
        element_size=lambda: 2,
        requires_grad=True,
    )
    frozen = SimpleNamespace(
        numel=lambda: 0 if partitioned else count,
        element_size=lambda: 1,
        requires_grad=False,
    )
    if partitioned:
        trainable.ds_numel = frozen.ds_numel = count
    model = SimpleNamespace(parameters=lambda: iter([trainable, frozen]))
    logger = Mock()

    DebugCallback(logger)._log_static_memory(model)

    logger.info.assert_called_once()
    line = logger.info.call_args.args[0]
    # 0.5 GiB trainable weights + 0.25 GiB frozen storage + 3 GiB training states.
    assert "Estimated static training memory: 3.75 GiB" in line
    assert "unsharded" in line and "excludes activations" in line


def test_estimate_is_emitted_once_at_training_start():
    callback = DebugCallback(Mock())
    callback._log_static_memory = Mock()
    for name in (
        "_log_distributed",
        "_log_arch",
        "_log_tokenizer",
        "_log_optimizer",
        "_log_first_batches",
    ):
        setattr(callback, name, Mock())
    model = object()

    callback.on_train_begin(SimpleNamespace(logging_steps=10), None, None, model=model)

    callback._log_static_memory.assert_called_once_with(model)


def test_unavailable_parameter_info_does_not_interrupt_training():
    logger = Mock()
    model = SimpleNamespace(parameters=Mock(side_effect=RuntimeError("unavailable")))
    DebugCallback(logger)._log_static_memory(model)
    assert "static memory estimate unavailable" in logger.info.call_args.args[0]


def test_no_model_skips_estimate():
    logger = Mock()
    DebugCallback(logger)._log_static_memory(None)
    logger.info.assert_not_called()
