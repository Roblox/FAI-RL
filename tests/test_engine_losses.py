import pytest
import torch

from engine import losses
from engine.types import AdamParams, Datum, ModelInput


def _datum(length, weights=None, **extra):
    inputs = {"target_tokens": list(range(1, length + 1)), **extra}
    if weights is not None:
        inputs["weights"] = weights
    return Datum(model_input=ModelInput.from_ints(range(length)), loss_fn_inputs=inputs)


def test_datum_converts_inputs_to_tensors_and_keeps_free_form_keys():
    datum = _datum(3, weights=[0, 1, 1], label=1)

    assert datum.loss_fn_inputs["target_tokens"].dtype == torch.long
    assert datum.loss_fn_inputs["weights"].dtype == torch.float32
    assert datum.loss_fn_inputs["label"].item() == 1
    assert datum.model_input.to_ints() == [0, 1, 2]


@pytest.mark.parametrize(
    "inputs, message",
    [
        ({}, "must include 'target_tokens'"),
        ({"target_tokens": [1, 2]}, "first dimension 3"),
        ({"target_tokens": [1, 2, 3], "weights": [1.0]}, "first dimension 3"),
        ({"target_tokens": torch.zeros((3, 2, 2))}, r"\(N,\) or \(N, K\)"),
    ],
)
def test_datum_validates_token_aligned_inputs(inputs, message):
    with pytest.raises(ValueError, match=message):
        Datum(model_input=ModelInput.from_ints([5, 6, 7]), loss_fn_inputs=inputs)


def test_datum_accepts_multi_target_tokens():
    datum = Datum(
        model_input=ModelInput.from_ints([5, 6, 7]),
        loss_fn_inputs={"target_tokens": torch.zeros((3, 2))},
    )
    assert datum.loss_fn_inputs["target_tokens"].shape == (3, 2)


def test_cross_entropy_is_weighted_negative_logprob_sum():
    data = [_datum(3, weights=[0.0, 0.5, 2.0]), _datum(2)]
    logprobs = [
        torch.tensor([-1.0, -2.0, -3.0], requires_grad=True),
        torch.tensor([-0.5, -0.25], requires_grad=True),
    ]

    loss, metrics = losses.cross_entropy(data, logprobs)
    loss.backward()

    expected = (0.5 * 2.0 + 2.0 * 3.0) + (0.5 + 0.25)
    assert loss.item() == pytest.approx(expected)
    assert metrics["loss:sum"] == pytest.approx(expected)
    assert logprobs[0].grad.tolist() == [-0.0, -0.5, -2.0]
    assert logprobs[1].grad.tolist() == [-1.0, -1.0]


def test_cross_entropy_rejects_multi_target_logprobs():
    datum = Datum(
        model_input=ModelInput.from_ints([5, 6]),
        loss_fn_inputs={"target_tokens": torch.zeros((2, 2))},
    )
    with pytest.raises(ValueError, match="forward_backward_custom"):
        losses.cross_entropy([datum], [torch.zeros((2, 2))])


def test_registry_lookup_and_duplicate_names():
    assert "cross_entropy" in losses.available_losses()
    assert losses.get_loss("cross_entropy").separable

    with pytest.raises(ValueError, match="available: .*cross_entropy"):
        losses.get_loss("no_such_loss")
    with pytest.raises(ValueError, match="already registered"):
        losses.register_loss("cross_entropy")(lambda data, logprobs: None)


def test_register_custom_loss(monkeypatch):
    monkeypatch.setattr(losses, "_LOSSES", dict(losses._LOSSES))

    @losses.register_loss("mean_logprob")
    def mean_logprob(data, logprobs):
        value = -torch.cat(logprobs).mean()
        return value, {"loss": float(value)}

    registered = losses.get_loss("mean_logprob")
    assert registered.fn is mean_logprob
    assert not registered.separable


def test_adam_params_defaults_match_hf_training_arguments():
    params = AdamParams()
    assert params.to_dict() == {
        "learning_rate": 1e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 0.0,
    }
