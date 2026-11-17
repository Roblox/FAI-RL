from pathlib import Path

import pytest
import torch
from conftest import make_tiny_lm
from transformers import GPT2LMHeadModel

from engine import AdamParams, Datum, ModelInput, ServiceClient, TrainingClient
from utils.checkpoint_utils import find_latest_checkpoint, is_resumable_checkpoint

TEXTS = [
    "What is two plus two? four",
    "Say hello. hello world",
    "Game A has score 10. Game B has score 4.",
    "Compare game A and game B.",
]


def _sft_data(tokenizer):
    data = []
    for text in TEXTS:
        ids = tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
        prompt = len(ids) // 2
        weights = [0.0] * (prompt - 1) + [1.0] * (len(ids) - prompt)
        data.append(
            Datum(
                model_input=ModelInput.from_ints(ids[:-1]),
                loss_fn_inputs={"target_tokens": ids[1:], "weights": weights},
            )
        )
    return data


def _grads(client):
    return [p.grad.detach().clone() for p in client._trainable]


def _clear_grads(client):
    for parameter in client._trainable:
        parameter.grad = None


def _coupled_loss(data, logprobs):
    """Non-separable: every datum's gradient depends on the whole batch."""
    means = torch.stack([lp.mean() for lp in logprobs])
    loss = (means - means.mean()).square().sum() - means.sum()
    return loss, {"loss": float(loss.detach())}


@pytest.fixture
def client(tiny_lm, tmp_path):
    model, tokenizer = tiny_lm
    return TrainingClient(model, tokenizer, device="cpu", output_dir=str(tmp_path))


def test_forward_logprobs_match_manual_log_softmax(client):
    data = _sft_data(client.tokenizer)

    output = client.forward(data)

    client.model.eval()
    for datum, result in zip(data, output.loss_fn_outputs):
        with torch.no_grad():
            logits = client.model(torch.tensor([datum.model_input.to_ints()])).logits[0]
        targets = datum.loss_fn_inputs["target_tokens"].unsqueeze(-1)
        expected = logits.log_softmax(-1).gather(-1, targets).squeeze(-1)
        assert torch.allclose(result["logprobs"], expected, atol=1e-5)


def test_forward_supports_multi_target_tokens(client):
    ids = client.tokenizer("Game A or game B?")["input_ids"]
    candidates = [5, 9]
    datum = Datum(
        model_input=ModelInput.from_ints(ids),
        loss_fn_inputs={"target_tokens": torch.tensor([candidates] * len(ids))},
    )

    logprobs = client.forward([datum]).loss_fn_outputs[0]["logprobs"]

    with torch.no_grad():
        logits = client.model(torch.tensor([ids])).logits[0]
    assert logprobs.shape == (len(ids), 2)
    assert torch.allclose(logprobs, logits.log_softmax(-1)[:, candidates], atol=1e-5)


def test_micro_batching_does_not_change_logprobs(client):
    data = _sft_data(client.tokenizer)
    whole = client.forward(data).loss_fn_outputs

    client.micro_batch_tokens = 16
    assert len(client._micro_batches(data)) > 1
    chunked = client.forward(data).loss_fn_outputs

    for a, b in zip(whole, chunked):
        assert torch.allclose(a["logprobs"], b["logprobs"], atol=1e-5)


@pytest.mark.parametrize("micro_batch_tokens", [None, 16])
def test_forward_backward_accumulates_until_optim_step(client, micro_batch_tokens):
    client.micro_batch_tokens = micro_batch_tokens
    data = _sft_data(client.tokenizer)

    first = client.forward_backward(data[:2], "cross_entropy")
    second = client.forward_backward(data[2:], "cross_entropy")
    accumulated = _grads(client)
    _clear_grads(client)
    whole = client.forward_backward(data, "cross_entropy")

    assert first.metrics["loss:sum"] + second.metrics["loss:sum"] == pytest.approx(
        whole.metrics["loss:sum"], rel=1e-5
    )
    for a, b in zip(accumulated, _grads(client)):
        assert torch.allclose(a, b, atol=1e-6)

    client.optim_step(AdamParams(learning_rate=1e-3))
    assert all(p.grad is None for p in client._trainable)


def test_custom_loss_two_pass_matches_single_pass(client):
    data = _sft_data(client.tokenizer)

    single = client.forward_backward_custom(data, _coupled_loss)
    single_grads = _grads(client)
    _clear_grads(client)

    client.micro_batch_tokens = 16
    assert len(client._micro_batches(data)) > 1
    two_pass = client.forward_backward_custom(data, _coupled_loss)

    assert two_pass.metrics["loss"] == pytest.approx(single.metrics["loss"], rel=1e-5)
    for a, b in zip(single_grads, _grads(client)):
        assert torch.allclose(a, b, atol=1e-6)


def test_optim_step_reports_grad_norm_and_applies_hyperparameters_each_step(client):
    data = _sft_data(client.tokenizer)
    client.forward_backward(data, "cross_entropy")
    expected_norm = torch.stack([p.grad.norm() for p in client._trainable]).norm()
    before = [p.detach().clone() for p in client._trainable]

    metrics = client.optim_step(AdamParams(learning_rate=1e-3, grad_clip_norm=1.0)).metrics

    assert metrics["grad_norm"] == pytest.approx(float(expected_norm), rel=1e-5)
    assert metrics["learning_rate"] == 1e-3
    assert metrics["step"] == 1
    assert any(not torch.equal(a, p) for a, p in zip(before, client._trainable))

    client.forward_backward(data, "cross_entropy")
    frozen = [p.detach().clone() for p in client._trainable]
    client.optim_step(AdamParams(learning_rate=0.0))
    assert all(torch.equal(a, p) for a, p in zip(frozen, client._trainable))
    assert client.step == 2


def test_save_state_round_trip_reproduces_next_step(client, tmp_path):
    data = _sft_data(client.tokenizer)
    params = AdamParams(learning_rate=1e-2, grad_clip_norm=1.0)
    client.forward_backward(data, "cross_entropy")
    client.optim_step(params)

    path = client.save_state("checkpoint-1")
    assert path == str(tmp_path / "checkpoint-1")
    assert is_resumable_checkpoint(Path(path))
    assert find_latest_checkpoint(str(tmp_path)) == path

    client.forward_backward(data, "cross_entropy")
    client.optim_step(params)
    expected = {k: v.clone() for k, v in client.model.state_dict().items()}

    model, tokenizer = make_tiny_lm(seed=123)
    resumed = TrainingClient(model, tokenizer, device="cpu")
    resumed.load_state(path)
    assert resumed.step == 1
    resumed.forward_backward(data, "cross_entropy")
    resumed.optim_step(params)

    for key, value in resumed.model.state_dict().items():
        assert torch.allclose(value, expected[key], atol=1e-6), key


def test_save_weights_for_sampler_loads_with_from_pretrained(client):
    data = _sft_data(client.tokenizer)
    client.forward_backward(data, "cross_entropy")
    client.optim_step(AdamParams(learning_rate=1e-2))

    path = client.save_weights_for_sampler("sampler")

    reloaded = GPT2LMHeadModel.from_pretrained(path)
    ids = torch.tensor([data[0].model_input.to_ints()])
    client.model.eval()
    with torch.no_grad():
        assert torch.allclose(reloaded(ids).logits, client.model(ids).logits, atol=1e-6)


def test_service_client_lora_save_and_resume(force_cpu, tiny_model_dir, tmp_path):
    service = ServiceClient(output_dir=str(tmp_path / "out"))
    client = service.create_lora_training_client(tiny_model_dir, rank=4, target_modules=["c_attn"])
    try:
        assert all("lora_" in name for name, p in client.model.named_parameters() if p.requires_grad)
        data = _sft_data(client.tokenizer)
        client.forward_backward(data, "cross_entropy")
        client.optim_step(AdamParams(learning_rate=1e-2))
        path = client.save_state("checkpoint-1")
        assert is_resumable_checkpoint(Path(path))
        reference = client.forward(data).loss_fn_outputs
    finally:
        client.close()

    resumed = service.create_training_client_from_state_with_optimizer(path)
    try:
        assert resumed.step == 1
        assert resumed._optimizer is not None
        for a, b in zip(reference, resumed.forward(data).loss_fn_outputs):
            assert torch.allclose(a["logprobs"], b["logprobs"], atol=1e-5)
    finally:
        resumed.close()
