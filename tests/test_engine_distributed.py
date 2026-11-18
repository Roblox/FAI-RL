"""Two-process (gloo, CPU) runs match one process on the same global batch."""

import logging
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 2
# GPT-2's attention key bias has an exactly-zero gradient; with a tiny eps,
# Adam's first step turns its float noise into +-lr updates that depend on the
# reduction order. A large eps keeps the update linear in the gradient.
ADAM = dict(learning_rate=1e-2, eps=1.0)
PROBE = "Game A or game B? hello world"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_rank(rank: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        FAI_RL_DISABLE_DEBUG_CALLBACK="1",
    )
    torch.backends.mps.is_available = lambda: False
    torch.cuda.is_available = lambda: False
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)


def _primitives_worker(rank, port, model_dir, result_dir):
    _init_rank(rank, port)
    try:
        from transformers import AutoTokenizer, GPT2LMHeadModel

        from engine import AdamParams, TrainingClient, all_reduce_sum, shard_for_rank
        from test_engine_training_client import _sft_data

        model = GPT2LMHeadModel.from_pretrained(model_dir)
        if rank == 1:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        client = TrainingClient(model, tokenizer, device="cpu")

        data = shard_for_rank(_sft_data(tokenizer))
        client.forward_backward(data, "cross_entropy")
        metrics = client.optim_step(AdamParams(**ADAM)).metrics
        torch.save(
            {
                "grad_norm": metrics["grad_norm"],
                "total": all_reduce_sum(len(data)),
                "state": {k: v.clone() for k, v in model.state_dict().items()},
            },
            os.path.join(result_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


def _sft_loop_worker(rank, port, model_dir, data_path, output_dir):
    _init_rank(rank, port)
    try:
        from engine.cookbook import sft as native_sft
        from test_native_sft_parity import sft_config

        config = sft_config(
            model_dir, data_path, output_dir, per_device_train_batch_size=2, max_steps=4
        )
        result = native_sft.run(config, logging.getLogger(f"rank{rank}"))
        torch.save(result.log_history, os.path.join(output_dir, f"log_history_rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _probe_logits(model_dir):
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).eval()
    with torch.no_grad():
        return model(torch.tensor([tokenizer(PROBE)["input_ids"]])).logits


def test_optim_step_sums_gradients_across_ranks(tiny_model_dir, tmp_path):
    from transformers import AutoTokenizer, GPT2LMHeadModel

    from engine import AdamParams, TrainingClient
    from test_engine_training_client import _sft_data

    mp.spawn(
        _primitives_worker,
        args=(_free_port(), tiny_model_dir, str(tmp_path)),
        nprocs=WORLD_SIZE,
        join=True,
    )
    results = [torch.load(tmp_path / f"rank{rank}.pt") for rank in range(WORLD_SIZE)]

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    model = GPT2LMHeadModel.from_pretrained(tiny_model_dir)
    client = TrainingClient(model, tokenizer, device="cpu")
    client.forward_backward(_sft_data(tokenizer), "cross_entropy")
    expected_norm = client.optim_step(AdamParams(**ADAM)).metrics["grad_norm"]

    for result in results:
        assert result["total"] == 4
        assert result["grad_norm"] == pytest.approx(expected_norm, rel=1e-5)
        for key, value in model.state_dict().items():
            assert torch.allclose(result["state"][key], value, atol=1e-6), key


def test_native_sft_loop_matches_single_process(tiny_model_dir, tmp_path, force_cpu):
    from engine.cookbook import sft as native_sft
    from test_native_sft_parity import sft_config, write_sft_dataset

    data_path = write_sft_dataset(tmp_path / "sft.jsonl", repeats=2)
    mp.spawn(
        _sft_loop_worker,
        args=(_free_port(), tiny_model_dir, data_path, str(tmp_path / "ddp")),
        nprocs=WORLD_SIZE,
        join=True,
    )

    config = sft_config(
        tiny_model_dir,
        data_path,
        tmp_path / "single",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=WORLD_SIZE,
        max_steps=4,
    )
    single = native_sft.run(config, logging.getLogger("single")).log_history

    for rank in range(WORLD_SIZE):
        ddp = torch.load(tmp_path / "ddp" / f"log_history_rank{rank}.pt")
        assert [entry["step"] for entry in ddp] == [entry["step"] for entry in single] == [1, 2, 3, 4]
        for ddp_entry, single_entry in zip(ddp, single):
            assert ddp_entry["loss"] == pytest.approx(single_entry["loss"], rel=1e-5)
            assert ddp_entry["grad_norm"] == pytest.approx(single_entry["grad_norm"], rel=1e-4)
    assert torch.allclose(
        _probe_logits(str(tmp_path / "ddp")), _probe_logits(str(tmp_path / "single")), atol=1e-4
    )
