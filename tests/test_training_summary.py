"""Training summaries and resume dispatch need no model or GPU installation."""

from types import SimpleNamespace
from unittest.mock import Mock

from utils.training_summary import run_trainer


def owner(rank=0, resume=None):
    parameter = SimpleNamespace(numel=lambda: 100, requires_grad=True)
    model = SimpleNamespace(parameters=lambda: [parameter])
    args = SimpleNamespace(
        process_index=rank,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        world_size=8,
        bf16=True,
        fp16=False,
    )
    return SimpleNamespace(
        trainer=SimpleNamespace(model=model, args=args, train=Mock()),
        config=SimpleNamespace(
            model=SimpleNamespace(base_model_name="test", torch_dtype="bfloat16"),
            training=SimpleNamespace(algorithm="sft", resume_from_checkpoint=resume),
        ),
        train_dataset=[1, 2],
        logger=Mock(),
    )


def test_default_train_call_and_batch_summary():
    trainer = owner()
    run_trainer(trainer)
    trainer.trainer.train.assert_called_once_with()
    call = trainer.logger.info.call_args.args
    output = call[0] % call[1:]
    assert "Effective batch size: 64" in output
    assert "Trainable params: 100 / 100" in output
    assert "Excludes activations" in output


def test_resume_dispatch_and_rank_suppression():
    trainer = owner(rank=1, resume="checkpoint-10")
    run_trainer(trainer)
    trainer.trainer.train.assert_called_once_with(resume_from_checkpoint="checkpoint-10")
    trainer.logger.info.assert_not_called()
