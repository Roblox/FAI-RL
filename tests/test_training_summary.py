"""Training summaries and resume dispatch need no model or GPU installation."""

from types import SimpleNamespace
from unittest.mock import Mock

from utils.training_summary import log_training_summary


def owner(rank=0):
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
            training=SimpleNamespace(algorithm="sft"),
        ),
        train_dataset=[1, 2],
        logger=Mock(),
    )


def test_batch_and_parameter_summary():
    trainer = owner()
    log_training_summary(trainer.config, trainer.trainer, trainer.train_dataset, trainer.logger)
    trainer.trainer.train.assert_not_called()
    call = trainer.logger.info.call_args.args
    output = call[0] % call[1:]
    assert "Effective batch size: 64" in output
    assert "Trainable params: 100 / 100" in output
    assert "Excludes activations" in output


def test_rank_suppression():
    trainer = owner(rank=1)
    log_training_summary(trainer.config, trainer.trainer, trainer.train_dataset, trainer.logger)
    trainer.trainer.train.assert_not_called()
    trainer.logger.info.assert_not_called()
