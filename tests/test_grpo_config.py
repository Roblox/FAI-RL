import logging
from types import SimpleNamespace

from datasets import Dataset
from trl import GRPOTrainer as TRLGRPOTrainer
from trl.trainer import grpo_trainer as trl_grpo_module

from core.config import (
    DataConfig,
    DatasetInfo,
    ToolDefinition,
    ToolEnvironmentConfig,
    TrainingConfig,
)
from trainers import grpo_trainer


def test_grpo_forwards_beta_to_trl_config():
    trainer = object.__new__(grpo_trainer.GRPOTrainer)
    trainer.logger = logging.getLogger("test_grpo_beta")
    trainer.config = SimpleNamespace(
        training=TrainingConfig(
            output_dir="out",
            beta=0.04,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            bf16=False,
            fp16=False,
        ),
        data=DataConfig(),
        wandb=SimpleNamespace(enabled=False),
    )

    args = trainer.setup_training_args()

    assert args.beta == 0.04


def test_grpo_wires_managed_environment_and_turn_limit(monkeypatch):
    environment = ToolEnvironmentConfig(
        gateway_url="https://tools.example",
        server_id="inventory",
        tools=[
            ToolDefinition(
                name="lookup_item",
                input_schema={"type": "object"},
            )
        ],
        max_turns=5,
    )
    trainer = object.__new__(grpo_trainer.GRPOTrainer)
    trainer.logger = logging.getLogger("test_grpo_tool_environment")
    trainer.config = SimpleNamespace(
        training=TrainingConfig(
            output_dir="out",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            bf16=False,
            fp16=False,
        ),
        data=DataConfig(),
        wandb=SimpleNamespace(enabled=False),
        tool_environment=environment,
    )
    trainer.model = object()
    trainer.tokenizer = object()
    trainer.train_dataset = object()
    trainer.build_callbacks = lambda: []
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(grpo_trainer, "TRLGRPOTrainer", FakeTrainer)
    monkeypatch.setattr(
        grpo_trainer,
        "build_reward_function",
        lambda *_args, **_kwargs: object(),
    )

    assert trainer.setup_training_args().max_tool_calling_iterations == 5
    trainer.setup_trainer()

    assert captured["processing_class"] is trainer.tokenizer
    rollout_environment = captured["environment_factory"]()
    assert rollout_environment.config is environment


def test_tool_grpo_keeps_only_selected_reward_context_columns(monkeypatch):
    trainer = object.__new__(grpo_trainer.GRPOTrainer)
    trainer.logger = logging.getLogger("test_grpo_reward_context")
    trainer.config = SimpleNamespace(
        data=DataConfig(
            datasets=[
                DatasetInfo(
                    name="Roblox/tool-training",
                    prompt_column="question",
                    reward_context_columns=["expected", "difficulty"],
                )
            ],
        ),
        tool_environment=ToolEnvironmentConfig(
            gateway_url="https://tools.example",
            server_id="inventory",
            tools=[
                ToolDefinition(
                    name="lookup_item",
                    input_schema={"type": "object"},
                )
            ],
        ),
    )
    trainer.tokenizer = object()
    monkeypatch.setattr(
        grpo_trainer,
        "load_raw_dataset",
        lambda _info: Dataset.from_dict(
            {
                "question": ["Where is item 42?"],
                "expected": ["aisle 7"],
                "difficulty": ["easy"],
                "private_metadata": ["must-not-reach-reward-api"],
            }
        ),
    )

    trainer.setup_data()

    assert set(trainer.train_dataset.column_names) == {
        "prompt",
        "expected",
        "difficulty",
    }


def test_pinned_trl_masks_tool_observation_tokens_from_grpo_loss(monkeypatch):
    trainer = object.__new__(TRLGRPOTrainer)
    trainer._sync_tool_dicts = [{"call_tool": lambda **_kwargs: '"result"'}]
    trainer._async_tool_dicts = [{}]
    trainer.max_tool_calling_iterations = 1
    trainer.max_completion_length = 32
    trainer.use_vllm = False
    trainer._is_vlm = False
    trainer.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128))
    trainer._get_tool_suffix_ids = lambda _messages: [90, 91]
    trainer._generate_single_turn = lambda _prompt_ids, _images, _fields: (
        [[7, 8]],
        None,
    )
    monkeypatch.setattr(
        trl_grpo_module,
        "parse_response",
        lambda _processing_class, _ids: {
            "role": "assistant",
            "content": "done",
        },
    )
    trainer.processing_class = object()

    (
        tool_mask,
        _completions,
        _completion_ids,
        _logprobs,
        tool_call_count,
        tool_failure_count,
        _tool_images,
    ) = trainer._tool_call_loop(
        prompts=[[{"role": "user", "content": "look up item 42"}]],
        prompt_ids=[[1]],
        completion_ids=[[2, 3]],
        completions=[
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "call_tool",
                                "arguments": {
                                    "tool_name": "lookup_item",
                                    "arguments": {"item_id": "42"},
                                },
                            },
                        }
                    ],
                }
            ]
        ],
        logprobs=None,
        images=None,
        multimodal_fields=None,
    )

    assert tool_mask == [[1, 1, 0, 0, 1, 1]]
    assert tool_call_count == 1
    assert tool_failure_count == 0
