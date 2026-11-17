"""In-process, synchronous training client for writing your own training loop.

``forward_backward`` accumulates gradients, ``optim_step`` applies them. Under
``torchrun`` every rank drives its own client on its own data; only
``optim_step`` (gradient all-reduce, SUM) and checkpoint saves are collective,
so every rank must call them the same number of times.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist

from . import distributed
from .losses import LossFn, get_loss
from .types import AdamParams, Datum, ForwardBackwardOutput, OptimStepOutput

TRAINER_STATE_FILE = "trainer_state.json"
OPTIMIZER_FILE = "optimizer.pt"
SCHEDULER_FILE = "scheduler.pt"

# Gradient all-reduce bucket size (elements) so full fine-tuning does not
# materialize one flat copy of every gradient at once.
_ALL_REDUCE_BUCKET_NUMEL = 64 * 1024 * 1024


def _is_peft_model(model) -> bool:
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(model, PeftModel)


def _load_full_state_dict(path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        state: Dict[str, torch.Tensor] = {}
        for shard in shards:
            state.update(load_file(os.path.join(path, shard)))
        return state
    raise FileNotFoundError(f"No safetensors weights found in {path}")


class TrainingClient:
    """Train one model with ``forward_backward`` / ``optim_step`` primitives.

    Usually created through ``engine.ServiceClient``; constructing it directly
    with an already-loaded model and tokenizer is supported too.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        micro_batch_tokens: Optional[int] = None,
        gradient_checkpointing: bool = False,
        bf16: bool = False,
        output_dir: str = "outputs/engine",
        device: Optional[Union[str, torch.device]] = None,
        logger=None,
    ):
        if micro_batch_tokens is not None and micro_batch_tokens <= 0:
            raise ValueError("micro_batch_tokens must be positive when set")
        self.model = model
        self.tokenizer = tokenizer
        self.micro_batch_tokens = micro_batch_tokens
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        self.step = 0
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._adam_params = AdamParams()
        self._close_callbacks: List[Callable[[], None]] = []

        distributed.init_process_group_if_needed()
        self.device = self._place_model(device)
        self._autocast_dtype = (
            torch.bfloat16 if bf16 and self.device.type == "cuda" else None
        )
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if getattr(model, "config", None) is not None:
                model.config.use_cache = False

        self._trainable = [p for p in model.parameters() if p.requires_grad]
        if not self._trainable:
            raise ValueError("TrainingClient model has no trainable parameters")
        if distributed.is_distributed():
            for parameter in self._trainable:
                dist.broadcast(parameter.data, src=0)

        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", None)
        self._pad_token_id = 0 if pad is None else int(pad)

    # ------------------------------------------------------------------ setup

    def _place_model(self, device) -> torch.device:
        if getattr(self.model, "hf_device_map", None):
            # device_map-loaded (quantized / MPS) models are already placed.
            return self.model.get_input_embeddings().weight.device
        if device is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                device = torch.device("cuda", local_rank)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        device = torch.device(device)
        self.model.to(device)
        return device

    def get_tokenizer(self):
        return self.tokenizer

    def add_close_callback(self, callback: Callable[[], None]) -> None:
        self._close_callbacks.append(callback)

    def close(self) -> None:
        callbacks, self._close_callbacks = self._close_callbacks, []
        for callback in callbacks:
            callback()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --------------------------------------------------------------- forward

    def _autocast(self):
        if self._autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype)

    def _micro_batches(self, data: List[Datum]) -> List[List[Datum]]:
        """Split ``data`` into consecutive chunks of at most
        ``micro_batch_tokens`` padded tokens (always at least one datum)."""
        if not self.micro_batch_tokens:
            return [list(data)] if data else []
        chunks: List[List[Datum]] = []
        current: List[Datum] = []
        width = 0
        for datum in data:
            new_width = max(width, datum.model_input.length)
            if current and new_width * (len(current) + 1) > self.micro_batch_tokens:
                chunks.append(current)
                current, new_width = [], datum.model_input.length
            current.append(datum)
            width = new_width
        if current:
            chunks.append(current)
        return chunks

    def _logprobs(self, chunk: List[Datum]) -> List[torch.Tensor]:
        """Target-token log-probs for each datum in ``chunk`` (one model call)."""
        lengths = [datum.model_input.length for datum in chunk]
        width = max(lengths)
        input_ids = torch.full((len(chunk), width), self._pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, datum in enumerate(chunk):
            input_ids[row, :lengths[row]] = torch.tensor(datum.model_input.tokens)
            attention_mask[row, :lengths[row]] = 1

        with self._autocast():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                use_cache=False,
            )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        result = []
        for row, (datum, length) in enumerate(zip(chunk, lengths)):
            row_logits = logits[row, :length].float()
            targets = datum.loss_fn_inputs["target_tokens"].to(row_logits.device)
            index = targets if targets.dim() == 2 else targets.unsqueeze(-1)
            logprobs = row_logits.gather(-1, index) - row_logits.logsumexp(-1, keepdim=True)
            result.append(logprobs if targets.dim() == 2 else logprobs.squeeze(-1))
        return result

    def _on_device(self, data: List[Datum]) -> List[Datum]:
        return [
            Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    key: value.to(self.device) for key, value in datum.loss_fn_inputs.items()
                },
            )
            for datum in data
        ]

    @staticmethod
    def _resolve_loss(loss_fn: Union[str, LossFn]) -> LossFn:
        return get_loss(loss_fn).fn if isinstance(loss_fn, str) else loss_fn

    def forward(
        self,
        data: List[Datum],
        loss_fn: Optional[Union[str, LossFn]] = None,
    ) -> ForwardBackwardOutput:
        """Log-probs (and the loss metrics, if ``loss_fn`` is given) without
        gradients, in eval mode."""
        if not data:
            return ForwardBackwardOutput()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logprobs = [lp for chunk in self._micro_batches(data) for lp in self._logprobs(chunk)]
                metrics: Dict[str, float] = {}
                if loss_fn is not None:
                    _, metrics = self._resolve_loss(loss_fn)(self._on_device(data), logprobs)
        finally:
            self.model.train(was_training)
        return ForwardBackwardOutput(
            loss_fn_outputs=[{"logprobs": lp} for lp in logprobs],
            metrics={key: float(value) for key, value in metrics.items()},
        )

    # -------------------------------------------------------------- backward

    def forward_backward(self, data: List[Datum], loss_fn: str) -> ForwardBackwardOutput:
        """Accumulate gradients of a registered loss (e.g. ``"cross_entropy"``)."""
        registered = get_loss(loss_fn)
        if registered.separable:
            return self._forward_backward_separable(data, registered.fn)
        return self._forward_backward_whole(data, registered.fn)

    def forward_backward_custom(self, data: List[Datum], loss_fn: LossFn) -> ForwardBackwardOutput:
        """Accumulate gradients of an arbitrary ``loss_fn(data, logprobs)``.

        The loss may couple datums (pairs, groups, batch statistics). When the
        batch spans several micro-batches, gradients are computed with a
        two-pass surrogate: a no-grad pass gives ``dloss/dlogprobs``, then a
        micro-batched backward of ``sum(logprobs * grad)``. That is exact up to
        dropout noise between the two passes.
        """
        return self._forward_backward_whole(data, loss_fn)

    def _forward_backward_separable(self, data: List[Datum], loss_fn: LossFn) -> ForwardBackwardOutput:
        self.model.train()
        outputs: List[Dict[str, torch.Tensor]] = []
        metrics: Dict[str, float] = {}
        for chunk in self._micro_batches(data):
            logprobs = self._logprobs(chunk)
            loss, chunk_metrics = loss_fn(self._on_device(chunk), logprobs)
            loss.backward()
            outputs.extend({"logprobs": lp.detach()} for lp in logprobs)
            for key, value in chunk_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + float(value)
        return ForwardBackwardOutput(loss_fn_outputs=outputs, metrics=metrics)

    def _forward_backward_whole(self, data: List[Datum], loss_fn: LossFn) -> ForwardBackwardOutput:
        chunks = self._micro_batches(data)
        if not chunks:
            return ForwardBackwardOutput()
        self.model.train()
        device_data = self._on_device(data)

        if len(chunks) == 1:
            logprobs = self._logprobs(chunks[0])
            loss, metrics = loss_fn(device_data, logprobs)
            if loss.requires_grad:
                loss.backward()
            detached = [lp.detach() for lp in logprobs]
        else:
            with torch.no_grad():
                detached = [lp for chunk in chunks for lp in self._logprobs(chunk)]
            leaves = [lp.detach().requires_grad_(True) for lp in detached]
            loss, metrics = loss_fn(device_data, leaves)
            if loss.requires_grad:
                loss.backward()
                grads = [
                    leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
                    for leaf in leaves
                ]
                offset = 0
                for chunk in chunks:
                    chunk_grads = grads[offset:offset + len(chunk)]
                    offset += len(chunk)
                    surrogate = sum(
                        (lp * grad).sum() for lp, grad in zip(self._logprobs(chunk), chunk_grads)
                    )
                    surrogate.backward()

        return ForwardBackwardOutput(
            loss_fn_outputs=[{"logprobs": lp} for lp in detached],
            metrics={key: float(value) for key, value in metrics.items()},
        )

    # ------------------------------------------------------------- optimizer

    def _build_optimizer(self, params: AdamParams) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self._trainable,
            lr=params.learning_rate,
            betas=(params.beta1, params.beta2),
            eps=params.eps,
            weight_decay=params.weight_decay,
        )

    def _all_reduce_grads(self) -> None:
        if not distributed.is_distributed():
            return
        for parameter in self._trainable:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        buckets: Dict[tuple, List[List[torch.Tensor]]] = {}
        for parameter in self._trainable:
            groups = buckets.setdefault((parameter.grad.device, parameter.grad.dtype), [[]])
            if sum(g.numel() for g in groups[-1]) + parameter.grad.numel() > _ALL_REDUCE_BUCKET_NUMEL and groups[-1]:
                groups.append([])
            groups[-1].append(parameter.grad)
        for groups in buckets.values():
            for grads in groups:
                flat = torch.cat([grad.reshape(-1) for grad in grads])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                offset = 0
                for grad in grads:
                    grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
                    offset += grad.numel()

    def optim_step(self, params: AdamParams) -> OptimStepOutput:
        """Apply the accumulated gradients with AdamW, then clear them."""
        if self._optimizer is None:
            self._optimizer = self._build_optimizer(params)
        for group in self._optimizer.param_groups:
            group["lr"] = params.learning_rate
            group["betas"] = (params.beta1, params.beta2)
            group["eps"] = params.eps
            group["weight_decay"] = params.weight_decay

        self._all_reduce_grads()
        max_norm = params.grad_clip_norm if params.grad_clip_norm > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable, max_norm)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self._adam_params = params
        return OptimStepOutput(
            metrics={
                "grad_norm": float(grad_norm),
                "learning_rate": params.learning_rate,
                "step": self.step,
            }
        )

    # ------------------------------------------------------------ save / load

    def _resolve_path(self, name: str) -> str:
        return name if os.path.dirname(name) else os.path.join(self.output_dir, name)

    def _save_weights(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_state(self, name: str, extra_state: Optional[dict] = None) -> str:
        """Save weights + optimizer state for resuming; returns the directory.

        The layout matches HF Trainer checkpoints (``trainer_state.json`` with
        ``global_step``, weights, ``optimizer.pt``, ``scheduler.pt``), so a
        ``checkpoint-<step>`` name is picked up by ``--auto-resume``.
        """
        path = self._resolve_path(name)
        if distributed.is_main_process():
            self._save_weights(path)
            optimizer_state = self._optimizer.state_dict() if self._optimizer else {}
            torch.save(optimizer_state, os.path.join(path, OPTIMIZER_FILE))
            torch.save(
                {"step": self.step, "adam_params": self._adam_params.to_dict()},
                os.path.join(path, SCHEDULER_FILE),
            )
            state = {"global_step": self.step, "format": "engine", **(extra_state or {})}
            with open(os.path.join(path, TRAINER_STATE_FILE), "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        distributed.barrier()
        return path

    def save_weights_for_sampler(self, name: str) -> str:
        """Save model weights (PEFT adapter or full model) + tokenizer only.

        The directory loads with ``from_pretrained`` / ``fai-rl-inference``.
        """
        path = self._resolve_path(name)
        if distributed.is_main_process():
            self._save_weights(path)
        distributed.barrier()
        return path

    def _load_weights(self, path: str) -> None:
        if _is_peft_model(self.model):
            from peft import set_peft_model_state_dict
            from peft.utils import load_peft_weights

            weights = load_peft_weights(path, device=str(self.device))
            result = set_peft_model_state_dict(self.model, weights)
            unexpected = [k for k in getattr(result, "unexpected_keys", []) if "lora_" in k]
            if unexpected:
                raise ValueError(f"Adapter at {path} has unexpected keys: {unexpected[:5]}")
            return
        state = _load_full_state_dict(path)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Weights at {path} have unexpected keys: {unexpected[:5]}")
        if missing:
            self.logger.warning(
                "Weights at %s did not include %d keys (e.g. tied embeddings): %s",
                path, len(missing), missing[:5],
            )

    def _load_training_state(self, path: str, with_optimizer: bool) -> None:
        state_file = os.path.join(path, TRAINER_STATE_FILE)
        if os.path.exists(state_file):
            with open(state_file, encoding="utf-8") as f:
                self.step = int(json.load(f).get("global_step", 0))
        scheduler_file = os.path.join(path, SCHEDULER_FILE)
        if os.path.exists(scheduler_file):
            schedule = torch.load(scheduler_file, map_location="cpu", weights_only=True)
            if "adam_params" in schedule:
                self._adam_params = AdamParams(**schedule["adam_params"])
        if not with_optimizer:
            return
        optimizer_file = os.path.join(path, OPTIMIZER_FILE)
        if not os.path.exists(optimizer_file):
            raise FileNotFoundError(f"No optimizer state at {optimizer_file}")
        optimizer_state = torch.load(optimizer_file, map_location="cpu", weights_only=True)
        if optimizer_state:
            if self._optimizer is None:
                self._optimizer = self._build_optimizer(self._adam_params)
            self._optimizer.load_state_dict(optimizer_state)

    def load_state(self, path: str, with_optimizer: bool = True) -> None:
        """Restore weights, step counter and (optionally) optimizer state."""
        self._load_weights(path)
        self._load_training_state(path, with_optimizer)
        self.logger.info("Loaded training state from %s (step %d)", path, self.step)
