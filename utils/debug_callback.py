"""DebugCallback: verbose, observation-only training instrumentation.

Always attached by BaseTrainer.build_callbacks(). Every hook is wrapped so a
logging failure can never crash training -- output is strictly best-effort.
Records emit at INFO level so they always print (subject to the usual rank-0
filter; set FAI_RL_LOG_ALL_RANKS=1 to see every rank). Per-step lines are
throttled (first FAI_RL_DEBUG_STEPS steps, then every logging_steps) to avoid
flooding long runs.

What it logs:
  - Config & arch: model class, dtype, attention impl, total/trainable params,
    PEFT adapters, tokenizer/processor id, special tokens, chat template.
  - First batches decoded: input_ids -> text, labels showing masked (-100) vs
    supervised tokens, attention mask, per-example sequence lengths, truncation.
  - Tokenization sanity: round-trip decode, BOS/EOS presence, padding side.
  - Loss/masking stats: % tokens masked per batch, best-effort per-example loss.
  - Optimizer/scheduler: LR at step 0, param groups, grad norm (pre-clip).
  - Distributed: world size / rank, per-rank batch fingerprint (data duplication).
  - Memory/perf: peak accelerator memory, approx tokens/sec, step time.
  - Progress: step count, checkpoint saves.
"""

import os
import time
from typing import Any, Optional

from transformers import TrainerCallback


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _rank() -> int:
    for key in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(key)
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


def _world_size() -> int:
    return _env_int("WORLD_SIZE", 1)


def _short(text: str, limit: int = 800) -> str:
    """Truncate long text for logging with an elision marker."""
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [+{len(text) - limit} chars]"


class DebugCallback(TrainerCallback):
    """Verbose, non-intrusive debug logging for HF/TRL Trainer runs."""

    def __init__(self, logger, config=None):
        # `logger` may be a SafeLogger wrapper or a stdlib Logger; both expose
        # .debug/.info/.warning.
        self.log = logger
        self.config = config
        # Log full per-step detail for the first N steps, then throttle to every
        # logging_steps to avoid flooding long runs.
        self.max_verbose_steps = _env_int("FAI_RL_DEBUG_STEPS", 5)
        self.num_first_batches = _env_int("FAI_RL_DEBUG_BATCHES", 1)
        self._last_step_time: Optional[float] = None
        self._approx_tokens_per_step: Optional[int] = None
        self._logging_steps = 10

    # ------------------------------------------------------------------ utils
    def _d(self, msg: str) -> None:
        """Rank-prefixed diagnostic line at INFO (best-effort, never raises)."""
        try:
            self.log.info(f"[diag r{_rank()}] {msg}")
        except Exception:
            pass

    @staticmethod
    def _get_tokenizer(processing_class):
        """Return a tokenizer from either a bare tokenizer or a VLM processor."""
        return getattr(processing_class, "tokenizer", processing_class)

    # ---------------------------------------------------------- train begin
    def on_train_begin(self, args, state, control, **kwargs):
        self._logging_steps = getattr(args, "logging_steps", 10) or 10
        model = kwargs.get("model")
        processing_class = kwargs.get("processing_class")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        train_dataloader = kwargs.get("train_dataloader")

        self._d("=" * 60)
        self._d("Training diagnostics (observation-only instrumentation)")
        self._d("=" * 60)

        self._log_distributed(args)
        self._log_arch(model)
        self._log_tokenizer(processing_class)
        self._log_optimizer(optimizer, lr_scheduler)
        self._log_first_batches(train_dataloader, processing_class, model, args)

    def _log_distributed(self, args):
        try:
            self._d(
                f"Distributed: world_size={_world_size()} rank={_rank()} "
                f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
                f"n_gpu={getattr(args, 'n_gpu', '?')} "
                f"parallel_mode={getattr(args, 'parallel_mode', '?')}"
            )
            ds = getattr(args, "deepspeed", None)
            self._d(f"DeepSpeed config: {ds if ds else 'none (DDP/single)'}")
            self._d(
                f"Effective batch: per_device={getattr(args, 'per_device_train_batch_size', '?')} "
                f"x grad_accum={getattr(args, 'gradient_accumulation_steps', '?')} "
                f"x world={_world_size()} = "
                f"{getattr(args, 'per_device_train_batch_size', 0) * getattr(args, 'gradient_accumulation_steps', 0) * _world_size()}"
            )
        except Exception as e:
            self._d(f"(distributed info failed: {e})")

    def _log_arch(self, model):
        if model is None:
            self._d("Model not available at on_train_begin")
            return
        try:
            self._d(f"Model class: {model.__class__.__name__}")
            cfg = getattr(model, "config", None)
            if cfg is not None:
                self._d(
                    f"Arch: model_type={getattr(cfg, 'model_type', '?')} "
                    f"attn_impl={getattr(cfg, '_attn_implementation', '?')} "
                    f"torch_dtype={getattr(cfg, 'torch_dtype', '?')} "
                    f"vocab_size={getattr(cfg, 'vocab_size', '?')}"
                )
            # Parameter dtype histogram (what the weights actually are at runtime).
            dtypes = {}
            for p in model.parameters():
                dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
            self._d("Param dtypes: " + ", ".join(f"{k}={v:,}" for k, v in dtypes.items()))

            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            pct = (100.0 * trainable / total) if total else 0.0
            self._d(f"Params: total={total:,} trainable={trainable:,} ({pct:.4f}%)")

            # PEFT adapter summary, if any.
            if hasattr(model, "peft_config"):
                for name, pc in model.peft_config.items():
                    self._d(
                        f"PEFT adapter '{name}': type={getattr(pc, 'peft_type', '?')} "
                        f"r={getattr(pc, 'r', '?')} alpha={getattr(pc, 'lora_alpha', '?')} "
                        f"targets={getattr(pc, 'target_modules', '?')}"
                    )
        except Exception as e:
            self._d(f"(arch info failed: {e})")

    def _log_tokenizer(self, processing_class):
        if processing_class is None:
            self._d("Processing class not available")
            return
        try:
            tok = self._get_tokenizer(processing_class)
            self._d(f"Processing class: {processing_class.__class__.__name__}")
            self._d(f"Tokenizer: {tok.__class__.__name__} vocab_size={getattr(tok, 'vocab_size', '?')}")
            self._d(
                f"Special tokens: bos={getattr(tok, 'bos_token', None)!r}({getattr(tok, 'bos_token_id', None)}) "
                f"eos={getattr(tok, 'eos_token', None)!r}({getattr(tok, 'eos_token_id', None)}) "
                f"pad={getattr(tok, 'pad_token', None)!r}({getattr(tok, 'pad_token_id', None)}) "
                f"padding_side={getattr(tok, 'padding_side', '?')}"
            )
            ct = getattr(tok, "chat_template", None)
            if ct:
                self._d(f"Chat template present ({len(ct)} chars): {_short(ct, 300)}")
            else:
                self._d("Chat template: none")
        except Exception as e:
            self._d(f"(tokenizer info failed: {e})")

    def _log_optimizer(self, optimizer, lr_scheduler):
        try:
            if optimizer is not None:
                self._d(f"Optimizer: {optimizer.__class__.__name__} "
                        f"({len(optimizer.param_groups)} param groups)")
                for i, g in enumerate(optimizer.param_groups):
                    n_params = sum(p.numel() for p in g.get("params", []))
                    self._d(
                        f"  group[{i}]: lr={g.get('lr')} weight_decay={g.get('weight_decay')} "
                        f"n_tensors={len(g.get('params', []))} n_params={n_params:,}"
                    )
            if lr_scheduler is not None:
                self._d(f"LR scheduler: {lr_scheduler.__class__.__name__}")
                try:
                    self._d(f"LR at step 0: {lr_scheduler.get_last_lr()}")
                except Exception:
                    pass
        except Exception as e:
            self._d(f"(optimizer info failed: {e})")

    # ------------------------------------------------------- first batches
    def _log_first_batches(self, train_dataloader, processing_class, model, args):
        if train_dataloader is None:
            self._d("train_dataloader not available; skipping batch inspection")
            return
        tok = self._get_tokenizer(processing_class)
        try:
            it = iter(train_dataloader)
        except Exception as e:
            self._d(f"(could not iterate train_dataloader: {e})")
            return

        for b in range(self.num_first_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            except Exception as e:
                self._d(f"(fetching batch {b} failed: {e})")
                break
            self._d(f"----- first batch #{b} -----")
            self._inspect_batch(batch, tok, args)
            self._maybe_per_example_loss(batch, model)

    def _inspect_batch(self, batch, tok, args):
        try:
            if not isinstance(batch, dict):
                self._d(f"Batch type {type(batch)} (not a dict); keys unavailable")
                return
            self._d("Batch keys: " + ", ".join(f"{k}:{tuple(getattr(v, 'shape', ()))}"
                                                for k, v in batch.items()))
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")

            # Multimodal signal
            if "pixel_values" in batch:
                pv = batch["pixel_values"]
                self._d(f"pixel_values present: shape={tuple(getattr(pv, 'shape', ()))} dtype={getattr(pv, 'dtype', '?')}")

            if input_ids is None:
                self._d("No input_ids in batch")
                return

            # Sequence-length stats + truncation.
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).tolist()
            else:
                lengths = [input_ids.shape[1]] * input_ids.shape[0]
            self._d(f"Seq lengths (real tokens): {lengths} | padded_to={input_ids.shape[1]}")
            # Rough tokens/step estimate for the tok/s readout in on_step_end:
            # one micro-batch's real tokens x grad_accum x world_size.
            if self._approx_tokens_per_step is None:
                ga = getattr(args, "gradient_accumulation_steps", 1) or 1
                self._approx_tokens_per_step = int(sum(lengths)) * ga * _world_size()
            max_len = getattr(args, "max_length", None)
            if max_len:
                n_trunc = sum(1 for L in lengths if L >= max_len)
                if n_trunc:
                    self._d(f"WARNING: {n_trunc}/{len(lengths)} examples at/over max_length={max_len} (possible truncation)")

            # Per-rank fingerprint to catch data duplication across ranks.
            try:
                fp = int(input_ids[0].sum().item())
                self._d(f"Data fingerprint (row0 input_ids sum): {fp} "
                        f"first8={input_ids[0][:8].tolist()} "
                        f"(compare across ranks with FAI_RL_LOG_ALL_RANKS=1)")
            except Exception:
                pass

            # Decode + round-trip sanity on example 0.
            ids0 = input_ids[0].tolist()
            decoded = tok.decode(ids0, skip_special_tokens=False)
            self._d(f"Decoded input[0] (with specials): {_short(decoded)}")
            self._d(f"BOS present: {tok.bos_token_id in ids0 if tok.bos_token_id is not None else 'n/a'} | "
                    f"EOS present: {tok.eos_token_id in ids0 if tok.eos_token_id is not None else 'n/a'}")
            try:
                reencoded = tok(decoded, add_special_tokens=False)["input_ids"]
                self._d(f"Round-trip decode: len_in={len(ids0)} len_reencoded={len(reencoded)} "
                        f"exact_match={reencoded == ids0}")
            except Exception:
                pass

            # Masking: masked (-100) vs supervised, aligned with tokens.
            if labels is not None:
                self._log_masking(input_ids, labels, tok)
        except Exception as e:
            self._d(f"(batch inspection failed: {e})")

    def _log_masking(self, input_ids, labels, tok):
        try:
            total = labels.numel()
            masked = int((labels == -100).sum().item())
            supervised = total - masked
            self._d(f"Label masking: {masked:,}/{total:,} masked (-100) = "
                    f"{100.0 * masked / total:.1f}% | supervised={supervised:,} "
                    f"({100.0 * supervised / total:.1f}%)")
            # Show supervised span for example 0.
            lab0 = labels[0]
            sup_ids = input_ids[0][lab0 != -100].tolist()
            if sup_ids:
                self._d(f"Supervised region (labels != -100) of example[0] decodes to: "
                        f"{_short(tok.decode(sup_ids, skip_special_tokens=False))}")
            else:
                self._d("Example[0] has NO supervised tokens (all labels == -100!)")
            # Compact aligned view of the boundary (first 40 positions).
            ids0 = input_ids[0][:40].tolist()
            l0 = lab0[:40].tolist()
            preview = " ".join(
                f"{tok.convert_ids_to_tokens([t])[0]}{'*' if lv == -100 else ''}"
                for t, lv in zip(ids0, l0)
            )
            self._d(f"Token/label align (first 40, * = masked): {_short(preview)}")
        except Exception as e:
            self._d(f"(masking stats failed: {e})")

    def _maybe_per_example_loss(self, batch, model):
        """Best-effort per-example loss via one no-grad forward. Skipped on any
        error (e.g. OOM on very large models / long sequences)."""
        if model is None or not isinstance(batch, dict) or "labels" not in batch:
            return
        try:
            import torch

            device = next(model.parameters()).device
            model_inputs = {}
            for k, v in batch.items():
                model_inputs[k] = v.to(device) if hasattr(v, "to") else v
            labels = model_inputs["labels"]
            was_training = model.training
            model.eval()
            with torch.no_grad():
                out = model(**model_inputs)
            if was_training:
                model.train()
            logits = out.logits.float()
            # Causal shift.
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_tok = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.size())
            valid = (shift_labels != -100)
            per_ex = (loss_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            self._d(f"Per-example loss (batch 0, no-grad): "
                    f"{[round(x, 4) for x in per_ex.tolist()]} | mean={per_ex.mean().item():.4f}")
        except Exception as e:
            self._d(f"(per-example loss skipped: {type(e).__name__}: {e})")

    # ----------------------------------------------------------- per step
    def _should_verbose(self, step: int) -> bool:
        return step <= self.max_verbose_steps or (step % self._logging_steps == 0)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Grad norm BEFORE clipping (post-clip shows up in on_log as grad_norm)."""
        step = int(getattr(state, "global_step", 0)) + 1
        if not self._should_verbose(step):
            return
        model = kwargs.get("model")
        if model is None:
            return
        try:
            import torch

            total_sq = 0.0
            n = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_sq += float(p.grad.detach().data.norm(2).item()) ** 2
                    n += 1
            self._d(f"[step {step}] grad norm (pre-clip) over {n} tensors: {total_sq ** 0.5:.4f} "
                    f"(max_grad_norm={getattr(args, 'max_grad_norm', '?')})")
        except Exception as e:
            self._d(f"(grad norm failed: {e})")

    def on_step_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0))
        now = time.time()
        dt = None if self._last_step_time is None else now - self._last_step_time
        self._last_step_time = now
        if not self._should_verbose(step):
            return
        try:
            parts = [f"[step {step}]"]
            if dt is not None:
                parts.append(f"step_time={dt:.3f}s")
                if self._approx_tokens_per_step:
                    parts.append(f"~{self._approx_tokens_per_step / dt:,.0f} tok/s (approx)")
            parts.append(self._mem_str())
            self._d(" ".join(parts))
        except Exception as e:
            self._d(f"(step_end stats failed: {e})")

    @staticmethod
    def _mem_str() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                torch.cuda.reset_peak_memory_stats()
                return f"mem alloc={alloc:.2f}GB peak={peak:.2f}GB reserved={reserved:.2f}GB"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return f"mps alloc={torch.mps.current_allocated_memory() / 1024**3:.2f}GB"
        except Exception:
            pass
        return "mem n/a (cpu)"

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Echo the trainer's own metrics (loss, grad_norm, lr) at debug level."""
        if not logs:
            return
        try:
            keep = {k: logs[k] for k in ("loss", "grad_norm", "learning_rate", "epoch") if k in logs}
            if keep:
                self._d(f"[step {int(getattr(state, 'global_step', 0))}] metrics: {keep}")
        except Exception:
            pass

    def on_save(self, args, state, control, **kwargs):
        try:
            step = int(getattr(state, "global_step", 0))
            ckpt = os.path.join(getattr(args, "output_dir", "?"), f"checkpoint-{step}")
            self._d(f"[step {step}] checkpoint saved -> {ckpt} "
                    f"(save_only_model={getattr(args, 'save_only_model', '?')})")
        except Exception:
            pass
