"""Shared fixtures for tests that need a real (tiny) causal LM without downloads."""

import pytest
import torch

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
_CORPUS = [
    "Game A has score 10. Game B has score 4. Which game has the higher score? A B",
    "Compare game A and game B. Reply with only A or B.",
    "What is two plus two? four. Say hello. hello world",
] * 50


def make_tiny_lm(seed: int = 0):
    """A 2-layer GPT-2 (dropout off) and a BPE tokenizer with a chat template."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        _CORPUS,
        vocab_size=400,
        special_tokens=["<eos>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer, eos_token="<eos>", pad_token="<eos>"
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_positions=256,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    config.eos_token_id = tokenizer.eos_token_id
    return GPT2LMHeadModel(config), tokenizer


@pytest.fixture
def tiny_lm():
    return make_tiny_lm()


@pytest.fixture
def tiny_model_dir(tmp_path):
    model, tokenizer = make_tiny_lm()
    path = tmp_path / "tiny-lm"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture
def force_cpu(monkeypatch):
    """Run model-loading code paths on CPU even on Apple Silicon."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FAI_RL_DISABLE_DEBUG_CALLBACK", "1")
