"""Paired base-vs-GRPO evaluation on held-out GSM8K.

Scores both models with the repo's OWN template and reward functions
(trainers/templates/gsm8k_template.py, trainers/rewards/*), so the numbers
measure exactly what GRPO optimized rather than a separate eval regex.

Both arms see identical prompts and identical tokenizer treatment; the only
difference is whether the LoRA adapter is attached. Greedy decoding, so the
comparison is deterministic.

Usage:
  python scripts/ab_eval_grpo.py --arm nokl=/tmp/grpo_trained --arm kl=/tmp/grpo_fixed --n 100
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainers.rewards.accuracy_rewards import extract_answer
from trainers.rewards.format_rewards import count_xml
from trainers.templates.gsm8k_template import GSM8KTemplate


def build_tokenizer(base_model):
    """Mirror the FIXED core/trainer_base.py:setup_tokenizer_with_model: add a pad
    token only when the tokenizer lacks one, so the vocab is never grown."""
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "left"
    return tok


def adapter_embed_rows(adapter):
    """Embedding rows baked into an adapter, or None if it carries none.

    Pre-fix checkpoints contain a full resized embed_tokens; post-fix ones don't.
    Detecting it lets one script score both layouts with no manual flag.
    """
    path = os.path.join(adapter, "adapter_model.safetensors")
    if not os.path.exists(path):
        return None
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.endswith("embed_tokens.weight"):
                return f.get_slice(k).get_shape()[0]
    return None


def load_model(base_model, tok, adapter=None, device="mps"):
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16)
    if adapter:
        rows = adapter_embed_rows(adapter)
        if rows and rows != model.get_input_embeddings().weight.shape[0]:
            print(f"    (legacy adapter carries a resized embedding: {rows} rows)")
            model.resize_token_embeddings(rows)
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return model.to(device).eval()


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size, device):
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            out.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True))
        print(f"    {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


def norm(s):
    """Normalize a numeric answer so 18,000 == 18000 and 7.0 == 7."""
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def last_number(text):
    """Format-agnostic GSM8K extraction: the last number in the completion.
    Needed because the repo's extract_answer requires <answer> tags, which the
    BASE model never emits -- scoring it with the strict extractor reports 0%
    regardless of whether the arithmetic was right."""
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return norm(m[-1]) if m else ""


def score(completions, answers, tok):
    """Score each completion two ways:
      strict_correct  - the repo's <answer>-tag extractor (what GRPO optimized)
      loose_correct   - last-number-in-text (true math accuracy, format-blind)
    """
    rows = []
    for comp, ans in zip(completions, answers):
        pred = extract_answer(comp)
        rows.append({
            "exact": 2.0 if pred == ans else 0.0,
            "xml": count_xml(comp),
            "digit": 0.5 if pred.isdigit() else 0.0,
            "ntok": len(tok(comp)["input_ids"]),
            "strict_correct": pred == ans,
            "loose_correct": last_number(comp) == norm(ans),
        })
    return rows


def summarize(name, rows):
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    strict = sum(r["strict_correct"] for r in rows) / n
    loose = sum(r["loose_correct"] for r in rows) / n
    print(f"  {name:<6} strict={strict:6.1%} ({sum(r['strict_correct'] for r in rows):>3}/{n})   "
          f"loose={loose:6.1%} ({sum(r['loose_correct'] for r in rows):>3}/{n})   "
          f"xml={mean('xml'):.3f}  digit={mean('digit'):.3f}  mean_tok={mean('ntok'):6.1f}")
    return strict, loose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                   help="repeatable; a 'base' arm (no adapter) is always run first")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--dump", default="/tmp/ab_eval_rows.json")
    args = p.parse_args()
    dump_path = args.dump

    ds = load_dataset("openai/gsm8k", "main", split=f"{args.split}[:{args.n}]")
    tok = build_tokenizer(args.base)

    prompts, answers = [], []
    for ex in ds:
        f = GSM8KTemplate.format_for_training(ex, "question", "answer")
        prompts.append(tok.apply_chat_template(f["prompt"], tokenize=False,
                                               add_generation_prompt=True))
        answers.append(f["answer"])

    print(f"Held-out GSM8K {args.split}[:{args.n}], greedy, max_new_tokens={args.max_new_tokens}\n")
    arms = [("base", None)] + [tuple(a.split("=", 1)) for a in args.arm]
    results = {}
    for name, adapter in arms:
        print(f"  generating: {name}")
        t0 = time.time()
        model = load_model(args.base, tok, adapter, args.device)
        comps = generate(model, tok, prompts, args.max_new_tokens, args.batch_size, args.device)
        results[name] = score(comps, answers, tok)
        json.dump(comps, open(f"/tmp/ab_completions_{name}.json", "w"), indent=1)
        del model
        torch.mps.empty_cache()
        print(f"    done in {time.time() - t0:.0f}s")

    print("\nRESULTS   strict = repo <answer>-tag extractor (what GRPO optimized)")
    print("          loose  = last-number-in-text (format-blind true accuracy)\n")
    accs = {name: summarize(name, results[name]) for name, _ in arms}

    print("\n  vs base (paired, identical problems):")
    for name, _ in arms[1:]:
        for i, (lbl, key) in enumerate([("strict", "strict_correct"),
                                        ("loose ", "loose_correct")]):
            gained = sum(1 for b, g in zip(results["base"], results[name])
                         if g[key] and not b[key])
            lost = sum(1 for b, g in zip(results["base"], results[name])
                       if b[key] and not g[key])
            print(f"    {name:<6} {lbl}: {accs[name][i] - accs['base'][i]:+6.1%}   "
                  f"({gained} fixed, {lost} broken, net {gained - lost:+d}"
                  f"/{len(results['base'])})")

    with open(dump_path, "w") as f:
        json.dump({k: [dict(r) for r in v] for k, v in results.items()}, f, indent=1)
    print(f"\n  per-example scores -> {dump_path}")


if __name__ == "__main__":
    main()
