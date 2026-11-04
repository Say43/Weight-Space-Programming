"""Measures candidate edits on a fixed part of fit. One run is one experiment."""
import json
import random
import sys

import torch

from baselines.run_b1 import split_validation
from editor import lab
from editor.tools import with_deltas
from eval import config as cfg
from eval.measure_train import load_train
from eval.metrics import condition_metrics
from eval.model import load_model, load_tokenizer
from eval.scoring import score_items

HEAD_DIM = 64


def dev_items(n, seed=cfg.SEED):
    """Fixed random sample of fit with its indices, so that runs stay comparable."""
    items = load_train()
    fit, _ = split_validation(len(items))
    picked = random.Random(f"{seed}:dev").sample(fit, n)
    return picked, [items[i] for i in picked]


def head_scale_deltas(model, scales):
    """Scales the o_proj columns of single heads; scales: {(layer, head): factor}."""
    params = dict(model.named_parameters())
    deltas = {}
    for (layer, head), factor in scales.items():
        name = f"model.layers.{layer}.self_attn.o_proj.weight"
        weight = params[name].data
        delta = deltas.setdefault(name, torch.zeros_like(weight))
        cols = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        delta[:, cols] = weight[:, cols] * (factor - 1.0)
    return deltas


def run(track_note, code_path, candidates, conditions, n, model=None, tok=None):
    """candidates: name -> function(model) -> deltas. Books one run and prints one line per candidate."""
    tok = tok or load_tokenizer()
    model = model or load_model()
    index, items = dev_items(n)
    base = torch.load(cfg.LOG_DIR / "train_baseline_predictions.pt")
    number = lab.start("editor", track_note, code_path, len(candidates) * len(conditions) * n)
    where = torch.tensor(index)
    results = {"baseline": condition_metrics(items, {c: base[c][where] for c in conditions})}
    for name, build in candidates.items():
        with with_deltas(model, build(model)):
            predictions = {c: score_items(model, tok, items, c) for c in conditions}
        results[name] = condition_metrics(items, predictions)
    summary = {k: {m: round(v["rate"], 3) for m, v in r.items()} for k, r in results.items()}
    lab.finish("editor", number, summary)
    for name, row in summary.items():
        print(name, json.dumps(row), flush=True)
    sys.stdout.flush()
    return results
