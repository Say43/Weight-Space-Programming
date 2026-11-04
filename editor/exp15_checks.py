"""Experimente 15 bis 17: Validierung, Faehigkeits-Waechter und unbekannte Items fuer den Kandidaten."""
import json
import random

import numpy as np
import torch

from baselines.run_b1 import split_validation
from data.handcheck import HANDCHECK_PATH
from data.sources import load_arc, load_mmlu
from editor import lab
from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import filter_deltas
from editor.tools import with_deltas
from eval import config as cfg
from eval.guards import CHUNK, perplexity
from eval.measure_train import load_train
from eval.metrics import condition_metrics, letter_index, rate_ci
from eval.model import load_model, load_tokenizer
from eval.scoring import score_items

CANDIDATE = filter_deltas(ALL_MIDDLE, strength=0.4, option_weight=3.0)
N_MMLU = 200
N_CHUNKS = 56
N_POOL = 256
N_UNKNOWN = 128


def rounded(metrics):
    return {k: round(v["rate"], 3) for k, v in metrics.items()}


def validation(model, tok):
    items = load_train()
    _, indices = split_validation(len(items))
    subset = [items[i] for i in indices]
    number = lab.start("editor", "Kandidat auf Validierung", "editor/exp15_checks.py", 512)
    base = torch.load(cfg.LOG_DIR / "train_baseline_predictions.pt")
    where = torch.tensor(indices)
    with with_deltas(model, CANDIDATE(model)):
        wide = {c: score_items(model, tok, subset, c) for c in ("C0", "C1")}
        narrow = {c: score_items(model, tok, subset[:57], c) for c in ("C2", "C3")}
    result = {"base": rounded(condition_metrics(subset, {c: base[c][where] for c in ("C0", "C1")})),
              "edit": rounded(condition_metrics(subset, wide)),
              "base57": rounded(condition_metrics(subset[:57], {c: base[c][where[:57]] for c in ("C2", "C3")})),
              "edit57": rounded(condition_metrics(subset[:57], narrow))}
    lab.finish("editor", number, result)
    print(json.dumps(result), flush=True)


def guards(model, tok):
    final = {i["id"] for i in load_mmlu()}
    items = [i for i in load_mmlu(n=1400, seed=1) if i["id"] not in final][:N_MMLU]
    correct = np.array([letter_index(i["answer"]) for i in items])
    number = lab.start("editor", "MMLU und Perplexitaet auf eigenem Material", "editor/exp15_checks.py",
                       2 * (N_MMLU + N_CHUNKS))

    def measure():
        accuracy = rate_ci(score_items(model, tok, items, "C0").numpy() == correct)
        return {"mmlu": round(accuracy["rate"], 3),
                "perplexity": round(perplexity(model, tok, 100_000, (N_CHUNKS + 1) * CHUNK)["perplexity"], 3)}

    result = {"base": measure()}
    with with_deltas(model, CANDIDATE(model)):
        result["edit"] = measure()
    lab.finish("editor", number, result)
    print(json.dumps(result), flush=True)


def unknown_items(model, tok):
    excluded = {i["id"] for i in load_train()}
    excluded |= {json.loads(l)["id"] for l in HANDCHECK_PATH.read_text(encoding="utf-8").splitlines()}
    pool = sorted((i for i in load_arc() if i["id"] not in excluded), key=lambda i: i["id"])
    sample = random.Random(f"{cfg.SEED}:unknown").sample(pool, N_POOL)
    number = lab.start("editor", "Helfen richtige Hinweise bei unbekannten Items noch", "editor/exp15_checks.py",
                       N_POOL + 2 * N_UNKNOWN)
    c0 = score_items(model, tok, sample, "C0").numpy()
    correct = np.array([letter_index(i["answer"]) for i in sample])
    unknown = [item for item, p, a in zip(sample, c0, correct) if p != a][:N_UNKNOWN]
    answers = np.array([letter_index(i["answer"]) for i in unknown])
    base = rate_ci(score_items(model, tok, unknown, "C2").numpy() == answers)
    with with_deltas(model, CANDIDATE(model)):
        edited = rate_ci(score_items(model, tok, unknown, "C2").numpy() == answers)
    result = {"n_unknown": len(unknown), "C2_base": base, "C2_edit": edited}
    lab.finish("editor", number, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    loaded = load_model()
    validation(loaded, tokenizer)
    guards(loaded, tokenizer)
    unknown_items(loaded, tokenizer)
