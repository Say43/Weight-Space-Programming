"""Experiment 1: C0, C1 und C2 an 128 geflippten Fit-Items aufzeichnen."""
import numpy as np
import torch

from baselines.run_b1 import split_validation
from editor import lab
from editor.probe import record
from eval import config as cfg
from eval.measure_train import load_train
from eval.metrics import letter_index
from eval.model import load_model, load_tokenizer
from eval.prompts import render

N = 128
CONDITIONS = ("C0", "C1", "C2")


def main():
    items = load_train()
    base = torch.load(cfg.LOG_DIR / "train_baseline_predictions.pt")
    fit, _ = split_validation(len(items))
    wrong = np.array([letter_index(i["wrong"]) for i in items])
    flipped = [i for i in fit if base["C1"][i].item() == wrong[i]][:N]
    number = lab.start("editor", "Kopier-Heads und Deferenz: Attention und Attribution in C0, C1, C2",
                       "editor/exp01_probe.py", N * len(CONDITIONS))
    tok = load_tokenizer()
    model = load_model(attn="eager")
    data = {c: record(model, tok, [render(tok, items[i], c) for i in flipped]) for c in CONDITIONS}
    torch.save({"indices": flipped, "data": data}, cfg.LOG_DIR / "probe01.pt")
    lab.finish("editor", number, {"saved": "logs_05b/probe01.pt", "items": len(flipped)})
    print("fertig", len(flipped))


if __name__ == "__main__":
    main()
