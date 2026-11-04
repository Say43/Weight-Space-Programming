"""B1 difference-of-means ablation and B2 random control, fixed algorithms without editor judgement."""
import json
import random

import numpy as np
import torch
from torch.nn.functional import normalize

from edits.apply_edit import save_edit
from editor import lab
from editor.tools import ablation_deltas, last_position_activations, with_deltas
from eval import config as cfg
from eval.measure_train import load_train
from eval.metrics import condition_metrics, letter_index
from eval.model import load_model, load_tokenizer
from eval.prompts import render
from eval.scoring import score_items

LAYERS = (2, 5, 8, 11, 14, 17, 20, 23)
VALIDATION_SHARE = 0.2
MAX_FLIPPED = 256
C2_TOLERANCE = 0.03


def split_validation(n, seed=cfg.SEED):
    order = list(range(n))
    random.Random(f"{seed}:b1").shuffle(order)
    cut = int(n * VALIDATION_SHARE)
    return order[cut:], order[:cut]


def choose_layer(rows, base_c2):
    """Lowest C1 flip among layers that cost C2 at most 3 points; otherwise the lowest sum."""
    allowed = [r for r in rows if r["C2_acc"]["rate"] >= base_c2 - C2_TOLERANCE]
    if allowed:
        return min(allowed, key=lambda r: r["C1_flip"]["rate"])
    return min(rows, key=lambda r: r["C1_flip"]["rate"] + base_c2 - r["C2_acc"]["rate"])


def main():
    items = load_train()
    base = torch.load(cfg.LOG_DIR / "train_baseline_predictions.pt")
    fit, validation = split_validation(len(items))
    wrong = np.array([letter_index(i["wrong"]) for i in items])
    flipped = [i for i in fit if base["C1"][i].item() == wrong[i]][:MAX_FLIPPED]
    tok = load_tokenizer()
    model = load_model()
    chosen = [items[i] for i in flipped]

    number = lab.start("baseline", "B1: difference of C1 and C0 at the last position", "baselines/run_b1.py",
                       2 * len(chosen))
    c1 = last_position_activations(model, tok, [render(tok, i, "C1") for i in chosen])
    c0 = last_position_activations(model, tok, [render(tok, i, "C0") for i in chosen])
    directions = normalize((c1 - c0).mean(0), dim=-1)
    torch.save(directions, cfg.LOG_DIR / "b1_directions.pt")
    lab.finish("baseline", number, {"flipped_items": len(chosen)})

    val_items = [items[i] for i in validation]
    index = torch.tensor(validation)
    base_val = condition_metrics(val_items, {c: base[c][index] for c in ("C1", "C2")})
    rows = []
    for layer in LAYERS:
        number = lab.start("baseline", f"B1: ablation from layer {layer} on validation", "baselines/run_b1.py",
                           2 * len(val_items))
        with with_deltas(model, ablation_deltas(model, directions[layer], layer)):
            predictions = {c: score_items(model, tok, val_items, c) for c in ("C1", "C2")}
        row = {"layer": layer, **condition_metrics(val_items, predictions)}
        rows.append(row)
        lab.finish("baseline", number, row)
        print(json.dumps({"layer": layer, "C1": row["C1_flip"]["rate"], "C2": row["C2_acc"]["rate"]}), flush=True)

    best = choose_layer(rows, base_val["C2_acc"]["rate"])
    layer = best["layer"]
    save_edit(ablation_deltas(model, directions[layer], layer), cfg.EDIT_DIR / "b1.safetensors")
    generator = torch.Generator().manual_seed(cfg.SEED)
    random_direction = normalize(torch.randn(model.config.hidden_size, generator=generator), dim=0)
    save_edit(ablation_deltas(model, random_direction, layer), cfg.EDIT_DIR / "b2.safetensors")
    summary = {"base_validation": base_val, "layers": rows, "chosen_layer": layer}
    (cfg.LOG_DIR / "b1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"chosen_layer": layer, "base_C1": base_val["C1_flip"]["rate"],
                      "base_C2": base_val["C2_acc"]["rate"]}))


if __name__ == "__main__":
    main()
