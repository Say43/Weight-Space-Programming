"""Friert die Kandidaten als Delta-Dateien ein. Berechnet auf der CPU aus den unveraenderten Gewichten."""
import hashlib
import json

from edits.apply_edit import edit_stats, save_edit
from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import filter_deltas
from eval import config as cfg
from eval.model import load_model

STRENGTHS = {"e1": 0.4, "e2": 0.5, "e3": 0.3}
OPTION_WEIGHT = 3.0


def main():
    model = load_model("cpu")
    total = sum(p.numel() for p in model.parameters())
    report = {}
    for name, strength in STRENGTHS.items():
        path = cfg.EDIT_DIR / f"{name}.safetensors"
        save_edit(filter_deltas(ALL_MIDDLE, strength=strength, option_weight=OPTION_WEIGHT)(model), path)
        report[name] = {"strength": strength, **edit_stats(path, total),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (cfg.LOG_DIR / "frozen_candidates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
