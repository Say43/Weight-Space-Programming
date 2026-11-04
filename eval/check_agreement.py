"""Prueft, ob die lokale GPU auf einer train-Stichprobe dieselben Buchstaben waehlt wie der Kaggle-Lauf."""
import json

import torch

from editor import lab
from eval import config as cfg
from eval.measure_train import load_train
from eval.model import load_model, load_tokenizer
from eval.scoring import score_items

SAMPLE = 40


def main():
    items = load_train()[:SAMPLE]
    remote = torch.load(cfg.LOG_DIR / "train_baseline_predictions.pt")
    number = lab.start("measurement", "Lokale GPU stimmt mit Kaggle ueberein", "eval/check_agreement.py", 2 * SAMPLE)
    tok = load_tokenizer()
    model = load_model()
    result = {}
    for condition in ("C0", "C1"):
        local = score_items(model, tok, items, condition, batch_size=1)
        result[condition] = int((local == remote[condition][:SAMPLE]).sum())
    result["n"] = SAMPLE
    lab.finish("measurement", number, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
