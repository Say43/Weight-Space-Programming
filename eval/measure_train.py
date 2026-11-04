"""Baseline measurement on train (phase 1). Does not use the editor budget."""
import json
import time

import torch

from editor import lab
from eval import config as cfg
from eval.metrics import condition_metrics
from eval.model import load_model, load_tokenizer
from eval.prompts import CONDITIONS
from eval.scoring import score_items

GO_THRESHOLD = 0.30


def load_train():
    return [json.loads(l) for l in cfg.TRAIN_PATH.read_text(encoding="utf-8").splitlines()]


def main():
    items = load_train()
    number = lab.start("measurement", "Baseline measurement, phase 1", "eval/measure_train.py", 4 * len(items))
    tok = load_tokenizer()
    model = load_model()
    begin = time.time()
    predictions = {c: score_items(model, tok, items, c) for c in CONDITIONS}
    result = condition_metrics(items, predictions)
    result["seconds"] = round(time.time() - begin, 1)
    result["go"] = result["C1_flip"]["rate"] >= GO_THRESHOLD
    torch.save(predictions, cfg.LOG_DIR / "train_baseline_predictions.pt")
    lab.finish("measurement", number, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
