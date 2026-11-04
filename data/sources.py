"""Normalises ARC and MMLU to a common item format."""
import random

from datasets import load_dataset

from eval import config as cfg
from eval.prompts import LETTERS


def false_letter(item_id, answer, n_choices, seed=cfg.SEED):
    """Wrong hint letter, drawn once per item from the seed."""
    rng = random.Random(f"{seed}:{item_id}")
    return rng.choice([l for l in LETTERS[:n_choices] if l != answer])


def _item(item_id, source, question, choices, answer_index):
    answer = LETTERS[answer_index]
    return {"id": item_id, "source": source, "question": question, "choices": list(choices),
            "answer": answer, "wrong": false_letter(item_id, answer, len(choices))}


def load_arc():
    items, seen = [], set()
    for source in ("ARC-Easy", "ARC-Challenge"):
        parts = load_dataset(cfg.ARC_ID, source, revision=cfg.ARC_SHA, token=False)
        for split in ("train", "validation", "test"):
            for row in parts[split]:
                labels = row["choices"]["label"]
                key = row["question"].strip().lower()
                if key in seen or len(labels) > len(LETTERS) or len(labels) < 2:
                    continue
                seen.add(key)
                items.append(_item(row["id"], source, row["question"], row["choices"]["text"],
                                   labels.index(row["answerKey"])))
    return items


def load_mmlu(n=1000, seed=cfg.SEED):
    rows = load_dataset(cfg.MMLU_ID, "all", split="test", revision=cfg.MMLU_SHA, token=False)
    picked = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [_item(f"mmlu-{i}", "MMLU", rows[i]["question"], rows[i]["choices"], rows[i]["answer"]) for i in picked]
