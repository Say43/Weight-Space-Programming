"""Filters known items (correct in C0, unmodified model) and splits them into train and test.

Prints aggregates only. test.jsonl is written but never printed; only eval/final_eval.py reads it.
"""
import hashlib
import json
import random

from data.handcheck import HANDCHECK_PATH, pick_handcheck
from data.sources import load_arc
from eval import config as cfg
from eval.metrics import letter_index
from eval.model import load_model, load_tokenizer
from eval.scoring import score_items

MAX_PER_SPLIT = 1000
META_PATH = cfg.LOG_DIR / "splits_meta.json"


def split_known(known, seed=cfg.SEED):
    """Splits each stratum 50/50, then cuts both sides proportionally to MAX_PER_SPLIT."""
    train, test = {}, {}
    for source in ("ARC-Easy", "ARC-Challenge"):
        pool = sorted((i for i in known if i["source"] == source), key=lambda i: i["id"])
        random.Random(f"{seed}:split:{source}").shuffle(pool)
        half = len(pool) // 2
        train[source], test[source] = pool[:half], pool[half : 2 * half]
    return _cap(train), _cap(test)


def _cap(by_source):
    total = sum(len(v) for v in by_source.values())
    keep = min(1.0, MAX_PER_SPLIT / total)
    return [i for items in by_source.values() for i in items[: int(len(items) * keep)]]


def write_jsonl(path, items):
    path.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    cfg.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    items = load_arc()
    reserved = {i["id"] for i in pick_handcheck(items)}
    assert reserved == {json.loads(l)["id"] for l in HANDCHECK_PATH.read_text(encoding="utf-8").splitlines()}
    pool = [i for i in items if i["id"] not in reserved]
    tok = load_tokenizer()
    model = load_model()
    predictions = score_items(model, tok, pool, "C0")
    known = [i for i, p in zip(pool, predictions.tolist()) if p == letter_index(i["answer"])]
    train, test = split_known(known)
    meta = {
        "pool": len(pool),
        "c0_acc_pool": len(known) / len(pool),
        "known": {s: sum(i["source"] == s for i in known) for s in ("ARC-Easy", "ARC-Challenge")},
        "train": {s: sum(i["source"] == s for i in train) for s in ("ARC-Easy", "ARC-Challenge")},
        "test": {s: sum(i["source"] == s for i in test) for s in ("ARC-Easy", "ARC-Challenge")},
        "train_sha256": write_jsonl(cfg.TRAIN_PATH, train),
        "test_sha256": write_jsonl(cfg.TEST_PATH, test),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
