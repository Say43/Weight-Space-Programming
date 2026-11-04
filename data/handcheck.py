"""Manual check on 20 reserved items that never enter train or test (seed 0)."""
import json
import random

import torch

from data.sources import load_arc
from eval import config as cfg
from eval.model import load_model, load_tokenizer
from eval.prompts import LETTERS, render
from eval.scoring import predict, score_texts

HANDCHECK_PATH = cfg.HANDCHECK_PATH
PER_SOURCE = 10


def pick_handcheck(items, seed=cfg.SEED):
    picked = []
    for source in ("ARC-Easy", "ARC-Challenge"):
        pool = sorted((i for i in items if i["source"] == source), key=lambda i: i["id"])
        picked += random.Random(f"{seed}:handcheck:{source}").sample(pool, PER_SOURCE)
    return picked


if __name__ == "__main__":
    items = pick_handcheck(load_arc())
    HANDCHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    HANDCHECK_PATH.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    tok = load_tokenizer()
    model = load_model()
    print("VRAM after loading, GiB:", round(torch.cuda.memory_allocated() / 2**30, 2))
    print("--- Example prompt C1 ---\n" + render(tok, items[0], "C1"))
    print("--- Example prompt C3 ---\n" + render(tok, items[10], "C3"))
    results = {}
    for condition in ("C0", "C1", "C2", "C3"):
        texts = [render(tok, i, condition) for i in items]
        batched = score_texts(model, tok, texts, batch_size=8)
        single = torch.cat([score_texts(model, tok, [t], batch_size=1) for t in texts])
        print(condition, "max |batch - single|:", round((batched - single).abs().max().item(), 4))
        results[condition] = predict(batched, [len(i["choices"]) for i in items])
    print("VRAM peak, GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print("--- item | correct | wrong hint | C0 C1 C2 C3 ---")
    for n, item in enumerate(items):
        row = [LETTERS[results[c][n]] for c in ("C0", "C1", "C2", "C3")]
        print(item["id"][:22].ljust(22), item["answer"], item["wrong"], *row, "|", item["question"][:60])
