"""Experiment 4: record keys and queries of layer 16 in C0, C1, C3."""
import numpy as np
import torch

from baselines.run_b1 import split_validation
from editor import lab
from eval import config as cfg
from eval.measure_train import load_train
from eval.metrics import letter_index
from eval.model import load_model, load_tokenizer
from eval.prompts import render
from eval.scoring import encode, right_pad

N = 64
LAYER = 16
CONDITIONS = ("C0", "C1", "C3")


@torch.inference_mode()
def record_keys(model, tok, texts, layer, batch_size=8):
    device = model.get_output_embeddings().weight.device
    sequences = encode(tok, texts)
    attention = model.model.layers[layer].self_attn
    store = {}
    hooks = [getattr(attention, f"{k}_proj").register_forward_hook(
        lambda _m, _i, out, key=k: store.__setitem__(key, out)) for k in "qkv"]
    keys, values, queries = [None] * len(texts), [None] * len(texts), [None] * len(texts)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            ids, mask = right_pad([sequences[i] for i in chunk], tok.pad_token_id, device)
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            for row, item in enumerate(chunk):
                length = len(sequences[item])
                keys[item] = store["k"][row, :length].float().cpu()
                values[item] = store["v"][row, :length].float().cpu()
                queries[item] = store["q"][row, length - 1].float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
    return {"k": keys, "v": values, "q": queries, "ids": sequences}


def main():
    items = load_train()
    base = torch.load(cfg.LOG_DIR / "train_baseline_predictions.pt")
    fit, _ = split_validation(len(items))
    wrong = np.array([letter_index(i["wrong"]) for i in items])
    flipped = [i for i in fit if base["C1"][i].item() == wrong[i]][:N]
    number = lab.start("editor", "Keys of the hint against keys of the options in layer 16", "editor/exp04_keys.py",
                       N * len(CONDITIONS))
    tok = load_tokenizer()
    model = load_model()
    data = {c: record_keys(model, tok, [render(tok, items[i], c) for i in flipped], LAYER) for c in CONDITIONS}
    torch.save({"indices": flipped, "data": data}, cfg.LOG_DIR / "probe04.pt")
    lab.finish("editor", number, {"saved": "logs_05b/probe04.pt", "items": len(flipped)})
    print("done", len(flipped))


if __name__ == "__main__":
    main()
