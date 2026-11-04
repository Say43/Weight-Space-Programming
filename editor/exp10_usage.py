"""Experiment 10: which key directions does the model use in C0? Attention-weighted covariance per layer and group."""
import torch

from editor import lab
from editor.screen import dev_items
from eval import config as cfg
from eval.model import load_model, load_tokenizer
from eval.prompts import render
from eval.scoring import encode, right_pad

N = 128
KV_DIM = 64
LAYERS = tuple(range(9, 19))
HEADS_PER_GROUP = 7


@torch.inference_mode()
def usage_covariance(model, tok, texts, batch_size=8):
    device = model.get_output_embeddings().weight.device
    sequences = encode(tok, texts)
    store = {}
    hooks = []
    for layer in LAYERS:
        attention = model.model.layers[layer].self_attn
        hooks.append(attention.k_proj.register_forward_hook(
            lambda _m, _i, out, key=("k", layer): store.__setitem__(key, out)))
        hooks.append(attention.register_forward_hook(
            lambda _m, _i, out, key=("a", layer): store.__setitem__(key, out[1])))
    used = {(l, g): torch.zeros(KV_DIM, KV_DIM, dtype=torch.float64, device=device) for l in LAYERS for g in (0, 1)}
    weight = {key: 0.0 for key in used}
    try:
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            ids, mask = right_pad(chunk, tok.pad_token_id, device)
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            for layer in LAYERS:
                keys = store[("k", layer)]
                attention = store[("a", layer)]
                for group in (0, 1):
                    block = keys[:, :, group * KV_DIM : (group + 1) * KV_DIM].double()
                    heads = attention[:, group * HEADS_PER_GROUP : (group + 1) * HEADS_PER_GROUP]
                    # How strongly each key is read in total, over all query positions of this group.
                    reads = (heads.sum(1) * mask[:, :, None]).sum(1).double() * mask
                    used[(layer, group)] += (block * reads[:, :, None]).transpose(1, 2).bmm(block).sum(0)
                    weight[(layer, group)] += reads.sum().item()
            store.clear()
    finally:
        for hook in hooks:
            hook.remove()
    return {key: (value / weight[key]).cpu() for key, value in used.items()}


def main():
    number = lab.start("editor", "Key directions used in C0, for a selective filter",
                       "editor/exp10_usage.py", N)
    tok = load_tokenizer()
    model = load_model(attn="eager")
    _, items = dev_items(N)
    covariance = usage_covariance(model, tok, [render(tok, i, "C0") for i in items])
    torch.save(covariance, cfg.LOG_DIR / "usage_cov.pt")
    lab.finish("editor", number, {"saved": "logs_05b/usage_cov.pt", "items": N})
    print("done", N)


if __name__ == "__main__":
    main()
