"""Records everything at the last position per prompt; the analysis then needs no model."""
import torch

from eval.scoring import encode, right_pad


@torch.inference_mode()
def record(model, tok, texts, batch_size=8):
    device = model.get_output_embeddings().weight.device
    sequences = encode(tok, texts)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    layers = model.model.layers
    store = {}

    def capture(name, layer, take):
        def hook(_module, inputs, output):
            store[(name, layer)] = take(inputs, output)
        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles += [
            layer.self_attn.register_forward_hook(capture("attention", i, lambda _in, out: out[1])),
            layer.self_attn.o_proj.register_forward_hook(capture("heads", i, lambda inp, _out: inp[0])),
            layer.self_attn.o_proj.register_forward_hook(capture("attn_out", i, lambda _in, out: out)),
            layer.mlp.down_proj.register_forward_hook(capture("mlp_out", i, lambda _in, out: out)),
        ]
    handles.append(model.model.norm.register_forward_hook(capture("final_in", 0, lambda inp, _out: inp[0])))
    handles.append(model.model.norm.register_forward_hook(capture("final_out", 0, lambda _in, out: out)))

    n, depth = len(sequences), len(layers)
    hidden = model.config.hidden_size
    out = {key: torch.empty(n, depth, hidden) for key in ("heads", "attn_out", "mlp_out")}
    out["final_in"], out["final_out"] = torch.empty(n, hidden), torch.empty(n, hidden)
    attention = [None] * n
    try:
        for start in range(0, n, batch_size):
            chunk = order[start : start + batch_size]
            ids, mask = right_pad([sequences[i] for i in chunk], tok.pad_token_id, device)
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            rows, last = torch.arange(len(chunk), device=device), mask.sum(1) - 1
            where = torch.tensor(chunk)
            for key in ("heads", "attn_out", "mlp_out"):
                for layer in range(depth):
                    out[key][where, layer] = store[(key, layer)][rows, last].float().cpu()
            out["final_in"][where] = store[("final_in", 0)][rows, last].float().cpu()
            out["final_out"][where] = store[("final_out", 0)][rows, last].float().cpu()
            stacked = torch.stack([store[("attention", l)][rows, :, last] for l in range(depth)], 1).float().cpu()
            for row, item in enumerate(chunk):
                attention[item] = stacked[row, :, :, : len(sequences[item])].clone()
            store.clear()
    finally:
        for handle in handles:
            handle.remove()
    out["attention"] = attention
    out["ids"] = sequences
    return out
