"""Werkzeuge fuer Baselines und Editor: temporaere Edits, Projektionen, Aktivierungen."""
from contextlib import contextmanager

import torch

from eval.scoring import BATCH_SIZE, encode, right_pad

OUTPUT_MATRICES = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


@contextmanager
@torch.no_grad()
def with_deltas(model, deltas):
    """Wendet Deltas an und stellt danach die Originalgewichte bit-identisch wieder her."""
    params = dict(model.named_parameters())
    saved = {name: params[name].data.clone() for name in deltas}
    try:
        for name, delta in deltas.items():
            params[name].add_(delta.to(params[name].device))
        yield model
    finally:
        for name, original in saved.items():
            params[name].copy_(original)


def projection_delta(weight, direction):
    """Delta, das die Einheitsrichtung aus den Ausgaben (Zeilen) einer Matrix entfernt: -d(d^T W)."""
    d = direction.to(weight.device, dtype=torch.float64)
    return (-torch.outer(d, d @ weight.double())).float()


def ablation_deltas(model, direction, first_layer):
    """Gewichts-Orthogonalisierung von o_proj und down_proj aller Layer ab first_layer."""
    params = dict(model.named_parameters())
    deltas = {}
    for layer in range(first_layer, len(model.model.layers)):
        for suffix in OUTPUT_MATRICES:
            name = f"model.layers.{layer}.{suffix}"
            deltas[name] = projection_delta(params[name].data, direction)
    return deltas


@torch.inference_mode()
def last_position_activations(model, tok, texts, batch_size=BATCH_SIZE):
    """Residual-Stream jedes Layers an der letzten Prompt-Position, Form [N, Layer, Hidden]."""
    device = model.get_output_embeddings().weight.device
    sequences = encode(tok, texts)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    layers = model.model.layers
    out = torch.empty(len(sequences), len(layers), model.config.hidden_size)
    captured = {}

    def capture(index):
        def hook(_module, _inputs, output):
            captured[index] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [layer.register_forward_hook(capture(i)) for i, layer in enumerate(layers)]
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            ids, mask = right_pad([sequences[i] for i in chunk], tok.pad_token_id, device)
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            rows = torch.arange(len(chunk), device=device)
            last = mask.sum(1) - 1
            for layer in range(len(layers)):
                out[torch.tensor(chunk), layer] = captured[layer][rows, last].float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return out
