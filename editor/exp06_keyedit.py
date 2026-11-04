"""Experiment 6: remove the hint direction from the keys of layer 16, full and half strength."""
import torch

from editor.screen import run
from eval import config as cfg

KV_DIM = 64
GROUPS = 2


def hint_key_directions():
    """Per KV group: mean hint key, orthogonal to the mean option key, normalised."""
    keys = torch.load(cfg.LOG_DIR / "keys16.pt")
    directions = []
    for group in range(GROUPS):
        cols = slice(group * KV_DIM, (group + 1) * KV_DIM)
        hint, options = keys["hint"][:, cols].mean(0).double(), keys["options"][:, cols].mean(0).double()
        d = hint - (hint @ options) / (options @ options) * options
        directions.append(d / d.norm())
    return directions


def key_deltas(layer=16, strength=1.0, directions=None):
    """Delta = -s d d^T W in the row block of each KV group of k_proj; rank 1 per group."""
    directions = directions or hint_key_directions()

    def build(model):
        name = f"model.layers.{layer}.self_attn.k_proj.weight"
        weight = dict(model.named_parameters())[name].data
        delta = torch.zeros_like(weight)
        for group, d in enumerate(directions):
            rows = slice(group * KV_DIM, (group + 1) * KV_DIM)
            d = d.to(weight.device)
            delta[rows] = (-strength * torch.outer(d, d @ weight[rows].double())).float()
        return {name: delta}

    return build


if __name__ == "__main__":
    run("Remove the hint key in layer 16", "editor/exp06_keyedit.py",
        {"k16_full": key_deltas(), "k16_half": key_deltas(strength=0.5)}, ("C0", "C1", "C2", "C3"), 64)
