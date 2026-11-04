"""Compares the orthogonal projection and the filter against C0 key usage (no model run)."""
import torch

from editor.keyedit import KV_DIM, direction
from eval import config as cfg

keys = torch.load(cfg.LOG_DIR / "keys_all.pt")
usage = torch.load(cfg.LOG_DIR / "usage_cov.pt")
RIDGES = (1e-3, 1e-2, 1e-1)
print("layer.group | leak orthogonal | leak filter per ridge (share of used key energy that the edit changes)")
for (layer, group), cov in usage.items():
    cols = slice(group * KV_DIM, (group + 1) * KV_DIM)
    hint = keys["hint"][:, layer, cols].mean(0).double()
    d = direction(keys, layer, group)
    r = d * (d @ hint)
    total = torch.trace(cov)
    text = f"{layer}.{group} orth {(r @ r) * (d @ cov @ d) / (d @ hint) ** 2 / total * (d @ hint) ** 2 / (r @ r):.5f} |"
    for ridge in RIDGES:
        w = torch.linalg.solve(cov + ridge * total / KV_DIM * torch.eye(KV_DIM, dtype=torch.float64), hint)
        w = w / (w @ hint)
        text += f" {(r @ r) * (w @ cov @ w) / total:.5f}"
    print(text)
