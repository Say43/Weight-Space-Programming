"""Edits sind Dicts param_name -> delta. Die kanonischen Gewichte sind immer Basis + Delta auf der CPU."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FORBIDDEN = ("embed_tokens", "lm_head")


def save_edit(deltas, path):
    assert not any(any(f in name for f in FORBIDDEN) for name in deltas), "Embeddings und LM-Head bleiben unveraendert"
    save_file({k: v.detach().cpu().float().contiguous() for k, v in deltas.items()}, str(path))


@torch.no_grad()
def apply_edit(model, path):
    deltas = load_file(str(path))
    params = dict(model.named_parameters())
    for name, delta in deltas.items():
        assert not any(f in name for f in FORBIDDEN), name
        params[name].add_(delta.to(params[name].device))
    return deltas


def weights_hash(model, names):
    """SHA-256 ueber die Rohbytes der genannten Parameter, in sortierter Reihenfolge."""
    params = dict(model.named_parameters())
    h = hashlib.sha256()
    for name in sorted(names):
        h.update(name.encode())
        h.update(params[name].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def edit_stats(path, total_params):
    deltas = load_file(str(path))
    changed = sum(int((d != 0).sum()) for d in deltas.values())
    ranks = {}
    for name, delta in deltas.items():
        ranks[name] = int(torch.linalg.matrix_rank(delta)) if delta.dim() == 2 else None
    return {"matrices": len(deltas), "changed_params": changed, "changed_share": changed / total_params,
            "max_delta_rank": max((r for r in ranks.values() if r is not None), default=0)}


if __name__ == "__main__":
    import argparse

    from eval.model import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    first = load_model("cpu")
    names = list(apply_edit(first, args.path))
    h1 = weights_hash(first, names)
    del first
    second = load_model("cpu")
    apply_edit(second, args.path)
    h2 = weights_hash(second, names)
    print(json.dumps({"edit": str(args.path), "sha256_first": h1, "sha256_second": h2, "bit_identical": h1 == h2}))
    raise SystemExit(0 if h1 == h2 else 1)
