"""Key-Edit in mehreren Layern: Hinweis-Richtung aus den Zeilenbloecken von k_proj entfernen."""
import torch

from eval import config as cfg

KV_DIM = 64
BOTH = (0, 1)


def direction(keys, layer, group):
    """Mittlerer Hinweis-Key, orthogonal zum mittleren Options-Key, normiert."""
    cols = slice(group * KV_DIM, (group + 1) * KV_DIM)
    hint = keys["hint"][:, layer, cols].mean(0).double()
    options = keys["options"][:, layer, cols].mean(0).double()
    d = hint - (hint @ options) / (options @ options) * options
    return d / d.norm()


def multi_key_deltas(spec, strength=1.0):
    """spec: {layer: gruppen}. Delta = -s d d^T W im Block jeder Gruppe, Rang 1 je Gruppe."""
    keys = torch.load(cfg.LOG_DIR / "keys_all.pt")

    def build(model):
        params = dict(model.named_parameters())
        deltas = {}
        for layer, groups in spec.items():
            name = f"model.layers.{layer}.self_attn.k_proj.weight"
            weight = params[name].data
            delta = torch.zeros_like(weight)
            for group in groups:
                rows = slice(group * KV_DIM, (group + 1) * KV_DIM)
                d = direction(keys, layer, group).to(weight.device)
                delta[rows] = (-strength * torch.outer(d, d @ weight[rows].double())).float()
            deltas[name] = delta
        return deltas

    return build


def filter_vector(keys, usage, layer, group, ridge, option_weight=0.0):
    """w = (C + lambda I)^-1 m, normiert auf w^T m = 1; m ist der mittlere Hinweis-Key.

    option_weight > 0 mischt die Kovarianz der Options-Keys zu C, mit diesem Anteil an der Spur von C.
    """
    cols = slice(group * KV_DIM, (group + 1) * KV_DIM)
    hint = keys["hint"][:, layer, cols].mean(0).double()
    cov = usage[(layer, group)]
    if option_weight:
        options = keys["options"][:, layer, cols].double()
        option_cov = options.T @ options / len(options)
        cov = cov + option_weight * torch.trace(cov) / torch.trace(option_cov) * option_cov
    w = torch.linalg.solve(cov + ridge * torch.trace(cov) / KV_DIM * torch.eye(KV_DIM, dtype=torch.float64), hint)
    return w / (w @ hint), hint


def filter_deltas(spec, ridge=0.01, strength=1.0, option_weight=0.0):
    """Schiefe Rang-1-Projektion: Delta = -s r w^T W, mit r = d d^T m."""
    keys = torch.load(cfg.LOG_DIR / "keys_all.pt")
    usage = torch.load(cfg.LOG_DIR / "usage_cov.pt")

    def build(model):
        params = dict(model.named_parameters())
        deltas = {}
        for layer, groups in spec.items():
            name = f"model.layers.{layer}.self_attn.k_proj.weight"
            weight = params[name].data
            delta = torch.zeros_like(weight)
            for group in groups:
                rows = slice(group * KV_DIM, (group + 1) * KV_DIM)
                d = direction(keys, layer, group)
                w, hint = filter_vector(keys, usage, layer, group, ridge, option_weight)
                r = (d * (d @ hint)).to(weight.device)
                delta[rows] = (-strength * torch.outer(r, w.to(weight.device) @ weight[rows].double())).float()
            deltas[name] = delta
        return deltas

    return build
