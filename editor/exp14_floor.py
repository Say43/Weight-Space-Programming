"""Experiment 14: Rauschboden von C0 mit gleich grossem Zufalls-Edit, dazu staerkerer Options-Schutz."""
import torch

from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import KV_DIM, filter_deltas
from editor.screen import run
from eval import config as cfg


def permuted(build):
    """Vertauscht die Zeilen des Deltas je Gruppe: gleiche Norm und gleicher Rang, sinnlose Richtung."""
    generator = torch.Generator().manual_seed(cfg.SEED)

    def wrapped(model):
        deltas = build(model)
        for name, delta in deltas.items():
            for group in (0, 1):
                order = torch.randperm(KV_DIM, generator=generator).to(delta.device) + group * KV_DIM
                delta[group * KV_DIM : (group + 1) * KV_DIM] = delta[order].clone()
        return deltas

    return wrapped


if __name__ == "__main__":
    run("Rauschboden von C0 und staerkerer Options-Schutz", "editor/exp14_floor.py",
        {"zufall_s0.35": permuted(filter_deltas(ALL_MIDDLE, strength=0.35, option_weight=1.0)),
         "prot3_s0.4": filter_deltas(ALL_MIDDLE, strength=0.4, option_weight=3.0)}, ("C0", "C1"), 128)
