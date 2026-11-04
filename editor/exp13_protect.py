"""Experiment 13: Filter mit Options-Schutz, nur C0 und C1, 128 Items."""
from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import filter_deltas
from editor.screen import run

if __name__ == "__main__":
    run("Options-Keys im Filter schuetzen", "editor/exp13_protect.py",
        {"prot_s0.5": filter_deltas(ALL_MIDDLE, strength=0.5, option_weight=1.0),
         "prot_s0.35": filter_deltas(ALL_MIDDLE, strength=0.35, option_weight=1.0)}, ("C0", "C1"), 128)
