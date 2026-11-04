"""Experiment 12: Filterstaerke, 64 Items je Bedingung und Kandidat."""
from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import filter_deltas
from editor.screen import run

if __name__ == "__main__":
    run("Filterstaerke 0,5 und 0,75", "editor/exp12_strength.py",
        {"s0.5": filter_deltas(ALL_MIDDLE, strength=0.5), "s0.75": filter_deltas(ALL_MIDDLE, strength=0.75)},
        ("C0", "C1", "C2", "C3"), 64)
