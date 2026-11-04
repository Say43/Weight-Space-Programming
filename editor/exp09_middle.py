"""Experiment 9: harm of middle, 128 items per condition."""
from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import multi_key_deltas
from editor.screen import run

if __name__ == "__main__":
    run("middle: effect on all conditions", "editor/exp09_middle.py", {"middle": multi_key_deltas(ALL_MIDDLE)},
        ("C0", "C1", "C2", "C3"), 128)
