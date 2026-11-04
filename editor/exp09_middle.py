"""Experiment 9: Schaeden von middle, 128 Items je Bedingung."""
from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import multi_key_deltas
from editor.screen import run

if __name__ == "__main__":
    run("middle: Wirkung auf alle Bedingungen", "editor/exp09_middle.py", {"middle": multi_key_deltas(ALL_MIDDLE)},
        ("C0", "C1", "C2", "C3"), 128)
