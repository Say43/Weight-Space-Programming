"""Experiment 3: harm of 16_all to C0, C2 and C3, 128 items per condition."""
from editor.exp02_screen import L16_ALL, mute
from editor.screen import run

if __name__ == "__main__":
    run("16_all: effect on all conditions", "editor/exp03_harm.py", {"16_all": mute(*L16_ALL)},
        ("C0", "C1", "C2", "C3"), 128)
