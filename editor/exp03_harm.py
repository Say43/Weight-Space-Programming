"""Experiment 3: Schaeden von 16_all an C0, C2 und C3, 128 Items je Bedingung."""
from editor.exp02_screen import L16_ALL, mute
from editor.screen import run

if __name__ == "__main__":
    run("16_all: Wirkung auf alle Bedingungen", "editor/exp03_harm.py", {"16_all": mute(*L16_ALL)},
        ("C0", "C1", "C2", "C3"), 128)
