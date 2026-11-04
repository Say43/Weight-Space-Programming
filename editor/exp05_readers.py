"""Experiment 5: harm of the early readers 9.13 and 11.12, 128 items per condition."""
from editor.exp02_screen import mute
from editor.screen import run

if __name__ == "__main__":
    run("Early hint readers without a role in C0: effect on all conditions", "editor/exp05_readers.py",
        {"9.13+11.12": mute((9, 13), (11, 12))}, ("C0", "C1", "C2", "C3"), 128)
