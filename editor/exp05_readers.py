"""Experiment 5: Schaeden der fruehen Leser 9.13 und 11.12, 128 Items je Bedingung."""
from editor.exp02_screen import mute
from editor.screen import run

if __name__ == "__main__":
    run("Fruehe Hinweis-Leser ohne Rolle in C0: Wirkung auf alle Bedingungen", "editor/exp05_readers.py",
        {"9.13+11.12": mute((9, 13), (11, 12))}, ("C0", "C1", "C2", "C3"), 128)
