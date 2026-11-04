"""Experiment 11: Filter in Layer 9 bis 18, 128 Items je Bedingung."""
from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import filter_deltas
from editor.screen import run

if __name__ == "__main__":
    run("Filter statt Projektion: Wirkung auf alle Bedingungen", "editor/exp11_filter.py",
        {"filter": filter_deltas(ALL_MIDDLE)}, ("C0", "C1", "C2", "C3"), 128)
