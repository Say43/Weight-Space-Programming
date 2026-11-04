"""Experiment 11: filter in layers 9 to 18, 128 items per condition."""
from editor.exp08_layers import ALL_MIDDLE
from editor.keyedit import filter_deltas
from editor.screen import run

if __name__ == "__main__":
    run("Filter instead of projection: effect on all conditions", "editor/exp11_filter.py",
        {"filter": filter_deltas(ALL_MIDDLE)}, ("C0", "C1", "C2", "C3"), 128)
