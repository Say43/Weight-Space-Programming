"""Experiment 8: Welche Layer tragen den Key-Edit? Nur C1, 64 Items je Kandidat."""
from editor.keyedit import BOTH, multi_key_deltas
from editor.screen import run

READERS = {9: (1,), 11: (1,)}
CORE = {**READERS, 16: BOTH}
WIDE = {**CORE, 13: BOTH, 15: BOTH, 18: BOTH}
ALL_MIDDLE = {layer: BOTH for layer in range(9, 19)}

CANDIDATES = {
    "k9": multi_key_deltas({9: (1,)}),
    "k11": multi_key_deltas({11: (1,)}),
    "k9_11": multi_key_deltas(READERS),
    "core": multi_key_deltas(CORE),
    "core_s2": multi_key_deltas(CORE, strength=2.0),
    "wide": multi_key_deltas(WIDE),
    "middle": multi_key_deltas(ALL_MIDDLE),
    "middle_s2": multi_key_deltas(ALL_MIDDLE, strength=2.0),
}

if __name__ == "__main__":
    run("Key-Edit ueber mehrere Lese-Layer", "editor/exp08_layers.py", CANDIDATES, ("C1",), 64)
