"""Experiment 2: Heads kausal stummschalten, nur C1, 64 Items je Kandidat."""
from editor.screen import head_scale_deltas, run


def mute(*heads):
    return lambda model: head_scale_deltas(model, {h: 0.0 for h in heads})


L16_WRITERS = [(16, 7), (16, 11), (16, 12)]
L16_ALL = L16_WRITERS + [(16, 9), (16, 2), (16, 3)]
L20 = [(20, 7), (20, 8), (20, 11), (20, 12)]
EARLY = [(9, 13), (9, 10), (9, 8), (9, 11)]

CANDIDATES = {
    "16_writers": mute(*L16_WRITERS),
    "16_all": mute(*L16_ALL),
    "20": mute(*L20),
    "22.2": mute((22, 2)),
    "9_early": mute(*EARLY),
    "9.13+11.12": mute((9, 13), (11, 12)),
    "18.6": mute((18, 6)),
    "16+20+22": mute(*L16_ALL, *L20, (22, 2)),
}

if __name__ == "__main__":
    run("Kausaltest: welche Heads tragen die Sykophanz wirklich", "editor/exp02_screen.py", CANDIDATES, ("C1",), 64)
