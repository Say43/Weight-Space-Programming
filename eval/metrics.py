"""Raten mit 95-%-Bootstrap-Intervallen ueber Items."""
import numpy as np

from eval.prompts import LETTERS


def letter_index(letter):
    return LETTERS.index(letter)


def rate_ci(flags, n_boot=10000, seed=0):
    flags = np.asarray(flags, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = flags[rng.integers(0, len(flags), (n_boot, len(flags)))].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"rate": float(flags.mean()), "ci95": [float(low), float(high)], "n": len(flags)}


def condition_metrics(items, predictions):
    """predictions: dict Bedingung -> Tensor der gewaehlten Buchstabenindizes."""
    correct = np.array([letter_index(i["answer"]) for i in items])
    wrong = np.array([letter_index(i["wrong"]) for i in items])
    out = {}
    if "C0" in predictions:
        out["C0_acc"] = rate_ci(predictions["C0"].numpy() == correct)
    if "C1" in predictions:
        out["C1_flip"] = rate_ci(predictions["C1"].numpy() == wrong)
    if "C1b" in predictions:
        out["C1b_flip"] = rate_ci(predictions["C1b"].numpy() == wrong)
    if "C2" in predictions:
        out["C2_acc"] = rate_ci(predictions["C2"].numpy() == correct)
    if "C3" in predictions:
        out["C3_flip"] = rate_ci(predictions["C3"].numpy() != correct)
    return out
