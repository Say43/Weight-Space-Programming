"""Builds report/pareto.png from report/results_05b/*.json and prints the key figures."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import config as cfg

NAMES = {"baseline": "Unedited", "b1": "B1", "b2": "B2", "b3": "B3", "e1": "e1 (primary)", "e2": "e2", "e3": "e3"}
GROUPS = {"Unedited model": (["baseline"], "#1baf7a", "s"),
          "Baselines": (["b1", "b2", "b3"], "#eb6834", "^"),
          "Editor candidates": (["e1", "e2", "e3"], "#2a78d6", "o")}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
OFFSETS = {"baseline": (8, 6), "b1": (8, 6), "b2": (-22, 6), "b3": (8, -12), "e1": (8, 4), "e2": (8, 2), "e3": (8, 2)}


def load(name):
    return json.loads((cfg.RESULT_DIR / f"{name}.json").read_text(encoding="utf-8"))


def panel(axis, results, base, key, label):
    for group, (names, color, marker) in GROUPS.items():
        xs = [100 * (base[key]["rate"] - results[n][key]["rate"]) for n in names]
        ys = [100 * results[n]["C1_flip"]["rate"] for n in names]
        low = [y - 100 * results[n]["C1_flip"]["ci95"][0] for y, n in zip(ys, names)]
        high = [100 * results[n]["C1_flip"]["ci95"][1] - y for y, n in zip(ys, names)]
        axis.errorbar(xs, ys, yerr=[low, high], fmt=marker, color=color, markersize=8, elinewidth=1.5, capsize=0,
                      markeredgecolor=SURFACE, markeredgewidth=1, label=group, zorder=3)
        for x, y, n in zip(xs, ys, names):
            axis.annotate(NAMES[n], (x, y), xytext=OFFSETS[n], textcoords="offset points", fontsize=9, color=INK)
    axis.set_xlabel(label, color=MUTED, fontsize=10)
    axis.grid(True, color="#e8e7e3", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#c9c8c2")


def main():
    results = {n: load(n) for n in NAMES}
    base = results["baseline"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, facecolor=SURFACE)
    panel(axes[0], results, base, "mmlu", "MMLU loss (percentage points)")
    panel(axes[1], results, base, "C0_acc", "C0 loss (percentage points)")
    axes[0].set_xlim(-1.2, 1.6)
    axes[1].set_xlim(-0.5, 10.5)
    axes[0].set_ylim(0, 90)
    axes[0].set_ylabel("C1 flip rate in % (lower is better)", color=MUTED, fontsize=10)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    figure.suptitle("The editor candidates lower the flip rate most, at a small cost in C0",
                    x=0.07, ha="left", fontsize=12, color=INK)
    figure.text(0.07, 0.905, "Test split, 999 items. Bars: 95 % bootstrap interval. Bottom left is best.",
                fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(cfg.ROOT / "report" / "pareto.png", dpi=160, facecolor=SURFACE)
    e1 = results["e1"]
    print("rel C1", 1 - e1["C1_flip"]["rate"] / base["C1_flip"]["rate"],
          "conservative", 1 - e1["C1_flip"]["ci95"][1] / base["C1_flip"]["ci95"][0],
          "rel C1b", 1 - e1["C1b_flip"]["rate"] / base["C1b_flip"]["rate"],
          "PPL %", 100 * (e1["perplexity"]["perplexity"] / base["perplexity"]["perplexity"] - 1),
          "B1 PPL %", 100 * (results["b1"]["perplexity"]["perplexity"] / base["perplexity"]["perplexity"] - 1),
          "B1 rel", 1 - results["b1"]["C1_flip"]["rate"] / base["C1_flip"]["rate"])


if __name__ == "__main__":
    main()
