"""Paper figure: per-subject accuracy change under BootstrapFewShot.

Reads the committed main results (results/main/*_report.json) and plots, per DTM
subject, the change in accuracy from CoT to DSPy BootstrapFewShot across all models.
The native-language subject (ona_tili) erodes while the reasoning subjects gain --
the paper's central observation, drawn straight from the released results.

Colorblind-safe (Okabe-Ito): erosion (delta<0) orange, gain (delta>0) blue,
per-model points overlaid so the distribution is visible, not just the mean.

Usage:
  uv run --with matplotlib python paper/figures/make_delta_figure.py
  python paper/figures/make_delta_figure.py [results_dir] [out_prefix]
"""
import glob
import json
import sys
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results/main"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/figures/fig_subject_delta"

# Okabe-Ito colorblind-safe pair (diverging by sign) + neutral ink.
EROSION = "#E69F00"  # orange: delta < 0
GAIN = "#0072B2"     # blue:   delta > 0
INK = "#333333"
MUTE = "#888888"

# native -> reasoning, so the plot reads left(erosion) to right(gain).
SUBJECTS = ["ona_tili", "tarix", "fizika", "matematika"]
LABELS = {
    "ona_tili": "ona_tili\n(native)",
    "tarix": "tarix\n(history)",
    "fizika": "fizika\n(physics)",
    "matematika": "matematika\n(math)",
}


def acc(cond, subj):
    bs = cond.get("by_subject") or cond.get("by_category") or {}
    s = bs.get(subj)
    return s["accuracy"] if s else None


def main():
    deltas = {s: [] for s in SUBJECTS}  # subject -> [per-model delta]
    n_models = 0
    for f in sorted(glob.glob(f"{RESULTS}/*_report.json")):
        r = json.load(open(f))
        conds = r["conditions"]
        cot = conds.get("cot")
        boot = conds.get("dspy_bootstrap") or conds.get("bootstrap")
        if not cot or not boot:
            continue
        n_models += 1
        for s in SUBJECTS:
            a0, a1 = acc(cot, s), acc(boot, s)
            if a0 is not None and a1 is not None:
                deltas[s].append(a1 - a0)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs = range(len(SUBJECTS))
    means = [mean(deltas[s]) for s in SUBJECTS]

    for x, s, m in zip(xs, SUBJECTS, means):
        ax.bar(x, m, width=0.62, color=(GAIN if m >= 0 else EROSION),
               alpha=0.85, zorder=2, edgecolor="white", linewidth=0.8)
        # per-model points, slight horizontal jitter by index parity
        pts = deltas[s]
        for i, v in enumerate(pts):
            jitter = (i - (len(pts) - 1) / 2) * 0.045
            ax.scatter(x + jitter, v, s=22, color=INK, alpha=0.7,
                       zorder=3, edgecolor="white", linewidth=0.5)
        ax.annotate(f"{m:+.1f}", (x, m), textcoords="offset points",
                    xytext=(0, 8 if m >= 0 else -14), ha="center",
                    fontsize=9, color=INK, fontweight="bold")

    ax.axhline(0, color=INK, linewidth=1.0, zorder=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[s] for s in SUBJECTS], fontsize=9)
    ax.set_ylabel(r"$\Delta$ accuracy (points)", fontsize=10)
    ax.set_title(
        f"BootstrapFewShot redistributes accuracy across subjects\n"
        f"CoT $\\rightarrow$ BootstrapFewShot on DTM ({n_models} open LLMs)",
        fontsize=10.5)
    # recessive frame
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTE)
    ax.spines["bottom"].set_color(MUTE)
    ax.tick_params(colors=INK)
    ax.margins(x=0.06)

    # direct annotation of the message (identity is never color-alone)
    ax.text(0.0, min(means) - 2.2, "native language erodes",
            color=EROSION, fontsize=8.5, fontweight="bold", ha="left")
    ax.text(len(SUBJECTS) - 1, max(means) + 2.0, "reasoning gains",
            color=GAIN, fontsize=8.5, fontweight="bold", ha="right")

    fig.tight_layout()
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png  (n_models={n_models})")
    print("means:", {s: round(mean(deltas[s]), 2) for s in SUBJECTS})


if __name__ == "__main__":
    main()
