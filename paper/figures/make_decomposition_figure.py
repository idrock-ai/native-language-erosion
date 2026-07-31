"""Paper figure: E1 flip decomposition on ona_tili.

Reads results/e1/decomposition.json (written by
`python analysis/decompose.py results/e1`) and plots, per model, a stacked bar
of the three b-flip classes (cot-correct -> bootstrap-wrong, broken down into
truncation / format_drift / content) with a diamond marker for the c-flips
(cot-wrong -> bootstrap-correct) overlaid for comparison.

Colorblind-safe (Okabe-Ito): the three erosion subtypes each get a warm hue
(they are all ways a b-flip happens); the c-flip marker reuses the "gain"
blue from fig_subject_delta so the color story is consistent across paper
figures (blue = a flip toward correct, warm = a flip toward wrong).

Usage:
  uv run --with matplotlib python paper/figures/make_decomposition_figure.py
  python paper/figures/make_decomposition_figure.py [decomposition.json] [out_prefix]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/e1/decomposition.json"
# Defaults to "next to the script" regardless of the caller's cwd.
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fig_decomposition")

# Okabe-Ito colorblind-safe palette.
TRUNCATION = "#E69F00"    # orange
FORMAT_DRIFT = "#D55E00"  # vermillion
CONTENT = "#CC79A7"       # reddish purple
GAIN = "#0072B2"          # blue -- matches fig_subject_delta's gain color
INK = "#333333"
MUTE = "#888888"

CLASSES = ["truncation", "format_drift", "content"]
COLORS = {"truncation": TRUNCATION, "format_drift": FORMAT_DRIFT, "content": CONTENT}
LABELS = {"truncation": "truncation", "format_drift": "format drift", "content": "content"}


def main():
    if not os.path.exists(IN_PATH):
        print(f"error: {IN_PATH} not found -- run "
              f"`python analysis/decompose.py results/e1` first", file=sys.stderr)
        sys.exit(1)

    data = json.load(open(IN_PATH))
    per_model = data.get("per_model", {})
    models = sorted(per_model)
    if not models:
        print(f"error: {IN_PATH} has no per-model data", file=sys.stderr)
        sys.exit(1)

    # Sized close to an ACL \columnwidth (~3.2in) so LaTeX barely rescales it; a wide
    # figure squeezed into one column shrinks the tick labels until they collide.
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    xs = range(len(models))

    for x, m in enumerate(models):
        row = per_model[m]
        flips = row["flips"]
        bottom = 0
        for cls in CLASSES:
            v = flips.get(cls, 0)
            ax.bar(x, v, bottom=bottom, width=0.6, color=COLORS[cls], alpha=0.85,
                   zorder=2, edgecolor="white", linewidth=0.8,
                   label=LABELS[cls] if x == 0 else None)
            bottom += v
        ax.scatter(x, row["c"], marker="D", s=26, color=GAIN, zorder=3,
                   edgecolor="white", linewidth=0.6,
                   label="wrong $\\rightarrow$ right (c)" if x == 0 else None)
        # above the bar OR the c-marker, whichever is higher: the marker sits above the
        # bar whenever c > b, and a fixed bar-top offset collides with it there.
        ax.annotate(f"b={row['b']}", (x, max(bottom, row["c"])),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=6.5, color=INK, fontweight="bold")

    ax.set_xticks(list(xs))
    # rotated + right-anchored: six "qwen3.5:27b"-length labels cannot sit flat in one
    # column at a legible size
    ax.set_xticklabels(models, fontsize=6.5, rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("flipped items (count)", fontsize=7.5)
    # No in-figure title: the LaTeX caption already states the contrast, and the two
    # lines it cost were forcing the legend outside the axes.

    # recessive frame
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTE)
    ax.spines["bottom"].set_color(MUTE)
    ax.tick_params(colors=INK)
    ax.margins(x=0.08, y=0.30)   # headroom for the legend strip above the bars
    # Horizontal strip above the axes, in the space the title used to occupy: keeps the
    # image narrow (so LaTeX does not rescale and shrink the tick labels) and still
    # cannot collide with a bar whatever the height ordering turns out to be.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5,
              frameon=False, borderaxespad=0, columnspacing=1.2, handletextpad=0.5)

    fig.tight_layout()
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png  (models={len(models)})")


if __name__ == "__main__":
    main()
