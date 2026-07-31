# Paper

*The Hidden Native-Language Cost of Prompt Optimization in Low-Resource LLMs.*

| File | What it is |
|---|---|
| [`main.pdf`](main.pdf) | Compiled paper — read this. |
| `main.tex` | LaTeX source (ACL/MRL `article` class). |
| `references.bib` | Bibliography. |
| `figures/fig_subject_delta.pdf` / `.png` | Per-subject Δ figure, drawn from `results/main/`. |
| `figures/make_delta_figure.py` | Regenerates the figure from the committed results. |

## Compiling

The source uses the official ACL style files (`acl.sty`, `acl_natbib.bst`), which are
**not** bundled here — download them from the ACL/ARR template
([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)) and drop them
next to `main.tex`, then:

```bash
latexmk -pdf main.tex     # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For anonymous submission, switch `\usepackage[final]{acl}` to `[review]` in `main.tex`.

## Regenerating the figure

```bash
# from the repo root
uv run --with matplotlib python paper/figures/make_delta_figure.py
# or, if matplotlib is already installed:
python paper/figures/make_delta_figure.py
```

The figure reads `results/main/*_report.json` and plots, per DTM subject, the mean change
in accuracy from CoT to `BootstrapFewShot` across all models (per-model points overlaid).
Its numbers are exactly Table 1's: `ona_tili` −4.8, `tarix` −0.7, `fizika` +7.5,
`matematika` +6.1.
