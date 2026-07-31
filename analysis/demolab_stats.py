#!/usr/bin/env python3
"""E3 paired contrasts: each demo mode vs cot on ona_tili; pooled long-vs-short and
native-vs-reason main effects. Holm-corrected exact McNemar.

Prints, in order: a per-model breakdown (b, c, net per model x contrast), a sign-test
line per contrast (how many models show net < 0), then the Holm-adjusted pooled table.
Every contrast is computed once per model via `pairs_for_contrasts`; the pooled numbers
are `pair_stats` applied to the concatenation of those same per-model pairs, so the two
views can never drift apart.
Usage: python analysis/demolab_stats.py [results/e3]"""
import glob, json, sys
sys.path.insert(0, ".")
from src.stats import mcnemar_exact, flips, holm

MODES = ("reason_short", "reason_long", "native_short", "native_long")
CONTRASTS = tuple(f"{m} vs cot" for m in MODES) + ("long vs short", "native vs reason")


def load_model_file(path):
    """One *_items.jsonl -> (model_name, {condition: {qid: is_correct}}), ona_tili only.
    model_name is read from the items themselves (the "model" field), not the filename."""
    by = {}
    model = None
    for line in open(path):
        i = json.loads(line)
        if model is None:
            model = i["model"]
        if i["subject"] == "ona_tili":
            by.setdefault(i["condition"], {})[i["qid"]] = i["is_correct"]
    return model, by


def pairs_for_contrasts(by):
    """The one pairing routine, shared by every model and by the pooled totals: builds the
    (reference, other) paired-outcome lists for all six contrasts from a single model's
    {condition: {qid: is_correct}} mapping. Naming convention "X vs Y" -> pair=(Y, X), so a
    positive net (see pair_stats) always means X net-beats Y."""
    out = {}
    for m in MODES:
        ref, other = by.get("cot", {}), by.get(m, {})
        out[f"{m} vs cot"] = [(ref[q], other[q]) for q in ref if q in other]
    length_pairs = []
    for group in ("reason", "native"):
        ref, other = by.get(f"{group}_short", {}), by.get(f"{group}_long", {})
        length_pairs += [(ref[q], other[q]) for q in ref if q in other]
    out["long vs short"] = length_pairs
    subject_pairs = []
    for length in ("short", "long"):
        ref, other = by.get(f"reason_{length}", {}), by.get(f"native_{length}", {})
        subject_pairs += [(ref[q], other[q]) for q in ref if q in other]
    out["native vs reason"] = subject_pairs
    return out


def pair_stats(pairs):
    """Reduce a paired-outcome list to discordant counts. b = reference right & other
    wrong; c = reference wrong & other right; net = c - b (positive -> "other"/X wins)."""
    b, c = flips(pairs)
    return {"b": b, "c": c, "n": len(pairs), "net": c - b}


def collect_stats(d):
    """Read every *_items.jsonl under `d`. Returns
    {"per_model": {model: {contrast: {b, c, n, net}}}, "pooled": {contrast: {b, c, n, net}}}
    Pooled is `pair_stats` over the concatenation of every model's own pairs for that
    contrast -- i.e. derived from the exact same per-model pairing, never recomputed."""
    per_model = {}
    pooled_pairs = {c: [] for c in CONTRASTS}
    for f in sorted(glob.glob(f"{d}/*_items.jsonl")):
        model, by = load_model_file(f)
        pairs = pairs_for_contrasts(by)
        per_model[model] = {c: pair_stats(pairs[c]) for c in CONTRASTS}
        for c in CONTRASTS:
            pooled_pairs[c] += pairs[c]
    pooled = {c: pair_stats(pooled_pairs[c]) for c in CONTRASTS}
    return {"per_model": per_model, "pooled": pooled}


def print_per_model(stats):
    print("== per-model breakdown (ona_tili) ==")
    print(f"{'model':16} {'contrast':22} {'b':>4} {'c':>4} {'n':>5} {'net':>5}")
    for model, contrasts in stats["per_model"].items():
        for contrast in CONTRASTS:
            s = contrasts[contrast]
            print(f"{model:16} {contrast:22} {s['b']:>4} {s['c']:>4} {s['n']:>5} "
                  f"{s['net']:>+5}")


def print_sign_tests(stats):
    print("== sign test (per-model net direction) ==")
    for contrast in CONTRASTS:
        with_data = [m[contrast]["net"] for m in stats["per_model"].values()
                     if m[contrast]["n"] > 0]
        neg = sum(1 for net in with_data if net < 0)
        print(f"{contrast.replace(' ', '-')}: {neg}/{len(with_data)} models negative")


def print_pooled_table(stats):
    rows, ps = [], []
    for contrast in CONTRASTS:
        s = stats["pooled"][contrast]
        p = mcnemar_exact(s["b"], s["c"])
        rows.append((f"{contrast} (pooled)", s["b"], s["c"], s["n"], p))
        ps.append(p)
    adj = holm(ps)
    print(f"{'contrast':30} {'b':>4} {'c':>4} {'n':>6} {'p':>8} {'holm':>8}")
    for (name, b_, c_, n, p), pa in zip(rows, adj):
        print(f"{name:30} {b_:>4} {c_:>4} {n:>6} {p:>8.4f} {pa:>8.4f}")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/e3"
    stats = collect_stats(d)
    print_per_model(stats)
    print()
    print_sign_tests(stats)
    print()
    print_pooled_table(stats)


if __name__ == "__main__":
    main()
