#!/usr/bin/env python3
"""E7 TurkishMMLU generalization analysis: does the DTM format-tax mechanism travel?

Three questions, in the order the paper asks them:
  (1) INCIDENCE  - is there a native-language erosion on Turkish at all?
                   Per-model + pooled exact McNemar (cot -> vanilla bootstrap) on
                   Turkish_Language_and_Literature, dual-scored (deployment / knowledge).
  (2) MECHANISM  - where erosion DOES land, is it the same machine? Truncation and
                   parse-failure counts per condition, and the flip decomposition.
  (3) REPAIR     - does the compliant metric remove the failures and recover accuracy,
                   without costing the reasoning subjects?

Reuses analysis/decompose.py for (1)/(2) pooling so E7 and E1 are scored identically.
Usage: python analysis/turkish_stats.py [results/e7]
"""
import argparse, glob, json, sys
sys.path.insert(0, ".")
from src.stats import mcnemar_exact, wilson, holm, flips
from analysis.decompose import decompose_dir, _pairs

NATIVE = "Turkish_Language_and_Literature"
REASONING = ["Mathematics", "Physics"]
KNOWLEDGE = "History"
CONDS = ["cot", "dspy_bootstrap", "dspy_bootstrap_compliant"]


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*_items.jsonl")):
        items = [json.loads(l) for l in open(f)]
        out[items[0]["model"]] = items
    return out


def acc(items, cond, subject, field="is_correct"):
    xs = [i[field] for i in items if i["condition"] == cond and i["subject"] == subject]
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def failures(items, cond, subject=None):
    """(truncated, parse_error) counts for one condition, optionally one subject."""
    xs = [i for i in items if i["condition"] == cond
          and (subject is None or i["subject"] == subject)]
    return sum(i["truncated"] for i in xs), sum(i["parse_error"] for i in xs)


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p for k successes in n trials (sign test)."""
    from math import comb
    if n == 0:
        return 1.0
    pk = comb(n, k) * p ** k * (1 - p) ** (n - k)
    return min(1.0, sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
                        for i in range(n + 1)
                        if comb(n, i) * p ** i * (1 - p) ** (n - i) <= pk * (1 + 1e-9)))


def spearman(xs, ys):
    """Rank correlation, ties averaged. Descriptive only at these n."""
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def spearman_perm_p(xs, ys):
    """Exact two-sided permutation p for Spearman rho by full enumeration. Honest at the
    n we have (n<=6 -> 720 permutations); refuses above n=8 rather than approximate."""
    import itertools
    n = len(xs)
    if n < 3 or n > 8:
        return None
    obs = spearman(xs, ys)
    if obs is None:
        return None
    hits = tot = 0
    for perm in itertools.permutations(range(n)):
        r = spearman(xs, [ys[i] for i in perm])
        tot += 1
        if r is not None and abs(r) >= abs(obs) - 1e-9:
            hits += 1
    return round(hits / tot, 4)


def analyse(d):
    data = load_dir(d)
    out = {"dir": d, "n_models": len(data), "per_model": {}, "pooled": {}}

    for model, items in data.items():
        row = {"accuracy": {}, "failures": {}}
        for cond in CONDS:
            row["accuracy"][cond] = {s: acc(items, cond, s) for s in
                                     [NATIVE] + REASONING + [KNOWLEDGE]}
            t_all, p_all = failures(items, cond)
            t_nat, p_nat = failures(items, cond, NATIVE)
            row["failures"][cond] = {"truncated": t_all, "parse_error": p_all,
                                     "truncated_native": t_nat, "parse_error_native": p_nat}
        # per-model paired test on the native subject, cot -> vanilla
        prs = _pairs(items, "cot", "dspy_bootstrap", NATIVE)
        b_, c_ = flips([(a["is_correct"], b["is_correct"]) for a, b in prs])
        row["native_mcnemar"] = {"b": b_, "c": c_, "n": len(prs),
                                 "p": round(mcnemar_exact(b_, c_), 4)}
        # repair: fraction of the cot->vanilla native loss returned by the compliant fix,
        # and what that fix does to the reasoning subjects (must not be a trade-off)
        cot_n = row["accuracy"]["cot"][NATIVE]
        van_n = row["accuracy"]["dspy_bootstrap"][NATIVE]
        fix_n = row["accuracy"]["dspy_bootstrap_compliant"][NATIVE]
        lost = cot_n - van_n
        row["repair"] = {
            "native_lost_pp": round(lost, 1),
            "native_recovery_pct": None if lost <= 0 else round(100 * (fix_n - van_n) / lost, 1),
            "reasoning_vanilla": round(sum(row["accuracy"]["dspy_bootstrap"][s]
                                           for s in REASONING) / len(REASONING), 1),
            "reasoning_compliant": round(sum(row["accuracy"]["dspy_bootstrap_compliant"][s]
                                             for s in REASONING) / len(REASONING), 1),
        }
        out["per_model"][model] = row

    # Holm across the per-model native tests
    models = list(out["per_model"])
    ps = [out["per_model"][m]["native_mcnemar"]["p"] for m in models]
    for m, p_adj in zip(models, holm(ps)):
        out["per_model"][m]["native_mcnemar"]["p_holm"] = round(p_adj, 4)

    # pooled McNemar + flip decomposition, scored exactly as E1 is
    dec = decompose_dir(d, subject=NATIVE)
    out["pooled"] = dec["pooled"]
    out["flips_per_model"] = {m: r["flips"] for m, r in dec["per_model"].items()}
    lo, hi, _ = wilson(out["pooled"]["deployment"]["b"],
                       out["pooled"]["deployment"]["b"] + out["pooled"]["deployment"]["c"])
    out["pooled"]["discordant_share_ci"] = [round(lo, 3), round(hi, 3)]

    # (2) the mechanistic link across models: does native accuracy fall exactly where
    # native format failures rise? Descriptive at n=len(data).
    d_acc, d_fail, tags = [], [], []
    for m, r in out["per_model"].items():
        a0 = r["accuracy"]["cot"][NATIVE]
        a1 = r["accuracy"]["dspy_bootstrap"][NATIVE]
        f0 = r["failures"]["cot"]["truncated_native"] + r["failures"]["cot"]["parse_error_native"]
        f1 = (r["failures"]["dspy_bootstrap"]["truncated_native"]
              + r["failures"]["dspy_bootstrap"]["parse_error_native"])
        d_acc.append(round(a1 - a0, 1))
        d_fail.append(f1 - f0)
        tags.append(m)
    out["mechanism_link"] = {
        "models": tags, "delta_native_acc_pp": d_acc, "delta_native_failures": d_fail,
        "spearman_rho": spearman(d_fail, d_acc),
        "spearman_perm_p": spearman_perm_p(d_fail, d_acc),
        "note": f"n={len(tags)} models; the item-level evidence is the flip "
                "decomposition and the per-model failure counts",
    }

    # Does the format PRESSURE travel, independent of whether it costs accuracy? Sign
    # test over models on total (all-subject) failure counts, cot -> vanilla bootstrap.
    up = sum(1 for m in out["per_model"]
             if (out["per_model"][m]["failures"]["dspy_bootstrap"]["truncated"]
                 > out["per_model"][m]["failures"]["cot"]["truncated"]))
    tied = sum(1 for m in out["per_model"]
               if (out["per_model"][m]["failures"]["dspy_bootstrap"]["truncated"]
                   == out["per_model"][m]["failures"]["cot"]["truncated"]))
    n_eff = len(out["per_model"]) - tied
    out["failure_pressure"] = {
        "models_with_more_truncations_under_vanilla": up, "models_tied": tied,
        "n_effective": n_eff, "sign_test_p": round(binom_two_sided(up, n_eff), 4),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/e7")
    a = ap.parse_args()
    out = analyse(a.dir)
    if not out["n_models"]:
        sys.exit(f"no *_items.jsonl in {a.dir}")

    print(f"=== E7 TurkishMMLU, {out['n_models']} model(s), native subject = {NATIVE}\n")
    print(f"{'model':16} {'cot':>6} {'vanilla':>8} {'compliant':>10}   "
          f"{'b':>3} {'c':>3} {'p':>7} {'p_holm':>7}")
    for m, r in out["per_model"].items():
        n = r["accuracy"]
        t = r["native_mcnemar"]
        print(f"{m:16} {n['cot'][NATIVE]:>6} {n['dspy_bootstrap'][NATIVE]:>8} "
              f"{n['dspy_bootstrap_compliant'][NATIVE]:>10}   "
              f"{t['b']:>3} {t['c']:>3} {t['p']:>7} {t['p_holm']:>7}")

    print(f"\n{'model':16} {'condition':22} {'trunc':>6} {'parse':>6} "
          f"{'trunc_nat':>10} {'parse_nat':>10}")
    for m, r in out["per_model"].items():
        for cond in CONDS:
            f = r["failures"][cond]
            print(f"{m:16} {cond:22} {f['truncated']:>6} {f['parse_error']:>6} "
                  f"{f['truncated_native']:>10} {f['parse_error_native']:>10}")

    print(f"\n{'model':16} {'lost_pp':>8} {'recovery%':>10} "
          f"{'reason_van':>11} {'reason_fix':>11}")
    for m, r in out["per_model"].items():
        p = r["repair"]
        rec = "n/a" if p["native_recovery_pct"] is None else p["native_recovery_pct"]
        print(f"{m:16} {p['native_lost_pp']:>8} {str(rec):>10} "
              f"{p['reasoning_vanilla']:>11} {p['reasoning_compliant']:>11}")

    print()
    for k in ("deployment", "knowledge", "rescue"):
        s = out["pooled"][k]
        print(f"POOLED {k:11} b={s['b']:>3} c={s['c']:>3} n={s['n']:>5} "
              f"exact-McNemar p={out['pooled'][k + '_p']}")
    print(f"flips (native, cot-correct -> vanilla-wrong): {out['flips_per_model']}")
    ml = out["mechanism_link"]
    print(f"mechanism link: d_failures={ml['delta_native_failures']} "
          f"d_acc={ml['delta_native_acc_pp']} spearman_rho={ml['spearman_rho']} "
          f"perm_p={ml['spearman_perm_p']} ({ml['note']})")
    fp = out["failure_pressure"]
    print(f"format pressure: vanilla raises truncations in "
          f"{fp['models_with_more_truncations_under_vanilla']}/{fp['n_effective']} models "
          f"({fp['models_tied']} tied), sign-test p={fp['sign_test_p']}")

    json.dump(out, open(f"{a.dir}/turkish_stats.json", "w"), indent=1)
    print(f"\nwrote {a.dir}/turkish_stats.json")


if __name__ == "__main__":
    main()
