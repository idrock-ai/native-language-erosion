#!/usr/bin/env python3
"""E5 fixes table: ona_tili recovery vs math retention for each mitigation.
Sources: results/e1 (vanilla/compliant/rescue @512), results/e2/mt2048 + results/e5/mt2048.
Usage: python analysis/fixes_table.py"""
import glob, json, sys
sys.path.insert(0, ".")


def _acc(items, cond, subject, field="is_correct"):
    xs = [i[field] for i in items if i["condition"] == cond and i["subject"] == subject]
    return 100 * sum(xs) / len(xs) if xs else None


def recovery(cot, vanilla, fixed):
    """% of the cot->vanilla erosion recovered by the fix (100 if no erosion)."""
    lost = cot - vanilla
    if lost <= 0:
        return 100.0
    return 100 * (fixed - vanilla) / lost


def load_dir(d):
    out = {}
    for f in glob.glob(f"{d}/*_items.jsonl"):
        items = [json.loads(l) for l in open(f)]
        out[items[0]["model"]] = items
    return out


def main():
    e1 = load_dir("results/e1")
    mt2048 = {**load_dir("results/e2/mt2048"), **load_dir("results/e5/mt2048")}
    rows = []
    for m, items in sorted(e1.items()):
        cot_o = _acc(items, "cot", "ona_tili")
        cot_m = _acc(items, "cot", "matematika")
        van_o = _acc(items, "dspy_bootstrap", "ona_tili")
        van_m = _acc(items, "dspy_bootstrap", "matematika")
        fixes = {
            "compliant": (_acc(items, "dspy_bootstrap_compliant", "ona_tili"),
                          _acc(items, "dspy_bootstrap_compliant", "matematika")),
            "rescue": (_acc(items, "dspy_bootstrap", "ona_tili", "rescue_correct"),
                       _acc(items, "dspy_bootstrap", "matematika", "rescue_correct")),
            "budget2048": (_acc(mt2048.get(m, []), "dspy_bootstrap", "ona_tili"),
                           _acc(mt2048.get(m, []), "dspy_bootstrap", "matematika")),
            "compliant+rescue": (
                _acc(items, "dspy_bootstrap_compliant", "ona_tili", "rescue_correct"),
                _acc(items, "dspy_bootstrap_compliant", "matematika", "rescue_correct")),
        }
        row = {"model": m, "cot_ona": cot_o, "vanilla_ona": van_o,
               "cot_math": cot_m, "vanilla_math": van_m, "fixes": {}}
        for name, (fo, fm) in fixes.items():
            if fo is None:
                continue
            gain = van_m - cot_m
            retention = 100.0 if gain <= 0 else 100 * (fm - cot_m) / gain
            row["fixes"][name] = {"ona": fo, "math": fm,
                                  "recovery": recovery(cot_o, van_o, fo),
                                  "retention": retention}
        rows.append(row)
    print(f"{'model':13} {'fix':17} {'ona':>6} {'recov%':>7} {'math':>6} {'reten%':>7}")
    agg = {}
    for r in rows:
        print(f"{r['model']:13} {'vanilla':17} {r['vanilla_ona']:>6.1f} {'':>7} "
              f"{r['vanilla_math']:>6.1f}")
        for name, f in r["fixes"].items():
            agg.setdefault(name, []).append((f["recovery"], f["retention"]))
            print(f"{'':13} {name:17} {f['ona']:>6.1f} {f['recovery']:>7.0f} "
                  f"{f['math']:>6.1f} {f['retention']:>7.0f}")
    print("\nMEANS (success bar: recovery >= 80 and retention >= 90):")
    for name, xs in agg.items():
        rec = sum(x[0] for x in xs) / len(xs)
        ret = sum(x[1] for x in xs) / len(xs)
        verdict = "PASS" if rec >= 80 and ret >= 90 else "miss"
        print(f"  {name:17} recovery={rec:5.1f} retention={ret:5.1f}  {verdict}")
    json.dump(rows, open("results/e5/fixes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
