#!/usr/bin/env python3
"""Per-subject accuracy + CoT->DSPy delta across models in a results dir.
Usage: erosion_table.py [results_dir]  (default results/main)"""
import json, glob, sys

d = sys.argv[1] if len(sys.argv) > 1 else "results/main"
SUBJ = ["ona_tili", "tarix", "matematika", "fizika"]

def acc(c, cond, s):
    try:
        return c[cond]["by_subject"][s]["accuracy"]
    except Exception:
        return None

print(f"{'model':14}{'subject':11}{'base':>6}{'cot':>6}{'dspy':>6}{'cot->dspy':>11}")
print("-" * 54)
rows = {}
for f in sorted(glob.glob(f"{d}/*_report.json")):
    r = json.load(open(f)); c = r["conditions"]; m = r["model"]
    for s in SUBJ:
        b, ct, dsp = acc(c, "baseline", s), acc(c, "cot", s), acc(c, "dspy_bootstrap", s)
        if None in (b, ct, dsp):
            continue
        flag = "  <== EROSION" if (dsp - ct) < 0 and s in ("ona_tili", "tarix") else ""
        print(f"{m:14}{s:11}{b:>6.0f}{ct:>6.0f}{dsp:>6.0f}{dsp-ct:>+11.0f}{flag}")
        rows.setdefault(s, []).append(dsp - ct)
    print()

print("=== mean CoT->DSPy delta per subject (across models) ===")
for s in SUBJ:
    if rows.get(s):
        v = rows[s]
        print(f"  {s:11} mean={sum(v)/len(v):+.1f}  (n={len(v)} models, {sum(1 for x in v if x<0)} negative)")
