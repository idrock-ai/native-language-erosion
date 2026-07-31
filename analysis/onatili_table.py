#!/usr/bin/env python3
"""Table of the ona_tili-only vLLM results: CoT vs BootstrapFewShot per model, McNemar."""
import json, glob, sys

d = sys.argv[1] if len(sys.argv) > 1 else "results/onatili_vllm"
print(f"{'model':24}{'n':>5}{'cot':>7}{'boot':>7}{'delta':>7}{'McNemar p':>11}  sig")
print("-" * 68)
neg = tot = 0
for f in sorted(glob.glob(f"{d}/*_onatili.json")):
    r = json.load(open(f))
    p = r["mcnemar_p"]
    sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."
    d_ = r["delta"]; tot += 1; neg += d_ < 0
    print(f"{r['model']:24}{r['n_test']:>5}{r['cot_acc']:>7.1f}{r['bootstrap_acc']:>7.1f}"
          f"{d_:>+7.1f}{p:>11.4f}  {sig}")
print("-" * 68)
print(f"erosion (delta<0) in {neg}/{tot} models served via vLLM")
