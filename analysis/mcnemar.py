#!/usr/bin/env python3
"""Paired significance (McNemar) of the REAL BootstrapFewShot erosion on ona_tili,
from the saved per-question traces (cot vs dspy_bootstrap, same questions). This tests
whether the observational -4.8/-9 erosion is real or noise, without re-running.
Usage: mcnemar.py [results_dir]  (default results/main)"""
import json, glob, sys, collections, math

d = sys.argv[1] if len(sys.argv) > 1 else "results/main"
SUBJ = "ona_tili"
A, B = "cot", "dspy_bootstrap"   # compare B vs A


def chi2_1df_sf(x):
    return math.erfc(math.sqrt(x/2.0)) if x > 0 else 1.0


def load(f):
    by = collections.defaultdict(list)
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("subject") == SUBJ:
            by[r["condition"]].append(int(r.get("is_correct", 0)))
    return by


print(f"{'model':14}{'n':>5}{'cot%':>6}{'boot%':>7}{'d':>5}   {'b':>3}{'c':>4}  {'McNemar p':>10}")
print("-"*60)
tb = tc = 0
per = []
for f in sorted(glob.glob(f"{d}/*_traces.jsonl")):
    by = load(f)
    if A not in by or B not in by:
        continue
    a, b_ = by[A], by[B]
    n = min(len(a), len(b_))
    if n == 0:
        continue
    b = sum(1 for i in range(n) if a[i] == 1 and b_[i] == 0)   # cot right, boot wrong
    c = sum(1 for i in range(n) if a[i] == 0 and b_[i] == 1)   # cot wrong, boot right
    tb += b; tc += c
    chi = (abs(b-c)-1)**2/(b+c) if (b+c) > 0 else 0.0
    p = chi2_1df_sf(chi)
    acc_a, acc_b = 100*sum(a[:n])/n, 100*sum(b_[:n])/n
    model = f.split("/")[-1].replace("_traces.jsonl", "")
    per.append((model, n, acc_a, acc_b, b, c, p))
    print(f"{model:14}{n:>5}{acc_a:>6.0f}{acc_b:>7.0f}{acc_b-acc_a:>+5.0f}   {b:>3}{c:>4}  {p:>10.3f}")

print("-"*60)
# pooled McNemar
chi = (abs(tb-tc)-1)**2/(tb+tc) if (tb+tc) > 0 else 0.0
p = chi2_1df_sf(chi)
nneg = sum(1 for r in per if r[3] < r[2])
print(f"POOLED  discordant b(cot>boot)={tb}  c(boot>cot)={tc}  McNemar p={p:.4f}")
print(f"        erosion (boot<cot) in {nneg}/{len(per)} models; "
      f"mean d = {sum(r[3]-r[2] for r in per)/len(per):+.1f}")
print("\nSign test on per-model direction: "
      f"{nneg}/{len(per)} negative, "
      f"p={sum(math.comb(len(per),k) for k in range(nneg,len(per)+1))/2**len(per):.3f} (one-sided)")
