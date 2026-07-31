#!/usr/bin/env python3
"""Significance for the causal demo-mix experiment. For each model + pooled, reports
ona_tili accuracy per mode with a Wilson 95% CI, and two-proportion z-tests for the
key contrasts: skewed vs cot (does the skew hurt?) and balanced vs skewed (does the
fix recover?). Aggregate counts -> two-proportion z-test (modes share the test set,
so this is conservative vs a paired McNemar test)."""
import json, glob, sys, math

d = sys.argv[1] if len(sys.argv) > 1 else "results/controlled"
SUBJ = "ona_tili"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (100*p, 100*(centre-half), 100*(centre+half))


def ztest(k1, n1, k2, n2):
    """Two-proportion z-test; returns (diff_pct, p_two_sided)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1+1/n2))
    if se == 0:
        return (100*(p1-p2), 1.0)
    z = (p1-p2)/se
    # two-sided p via erfc
    pval = math.erfc(abs(z)/math.sqrt(2))
    return (100*(p1-p2), pval)


def counts(report, mode, subj=SUBJ):
    b = report["modes"][mode]["by_subject"].get(subj, {})
    return int(b.get("correct", 0)), int(b.get("total", 0))


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


pool = {m: [0, 0] for m in ("cot", "skewed", "balanced", "native")}
files = sorted(glob.glob(f"{d}/*_causal.json"))
if not files:
    print(f"no causal results in {d}"); sys.exit(0)

print(f"{'model':13} {'mode':9} {'ona%':>6} {'95% CI':>13}   contrasts (vs) p")
print("-"*70)
for f in files:
    r = json.load(open(f)); model = r["model"]
    for m in ("cot", "skewed", "balanced", "native"):
        if m not in r["modes"]:
            continue
        k, n = counts(r, m); pool[m][0]+=k; pool[m][1]+=n
        acc, lo, hi = wilson(k, n)
        print(f"{model:13} {m:9} {acc:>6.1f} [{lo:>4.0f},{hi:>4.0f}]")
    # contrasts
    ks, ns = counts(r, "skewed"); kc, nc = counts(r, "cot"); kb, nb = counts(r, "balanced")
    dsc, psc = ztest(ks, ns, kc, nc)      # skewed - cot
    dbs, pbs = ztest(kb, nb, ks, ns)      # balanced - skewed
    print(f"{'':13} skewed vs cot:      {dsc:+5.1f}  p={psc:.1e} {stars(psc)}")
    print(f"{'':13} balanced vs skewed: {dbs:+5.1f}  p={pbs:.1e} {stars(pbs)}")
    print("-"*70)

print("POOLED across models:")
for m in ("cot", "skewed", "balanced", "native"):
    k, n = pool[m]; acc, lo, hi = wilson(k, n)
    print(f"  {m:9} ona={acc:.1f}% [{lo:.0f},{hi:.0f}]  (n={n})")
dsc, psc = ztest(*pool["skewed"], *pool["cot"])
dbs, pbs = ztest(*pool["balanced"], *pool["skewed"])
print(f"  skewed vs cot:      {dsc:+.1f}  p={psc:.2e} {stars(psc)}")
print(f"  balanced vs skewed: {dbs:+.1f}  p={pbs:.2e} {stars(pbs)}")
