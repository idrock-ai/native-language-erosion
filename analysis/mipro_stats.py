#!/usr/bin/env python3
"""E8 analysis: did the native-subject differential survive a different optimizer?

The judgement this script exists to make is which models MIPROv2 actually *treated*.
With a light search budget the optimizer sometimes returns the unmodified baseline --
same instruction, zero demonstrations. Such a model cannot speak to whether MIPROv2
harms the native subject, and it must not be scored as though it could.

Detecting that by "zero discordant pairs" is NOT sufficient. On this stack the gemma
models are not run-to-run deterministic even under greedy decoding at temperature 0
(gemma4:31b: 132/251 completions differ when the SAME program is run twice), so an
untreated gemma still yields discordant pairs -- every one of them noise. Untreated
models are therefore identified from the saved program, not from the outcome.

Usage: python analysis/mipro_stats.py [results/e8] [--native ona_tili]
"""
import argparse, glob, json, sys
sys.path.insert(0, ".")
from analysis.interaction import analyse

BASELINE_INSTRUCTION = "O'zbek tilidagi test savoliga javob ber."
COND_A, COND_B = "cot", "dspy_mipro"


def untreated_models(d, baseline=BASELINE_INSTRUCTION):
    """Models where MIPROv2 returned the baseline unchanged: no instruction edit and no
    demonstrations. Returns {model: reason}."""
    out = {}
    for f in sorted(glob.glob(f"{d}/*_report.json")):
        r = json.loads(open(f).read())
        p = r.get("mipro_program")
        if not p:
            continue
        changed = p["instructions"].strip() != baseline.strip()
        if not changed and p["n_demos"] == 0:
            out[r["model"]] = "instruction unchanged and zero demonstrations"
    return out


def program_table(d, baseline=BASELINE_INSTRUCTION):
    rows = []
    for f in sorted(glob.glob(f"{d}/*_report.json")):
        r = json.loads(open(f).read())
        p = r.get("mipro_program")
        if not p:
            continue
        rows.append({"model": r["model"], "n_demos": p["n_demos"],
                     "payload_chars": p["demo_payload_chars"],
                     "instruction_changed": p["instructions"].strip() != baseline.strip()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/e8")
    ap.add_argument("--native", default="ona_tili")
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()

    progs = program_table(a.dir)
    if not progs:
        sys.exit(f"no *_report.json with a mipro_program in {a.dir}")
    untreated = untreated_models(a.dir)

    print(f"=== E8: what MIPROv2 selected ({len(progs)} models)")
    print(f"{'model':14} {'demos':>6} {'payload':>8} {'instruction changed':>20} {'treated':>8}")
    for r in progs:
        print(f"{r['model']:14} {r['n_demos']:>6} {r['payload_chars']:>8} "
              f"{str(r['instruction_changed']):>20} "
              f"{'no' if r['model'] in untreated else 'yes':>8}")
    if untreated:
        print(f"\n{len(untreated)} model(s) UNTREATED -- excluded from the differential, "
              f"because MIPROv2 returned the baseline unchanged:")
        for m, why in untreated.items():
            print(f"   {m}: {why}")

    out = analyse(a.dir, a.native, n_boot=a.boot, cond_a=COND_A, cond_b=COND_B,
                  exclude_models=list(untreated))
    if out is None:
        sys.exit("no treated models left to analyse")

    print(f"\n=== differential on the {out['n_models']} treated model(s)")
    print(f"{'model':14} {'nat L/G':>10} {'nat harm%':>10} {'oth L/G':>10} "
          f"{'oth harm%':>10} {'OR':>7}")
    for m, r in out["per_model"].items():
        nat = "{}/{}".format(r["native_lost"], r["native_gained"])
        oth = "{}/{}".format(r["other_lost"], r["other_gained"])
        print(f"{m:14} {nat:>10} {str(r['native_harm_share']):>10} {oth:>10} "
              f"{str(r['other_harm_share']):>10} {str(r['odds_ratio']):>7}")
    b = out["bootstrap"]
    print(f"\nMantel-Haenszel OR = {out['mh_odds_ratio']}"
          + (f"   item-cluster 95% CI {b['ci95']}   P(OR<=1)={b['p_or_le_1']}"
             if b["ci95"] else "   (OR unidentified)"))
    print(f"direction consistent in {out['models_agreeing']}/{out['models_comparable']} "
          f"treated models")

    out["untreated_models"] = untreated
    out["programs"] = progs
    dest = f"{a.dir}/mipro_stats.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
