#!/usr/bin/env python3
"""E4 pooled residual: dual-scored exact McNemar per test set, plus style metrics on
clean flips (length, hedges, year-tokens, guillemet quotes).
Usage: python analysis/residual_stats.py [results/e4]"""
import glob, json, re, sys
sys.path.insert(0, ".")
from src.stats import mcnemar_exact, flips

HEDGES = re.compile(r"\b(ehtimol|balki|shekilli|bo'lishi mumkin|taxminan|deb hisoblanadi)\b", re.I)
YEARS = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
QUOTES = re.compile(r"[«»]")


def _stat(pairs):
    b_, c_ = flips(pairs)
    return {"b": b_, "c": c_, "n": len(pairs), "p": mcnemar_exact(b_, c_)}


def collect_set(d: str, set_name: str) -> dict:
    """Read every `{d}/*_{set_name}_items.jsonl`, pool cot/dspy_bootstrap pairs by qid
    across all model files. Returns {"deployment": {b,c,n,p}, "knowledge": {b,c,n,p},
    "rescue": {b,c,n,p}, "style": {b_len, c_len, stable_len, b_hedge, b_year, b_quote,
    b_n}}. "deployment" = every paired qid (dual-scored, on is_correct); "knowledge" =
    the subset with no parse_error/truncation on either condition (dual-scored, on
    is_correct); "rescue" = every paired qid, dual-scored on rescue_correct. Style
    metrics are computed only over knowledge-clean pairs, on the boot condition's
    reasoning text."""
    pooled, clean, rescue = [], [], []
    style = {"b_len": [], "c_len": [], "stable_len": [], "b_hedge": 0, "b_year": 0,
             "b_quote": 0, "b_n": 0}
    for f in sorted(glob.glob(f"{d}/*_{set_name}_items.jsonl")):
        items = [json.loads(l) for l in open(f)]
        cot = {i["qid"]: i for i in items if i["condition"] == "cot"}
        boo = {i["qid"]: i for i in items if i["condition"] == "dspy_bootstrap"}
        for q in cot:
            if q not in boo:
                continue
            a, b = cot[q], boo[q]
            pooled.append((a["is_correct"], b["is_correct"]))
            rescue.append((a["rescue_correct"], b["rescue_correct"]))
            bad = a["parse_error"] or b["parse_error"] or a["truncated"] or b["truncated"]
            if not bad:
                clean.append((a["is_correct"], b["is_correct"]))
                if a["is_correct"] and not b["is_correct"]:
                    style["b_len"].append(len(b["reasoning"]))
                    style["b_n"] += 1
                    style["b_hedge"] += bool(HEDGES.search(b["reasoning"]))
                    style["b_year"] += bool(YEARS.search(b["reasoning"]))
                    style["b_quote"] += bool(QUOTES.search(b["reasoning"]))
                elif not a["is_correct"] and b["is_correct"]:
                    style["c_len"].append(len(b["reasoning"]))
                else:
                    style["stable_len"].append(len(b["reasoning"]))
    return {"deployment": _stat(pooled), "knowledge": _stat(clean),
            "rescue": _stat(rescue), "style": style}


def print_set(set_name: str, stats: dict) -> None:
    print(f"== {set_name}")
    for name in ("deployment", "knowledge", "rescue"):
        s = stats[name]
        print(f"  {name:11} b={s['b']:>3} c={s['c']:>3} n={s['n']:>5} "
              f"p={s['p']:.4f}")
    style = stats["style"]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    print(f"  style: mean boot-reasoning chars on b-flips={mean(style['b_len']):.0f} "
          f"c-flips={mean(style['c_len']):.0f} stable={mean(style['stable_len']):.0f}; "
          f"b-flips with hedge={style['b_hedge']}/{style['b_n']} "
          f"year={style['b_year']} quote={style['b_quote']}")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/e4"
    for set_name in ("paper", "replication"):
        print_set(set_name, collect_set(d, set_name))


if __name__ == "__main__":
    main()
