#!/usr/bin/env python3
"""E2 dose-response: erosion and truncation vs max_tokens.
Reads results/e2/mt{256,1024,2048}/ plus results/e1 (the 512 cell) unless a single
root with mtNNN/ subdirs is given. Usage: python analysis/dose_response.py [root]"""
import argparse, glob, json, sys
sys.path.insert(0, ".")
from src.stats import cochran_armitage


def _load(d):
    items = []
    for f in glob.glob(f"{d}/*_items.jsonl"):
        items += [json.loads(l) for l in open(f)]
    return items


def collect(root, budgets=(256, 512, 1024, 2048), e1_dir=None):
    table = {}
    for mt in budgets:
        d = f"{root}/mt{mt}"
        items = _load(d)
        if not items and mt == 512 and e1_dir:      # 512 cell lives in results/e1
            items = [i for i in _load(e1_dir)]
        for i in items:
            if i["subject"] != "ona_tili":
                continue
            m = table.setdefault(i["model"], {})
            cell = m.setdefault(i["max_tokens"], {"cot_k": 0, "cot_n": 0, "boot_k": 0,
                                                  "boot_n": 0, "trunc_boot": 0})
            if i["condition"] == "cot":
                cell["cot_k"] += i["is_correct"]; cell["cot_n"] += 1
            elif i["condition"] == "dspy_bootstrap":
                cell["boot_k"] += i["is_correct"]; cell["boot_n"] += 1
                cell["trunc_boot"] += int(i["truncated"])
    for m, row in table.items():
        bs = [b for b in budgets if b in row]
        ks = [row[b]["trunc_boot"] for b in bs]
        ns = [max(row[b]["boot_n"], 1) for b in bs]
        _, p = cochran_armitage(ks, ns)
        row["trend_p"] = p
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/e2")
    ap.add_argument("--e1", default="results/e1")
    a = ap.parse_args()
    table = collect(a.root, e1_dir=a.e1)
    for m, row in table.items():
        print(f"== {m} (truncation trend p={row['trend_p']:.4f})")
        for b in sorted(k for k in row if isinstance(k, int)):
            c = row[b]
            ca = 100 * c["cot_k"] / max(c["cot_n"], 1)
            ba = 100 * c["boot_k"] / max(c["boot_n"], 1)
            print(f"  mt={b:<5} cot={ca:5.1f} boot={ba:5.1f} d={ba - ca:+5.1f} "
                  f"trunc(boot)={c['trunc_boot']}")
    json.dump(table, open(f"{a.root}/dose_response.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
