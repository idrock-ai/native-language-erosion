#!/usr/bin/env python3
"""Empirical token fertility from logged usage: completion_tokens per reasoning char,
and mean completion tokens per item, by subject x model x condition. Higher tokens/char
means the fixed max_tokens budget buys LESS text for that subject.
Usage: python analysis/fertility.py results/e1 [more dirs...]"""
import glob
import json
import sys

sys.path.insert(0, ".")


def collect_cells(dirs):
    """Aggregate token/char and token/item metrics from items.jsonl files.

    Args:
        dirs: List of directory paths to search for *_items.jsonl files

    Returns:
        dict: Keys are (model, condition, subject) tuples, values are dicts with
              'tok', 'chars', 'n' keys for cumulative tokens, characters, and item count
    """
    cells = {}
    for d in dirs:
        for f in glob.glob(f"{d}/*_items.jsonl"):
            for line in open(f):
                i = json.loads(line)
                txt = i.get("raw_text") or i.get("reasoning") or ""
                ct = i.get("completion_tokens")
                if not txt or not ct:
                    continue
                k = (i["model"], i["condition"], i["subject"])
                c = cells.setdefault(k, {"tok": 0, "chars": 0, "n": 0})
                c["tok"] += ct
                c["chars"] += len(txt)
                c["n"] += 1
    return cells


def main():
    """Print token fertility analysis from collected cells."""
    dirs = sys.argv[1:] or ["results/e1"]
    cells = collect_cells(dirs)

    print(f"{'model':22} {'condition':24} {'subject':11} {'tok/char':>9} {'tok/item':>9}")
    for (m, cond, s), c in sorted(cells.items()):
        print(f"{m:22} {cond:24} {s:11} {c['tok'] / c['chars']:>9.3f} "
              f"{c['tok'] / c['n']:>9.0f}")


if __name__ == "__main__":
    main()
