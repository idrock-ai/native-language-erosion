#!/usr/bin/env python3
"""Verify load_splits() reproduces the shipped traces exactly (subject sequence and
gold letters, all six models). Run after any data or loader change. Exits 1 on drift."""
import glob, json, sys
sys.path.insert(0, ".")
from src.data import load_splits

_, _, test = load_splits()
ok = True
files = sorted(glob.glob("results/main/*_traces.jsonl"))
if not files:
    print("VERDICT: MISMATCH — no trace files found (wrong working directory?)")
    sys.exit(1)
for f in files:
    rows = [json.loads(l) for l in open(f)]
    cot = [r for r in rows if r["condition"] == "cot"]
    subj = [e.subject for e in test] == [r["subject"] for r in cot]
    gold = [e.answer_letter for e in test] == [r["correct"] for r in cot]
    print(f"{f.split('/')[-1]:34} subjects={'OK' if subj else 'MISMATCH'} gold={'OK' if gold else 'MISMATCH'}")
    ok = ok and subj and gold
print("VERDICT:", "MATCH — split reproduces the paper" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
