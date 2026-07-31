"""Controlled demonstration-mix experiment (the paper's Table 3 / negative result).

Holds the model, test set, and demonstration quality fixed (all demos are the model's
own correct CoT traces) and varies ONLY the demonstration subject mix:
  cot       no demonstrations (reference)
  skewed    four reasoning-subject demonstrations
  balanced  one demonstration per subject
  native    four knowledge-subject demonstrations

Writes results/controlled/<model>_causal.json. Analyze with analysis/causal_stats.py.
Usage: python -m src.controlled --model qwen3.5:9b
"""
from __future__ import annotations
import argparse, json, random, collections
from pathlib import Path

import dspy

from .data import load_splits, cap_per_subject
from .program import CoTSolver, parse_letter
from .run import eval_condition, score, REPO

REASONING = ("matematika", "fizika")
KNOWLEDGE = ("ona_tili", "tarix")
ALL_SUBJ = ("ona_tili", "tarix", "matematika", "fizika")
MODES = ("cot", "skewed", "balanced", "native")


def make_demo(ex, reasoning):
    return dspy.Example(question=ex.question, options=ex.options, reasoning=reasoning,
                        answer_letter=str(ex.answer_letter).strip().upper()
                        ).with_inputs("question", "options")


def collect_correct_demos(train_pool, workers):
    """Zero-shot CoT over the train pool. Keep the demos the model gets right, by subject."""
    preds = eval_condition(train_pool, CoTSolver(), workers, "collect-demos")
    by = collections.defaultdict(list)
    for ex, p in zip(train_pool, preds):
        ok = str(p["predicted"]).strip().upper() == str(ex.answer_letter).strip().upper()
        if ok and p["reasoning"] and not p["reasoning"].startswith("__ERROR__"):
            by[ex.subject].append(make_demo(ex, p["reasoning"]))
    return by


def select(by, mode, k, seed):
    rng = random.Random(seed)
    pool = {s: list(v) for s, v in by.items()}
    for s in pool:
        rng.shuffle(pool[s])
    if mode == "skewed":
        cand = [d for s in REASONING for d in pool.get(s, [])]
    elif mode == "native":
        cand = [d for s in KNOWLEDGE for d in pool.get(s, [])]
    elif mode == "balanced":
        out, i = [], 0
        while len(out) < k and i < 1000:
            s = ALL_SUBJ[i % 4]; i += 1
            if pool.get(s):
                out.append(pool[s].pop())
        return out
    else:  # cot: no demonstrations
        return []
    rng.shuffle(cand)
    return cand[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-cap", type=int, default=30)
    ap.add_argument("--demos", type=int, default=4)
    ap.add_argument("--out-dir", default="results/controlled")
    args = ap.parse_args()

    dspy.configure(lm=dspy.LM(model=f"ollama_chat/{args.model}", api_base=args.api_base,
                              api_key="ollama", temperature=0.0, max_tokens=args.max_tokens,
                              think=False, num_retries=1))
    train, _dev, test = load_splits(seed=args.seed)
    train_pool = cap_per_subject(train, args.train_cap)
    random.Random(args.seed).shuffle(train_pool)
    print(f"[{args.model}] test={len(test)} train_pool={len(train_pool)}")

    by = collect_correct_demos(train_pool, args.workers)
    print("correct-demo pool: " + "  ".join(f"{s}={len(by.get(s, []))}" for s in ALL_SUBJ))

    report = {"model": args.model, "seed": args.seed, "modes": {}}
    for mode in MODES:
        demos = select(by, mode, args.demos, args.seed)
        solver = CoTSolver(); solver.predict.demos = demos
        preds = eval_condition(test, solver, args.workers, f"{mode}({len(demos)}d)")
        report["modes"][mode] = {"n_demos": len(demos), **score(test, preds)}
        ona = report["modes"][mode]["by_subject"].get("ona_tili", {}).get("accuracy")
        print(f"  {mode:9} demos={len(demos)}  ona_tili={ona}")

    out = REPO / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    (out / f"{safe}_causal.json").write_text(json.dumps(report, indent=1))
    print(f"saved -> {out}/{safe}_causal.json")


if __name__ == "__main__":
    main()
