"""E3: demo LENGTH x SUBJECT controlled experiment (the spec's missing cell).

Modes: cot, reason_short, reason_long, native_short, native_long
Demos are the model's own correct CoT traces over the train pool, partitioned by
subject group (reason=matematika+fizika, native=ona_tili+tarix) and by compliance
(short = is_compliant, long = not is_compliant).
Usage: python -m src.demolab --model qwen3.5:9b --out-dir results/e3"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import dspy

from .data import load_splits, cap_per_subject
from .program import CoTSolver, is_compliant
from .instrument import instrumented_eval
from .run import make_lm_factory, build_items, score_items

REPO = Path(__file__).resolve().parent.parent
REASON = {"matematika", "fizika"}
MODES = (("reason", "short"), ("reason", "long"), ("native", "short"), ("native", "long"))


def make_demo(ex, reasoning):
    return dspy.Example(question=ex.question, options=ex.options, reasoning=reasoning,
                        answer_letter=str(ex.answer_letter).strip().upper()
                        ).with_inputs("question", "options")


def split_pools(correct_traces):
    """correct_traces: objects with .subject and .reasoning (the model's own trace)."""
    pools = {k: [] for k in MODES}
    for t in correct_traces:
        group = "reason" if t.subject in REASON else "native"
        length = "short" if is_compliant(t.reasoning) else "long"
        pools[(group, length)].append(t)
    return pools


def select_demos(pools, group, length, k=4, seed=42):
    cand = list(pools[(group, length)])
    random.Random(seed).shuffle(cand)
    if len(cand) < k:
        print(f"[demolab] WARNING: only {len(cand)} {group}/{length} demos available")
    return cand[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-cap", type=int, default=40)
    ap.add_argument("--out-dir", default="results/e3")
    args = ap.parse_args()

    factory = make_lm_factory(args.engine, args.model, args.api_base, args.max_tokens)
    dspy.configure(lm=factory())
    train, _dev, test = load_splits(seed=args.seed)
    pool = cap_per_subject(train, args.train_cap)
    random.Random(args.seed).shuffle(pool)

    recs = instrumented_eval(pool, CoTSolver(), factory, args.workers, "collect-demos",
                             max_tokens=args.max_tokens)
    correct = []
    for ex, r in zip(pool, recs):
        ok = r["predicted"] == str(ex.answer_letter).strip().upper()
        if ok and r["reasoning"] and not r["error_type"]:
            d = make_demo(ex, r["reasoning"])
            d.subject = ex.subject
            correct.append(d)
    pools = split_pools(correct)
    print("pool sizes:", {f"{g}/{l}": len(v) for (g, l), v in pools.items()})

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    all_items, meta = [], {"model": args.model, "pools": {f"{g}/{l}": len(v)
                                                          for (g, l), v in pools.items()},
                           "conditions": {}}

    def run_mode(name, demos):
        solver = CoTSolver()
        solver.predict.demos = demos
        recs = instrumented_eval(test, solver, factory, args.workers, name,
                                 max_tokens=args.max_tokens)
        items = build_items(test, recs, args.model, args.engine, name, args.max_tokens)
        all_items.extend(items)
        meta["conditions"][name] = {
            "n_demos": len(demos),
            "demo_chars": [len(d.reasoning) for d in demos],
            "demo_subjects": [d.subject if hasattr(d, "subject") else "?" for d in demos]}
        ona = score_items([i for i in items if i["subject"] == "ona_tili"], "is_correct")
        print(f"  {name:13} demos={len(demos)} ona_tili={ona['overall']}")

    run_mode("cot", [])
    for group, length in MODES:
        run_mode(f"{group}_{length}", select_demos(pools, group, length, 4, args.seed))

    (out / f"{safe}_items.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in all_items))
    (out / f"{safe}_demo_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved -> {out}/{safe}_*")


if __name__ == "__main__":
    main()
