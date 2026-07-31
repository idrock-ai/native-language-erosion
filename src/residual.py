"""E4: powered ona_tili-only protocol with instrumentation and a frozen replication set.
Benchmark ona_tili (usable rows) -> shuffle(seed) -> first --n-train are the bootstrap
train pool, the rest are the paper-side test; the public replication set is evaluated
with the SAME compiled program. Usage: python -m src.residual --model qwen3.5:9b"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

from .data import load_raw, normalize_row, to_example, replication_onatili
from .program import CoTSolver, metric
from .instrument import instrumented_eval
from .run import make_lm_factory, build_items, score_items

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="results/e4")
    args = ap.parse_args()

    factory = make_lm_factory(args.engine, args.model, args.api_base, args.max_tokens)
    dspy.configure(lm=factory())

    rng = random.Random(args.seed)
    recs = [normalize_row(r, rng) for r in load_raw() if r.get("subject") == "ona_tili"]
    recs = [r for r in recs if r["usable"]]
    rng.shuffle(recs)
    train = [to_example(r) for r in recs[:args.n_train]]
    paper_test = [to_example(r) for r in recs[args.n_train:]]
    repl_test = replication_onatili()
    print(f"[{args.model}] train={len(train)} paper_test={len(paper_test)} "
          f"replication_test={len(repl_test)}")

    compiled = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4,
                                max_labeled_demos=0).compile(CoTSolver(), trainset=train)
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    compiled.save(str(out / f"{safe}_bootstrap.json"))

    for set_name, test in (("paper", paper_test), ("replication", repl_test)):
        items = []
        for cond, prog in (("cot", CoTSolver()), ("dspy_bootstrap", compiled)):
            r = instrumented_eval(test, prog, factory, args.workers,
                                  f"{set_name}:{cond}", max_tokens=args.max_tokens)
            items += build_items(test, r, args.model, args.engine, cond, args.max_tokens)
        (out / f"{safe}_{set_name}_items.jsonl").write_text(
            "\n".join(json.dumps(i, ensure_ascii=False) for i in items))
        cot = score_items([i for i in items if i["condition"] == "cot"], "is_correct")
        boo = score_items([i for i in items if i["condition"] == "dspy_bootstrap"],
                          "is_correct")
        print(f"  {set_name}: cot={cot['overall']} boot={boo['overall']} "
              f"d={boo['overall'] - cot['overall']:+.1f}")


if __name__ == "__main__":
    main()
