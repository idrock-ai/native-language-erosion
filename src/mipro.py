"""E8: is the differential harm a BootstrapFewShot quirk, or does it survive a different
optimizer?

\\textsc{MIPROv2} differs from \\textsc{BootstrapFewShot} on the axis that matters for our
mechanism: it searches over *instructions* as well as demonstrations, using Bayesian
optimization against a validation set, rather than simply keeping whatever training
examples the model happens to answer correctly. If the native-subject differential is a
property of correctness-based demonstration selection specifically, it should weaken or
vanish here. If it appears anyway, the finding generalizes beyond one teleprompter.

Protocol is deliberately identical to E1 so the two are directly comparable: same split,
same seed, same Q4/Ollama stack, same 512-token budget, same correctness metric, same
251 test items. CoT is re-run in the same process rather than reused from E1, so the
paired contrast cannot be contaminated by session-to-session drift -- and comparing the
fresh CoT to E1's is a free determinism check.

Search budget is set explicitly (auto=None) rather than via auto='light', because
'light' resolves trial counts internally and would make the run's cost unreproducible.

Writes: <out-dir>/<safe>_items.jsonl, <safe>_report.json, <safe>_mipro.json (compiled
program: instructions + demos).
Usage: python -m src.mipro --model qwen3.5:9b --engine ollama --out-dir results/e8
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import dspy
from dspy.teleprompt import MIPROv2

from .data import load_splits, cap_per_subject
from .program import CoTSolver, metric
from .instrument import instrumented_eval
from .run import build_items, make_lm_factory, score_items

REPO = Path(__file__).resolve().parent.parent
MIPRO_CONDITION = "dspy_mipro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--conditions", default="cot,mipro")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-cap", type=int, default=15)
    # search budget -- see module docstring on why this is explicit
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--num-trials", type=int, default=10)
    ap.add_argument("--minibatch-size", type=int, default=25)
    ap.add_argument("--val-cap", type=int, default=100,
                    help="cap the dev split used as MIPROv2's valset (runtime control)")
    ap.add_argument("--out-dir", default="results/e8")
    args = ap.parse_args()

    factory = make_lm_factory(args.engine, args.model, args.api_base, args.max_tokens)
    dspy.configure(lm=factory())
    train, dev, test = load_splits(seed=args.seed)
    train_pool = cap_per_subject(train, args.train_cap)
    random.Random(args.seed).shuffle(train_pool)
    # valset comes from dev, which no other experiment touches -- test stays untouched
    valset = list(dev)
    random.Random(args.seed).shuffle(valset)
    valset = valset[:args.val_cap]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"[{args.model}@{args.engine}] test={len(test)} pool={len(train_pool)} "
          f"val={len(valset)} max_tokens={args.max_tokens} conditions={conds} "
          f"candidates={args.num_candidates} trials={args.num_trials}")

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    report = {"model": args.model, "engine": args.engine, "seed": args.seed,
              "max_tokens": args.max_tokens, "optimizer": "MIPROv2",
              "search": {"num_candidates": args.num_candidates,
                         "num_trials": args.num_trials,
                         "minibatch_size": args.minibatch_size,
                         "val_size": len(valset)},
              "conditions": {}}
    all_items = []

    def record(name, program):
        recs = instrumented_eval(test, program, factory, args.workers, name,
                                 max_tokens=args.max_tokens)
        items = build_items(test, recs, args.model, args.engine, name, args.max_tokens)
        all_items.extend(items)
        report["conditions"][name] = {"deployment": score_items(items, "is_correct"),
                                      "rescue": score_items(items, "rescue_correct")}
        d = report["conditions"][name]["deployment"]["overall"]
        errs = sum(i["parse_error"] for i in items)
        trunc = sum(i["truncated"] for i in items)
        print(f"  {name}: overall={d} parse_errors={errs} truncated={trunc}")

    for cond in conds:
        if cond == "cot":
            record("cot", CoTSolver())
        elif cond == "mipro":
            opt = MIPROv2(
                metric=metric, auto=None, num_candidates=args.num_candidates,
                max_bootstrapped_demos=4, max_labeled_demos=0,
                num_threads=args.workers, seed=args.seed, verbose=False,
            )
            compiled = opt.compile(
                CoTSolver(), trainset=train_pool, valset=valset,
                num_trials=args.num_trials, minibatch=True,
                minibatch_size=args.minibatch_size,
                # unattended: MIPROv2 otherwise blocks on an interactive y/n prompt
                requires_permission_to_run=False,
            )
            compiled.save(str(out / f"{safe}_mipro.json"))
            # What MIPROv2 actually chose matters for interpretation. It searches
            # instructions AND demonstrations, and may settle on zero demos -- in which
            # case the verbose-demo-import mechanism cannot fire by construction, and a
            # differential here would have to come from the instruction instead. Record
            # the choice rather than inferring it later from the saved program.
            demos = getattr(compiled.predict, "demos", []) or []
            instr = compiled.predict.signature.instructions
            report["mipro_program"] = {
                "n_demos": len(demos),
                "demo_payload_chars": sum(len(json.dumps(d, ensure_ascii=False,
                                                         default=str)) for d in demos),
                "instructions": instr,
            }
            # model name included so log lines stay unique per model -- otherwise two
            # models with identical demo counts collide in any dedup-by-line watcher
            print(f"  mipro program [{args.model}]: {len(demos)} demos, "
                  f"{report['mipro_program']['demo_payload_chars']} payload chars")
            record(MIPRO_CONDITION, compiled)
        else:
            raise SystemExit(f"unknown condition: {cond}")

    (out / f"{safe}_report.json").write_text(json.dumps(report, indent=1))
    (out / f"{safe}_items.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in all_items))
    print(f"saved -> {out}/{safe}_*")


if __name__ == "__main__":
    main()
