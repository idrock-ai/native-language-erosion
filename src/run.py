"""Instrumented experiment runner (E1/E2/E5 surface).

Conditions:
  direct                no-reasoning answer (paper ladder reference)
  cot                   brevity-constrained zero-shot CoT
  bootstrap             DSPy BootstrapFewShot on CoT (correctness metric)
  bootstrap_compliant   BootstrapFewShot with the compliance-enforcing metric (fix E5a)

Writes: <out-dir>/<safe>_items.jsonl (canonical schema), <safe>_report.json,
        <safe>_bootstrap[.compliant].json (demos).
Usage: python -m src.run --model qwen3.5:9b --engine ollama \
         --conditions direct,cot,bootstrap,bootstrap_compliant --out-dir results/e1
"""
from __future__ import annotations
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot
from tqdm import tqdm

from .data import load_splits, cap_per_subject
from .program import CoTSolver, DirectSolver, metric, compliant_metric, parse_letter
from .instrument import instrumented_eval

REPO = Path(__file__).resolve().parent.parent
CONDITION_NAMES = {"bootstrap": "dspy_bootstrap",
                   "bootstrap_compliant": "dspy_bootstrap_compliant"}


def make_lm_factory(engine, model, api_base, max_tokens):
    if engine == "ollama":
        # cache=False: cached hits null out usage; we need real token counts every call
        return lambda: dspy.LM(model=f"ollama_chat/{model}", api_base=api_base,
                               api_key="ollama", temperature=0.0, cache=False,
                               max_tokens=max_tokens, think=False, num_retries=1)
    # cache=False: cached hits null out usage; we need real token counts every call
    return lambda: dspy.LM(model=f"openai/{model}", api_base=api_base, api_key="EMPTY",
                           temperature=0.0, cache=False, max_tokens=max_tokens,
                           num_retries=1,
                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def build_items(examples, recs, model, engine, condition, max_tokens):
    items = []
    for ex, r in zip(examples, recs):
        gold = str(ex.answer_letter).strip().upper()
        items.append({"model": model, "engine": engine, "condition": condition,
                      "max_tokens": max_tokens, "subject": ex.subject,
                      "qid": getattr(ex, "qid", None), "correct": gold,
                      "is_correct": int(r["predicted"] == gold),
                      "rescue_correct": int(r["rescue_predicted"] == gold), **r})
    return items


def score_items(items, field="is_correct"):
    by = {}
    for it in items:
        b = by.setdefault(it["subject"], {"total": 0, "correct": 0})
        b["total"] += 1
        b["correct"] += it[field]
    tot = sum(b["total"] for b in by.values())
    cor = sum(b["correct"] for b in by.values())
    for b in by.values():
        b["accuracy"] = round(100 * b["correct"] / b["total"], 1) if b["total"] else 0.0
    return {"overall": round(100 * cor / tot, 1) if tot else 0.0, "by_subject": by}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--conditions", default="direct,cot,bootstrap,bootstrap_compliant")
    ap.add_argument("--subjects", default="", help="comma list; empty = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-cap", type=int, default=15)
    ap.add_argument("--out-dir", default="results/e1")
    args = ap.parse_args()

    factory = make_lm_factory(args.engine, args.model, args.api_base, args.max_tokens)
    dspy.configure(lm=factory())
    train, _dev, test = load_splits(seed=args.seed)
    if args.subjects:
        subj = {s.strip() for s in args.subjects.split(",") if s.strip()}
        test = [e for e in test if e.subject in subj]
    train_pool = cap_per_subject(train, args.train_cap)
    import random
    random.Random(args.seed).shuffle(train_pool)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"[{args.model}@{args.engine}] test={len(test)} pool={len(train_pool)} "
          f"max_tokens={args.max_tokens} conditions={conds}")

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    report = {"model": args.model, "engine": args.engine, "seed": args.seed,
              "max_tokens": args.max_tokens, "conditions": {}}
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
        name = CONDITION_NAMES.get(cond, cond)
        if cond == "direct":
            record(name, DirectSolver())
        elif cond == "cot":
            record(name, CoTSolver())
        elif cond in ("bootstrap", "bootstrap_compliant"):
            m = metric if cond == "bootstrap" else compliant_metric
            compiled = BootstrapFewShot(metric=m, max_bootstrapped_demos=4,
                                        max_labeled_demos=0).compile(CoTSolver(),
                                                                     trainset=train_pool)
            suffix = "_bootstrap.json" if cond == "bootstrap" else "_bootstrap_compliant.json"
            compiled.save(str(out / f"{safe}{suffix}"))
            record(name, compiled)
        else:
            raise SystemExit(f"unknown condition: {cond}")

    (out / f"{safe}_report.json").write_text(json.dumps(report, indent=1))
    (out / f"{safe}_items.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in all_items))
    print(f"saved -> {out}/{safe}_*")


if __name__ == "__main__":
    main()


# legacy (paper-era) helpers for controlled.py/onatili_vllm.py
# Frozen paper-era surface, copied verbatim from the pre-instrumentation run.py.
# Do not change and do not use from new code: use instrumented_eval/build_items/
# score_items instead (they capture raw_text, usage, finish_reason, and rescue).

def eval_condition(examples, program, workers, desc):
    def one(ex):
        try:
            with dspy.context(lm=dspy.settings.lm):
                p = program(question=ex.question, options=ex.options)
                return {"predicted": parse_letter(p.answer_letter),
                        "reasoning": (p.reasoning or "")[:2000]}
        except Exception as e:
            return {"predicted": "", "reasoning": f"__ERROR__ {type(e).__name__}"}
    out = [None]*len(examples)
    with ThreadPoolExecutor(max_workers=workers) as ex_:
        for i, r in tqdm(zip(range(len(examples)), ex_.map(one, examples)),
                         total=len(examples), desc=desc, unit="q"):
            out[i] = r
    return out


def score(examples, preds):
    by = {}
    for ex, pr in zip(examples, preds):
        gold = str(ex.answer_letter).strip().upper()
        b = by.setdefault(ex.subject, {"total": 0, "correct": 0})
        b["total"] += 1
        b["correct"] += int(str(pr["predicted"]).strip().upper() == gold)
    tot = sum(b["total"] for b in by.values()); cor = sum(b["correct"] for b in by.values())
    for b in by.values():
        b["accuracy"] = round(100*b["correct"]/b["total"], 1) if b["total"] else 0.0
    return {"overall": round(100*cor/tot, 1) if tot else 0.0, "by_subject": by}
