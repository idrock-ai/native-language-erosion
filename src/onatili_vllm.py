#!/usr/bin/env python3
"""ona_tili-only test under vLLM (the paper's Robustness section). Splits the 400
ona_tili questions into train/test, bootstraps demos from ona_tili TRAIN, and compares
CoT vs BootstrapFewShot on ona_tili TEST (n~250) with a paired McNemar test. Serve the
model with vLLM (bf16) and point --api-base at it.
Usage: python -m src.onatili_vllm --model Qwen/Qwen3.5-9B --api-base http://localhost:8000/v1
"""
from __future__ import annotations
import argparse, json, random, math
from pathlib import Path
import dspy
from dspy.teleprompt import BootstrapFewShot

from .data import load_raw, normalize_row, to_example
from .program import CoTSolver, metric
from .run import eval_condition, score, REPO

DATASET = str(REPO / "data" / "DTM_benchmark.json")


def mcnemar(cot, boot):
    n = min(len(cot), len(boot))
    b = sum(1 for i in range(n) if cot[i] == 1 and boot[i] == 0)
    c = sum(1 for i in range(n) if cot[i] == 0 and boot[i] == 1)
    chi = (abs(b-c)-1)**2/(b+c) if (b+c) > 0 else 0.0
    return b, c, (math.erfc(math.sqrt(chi/2)) if chi > 0 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo served by vLLM")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--out-dir", default="results/onatili_vllm")
    args = ap.parse_args()

    dspy.configure(lm=dspy.LM(model=f"openai/{args.model}", api_base=args.api_base,
                              api_key="EMPTY", temperature=0.0, max_tokens=args.max_tokens,
                              num_retries=1,
                              extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
    raw = [r for r in load_raw(DATASET) if r.get("subject") == "ona_tili"]
    rng = random.Random(args.seed)
    records = [normalize_row(r, rng, shuffle=True) for r in raw]
    records = [r for r in records if r["usable"]]
    rng.shuffle(records)
    train = [to_example(r) for r in records[:args.n_train]]
    test = [to_example(r) for r in records[args.n_train:]]
    print(f"[{args.model}] ona_tili train={len(train)} test={len(test)}")

    cot_preds = eval_condition(test, CoTSolver(), args.workers, "cot")
    compiled = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4,
                                max_labeled_demos=0).compile(CoTSolver(), trainset=train)
    boot_preds = eval_condition(test, compiled, args.workers, "bootstrap")

    def ok(preds):
        return [int(str(p["predicted"]).strip().upper() == str(e.answer_letter).strip().upper())
                for p, e in zip(preds, test)]
    cot_ok, boot_ok = ok(cot_preds), ok(boot_preds)
    ca, ba = 100*sum(cot_ok)/len(test), 100*sum(boot_ok)/len(test)
    b, c, p = mcnemar(cot_ok, boot_ok)

    out = REPO / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    (out / f"{safe}_onatili.json").write_text(json.dumps(
        {"model": args.model, "n_test": len(test), "n_train": len(train),
         "cot_acc": round(ca, 1), "bootstrap_acc": round(ba, 1), "delta": round(ba-ca, 1),
         "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(p, 4),
         "cot_correct": cot_ok, "bootstrap_correct": boot_ok}, indent=1))
    print(f"  ona_tili cot={ca:.1f} bootstrap={ba:.1f} delta={ba-ca:+.1f} "
          f"McNemar(b={b},c={c}) p={p:.3f}")


if __name__ == "__main__":
    main()
