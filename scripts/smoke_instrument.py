#!/usr/bin/env python3
"""Live smoke: 6 real generations through instrumented_eval against a served model.
Usage: python scripts/smoke_instrument.py --model qwen3.5:4b [--api-base http://localhost:11434]"""
import argparse, json, sys
sys.path.insert(0, ".")
import dspy
from src.data import load_splits
from src.program import CoTSolver
from src.instrument import instrumented_eval

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3.5:4b")
ap.add_argument("--api-base", default="http://localhost:11434")
ap.add_argument("--max-tokens", type=int, default=512)
a = ap.parse_args()

factory = lambda: dspy.LM(model=f"ollama_chat/{a.model}", api_base=a.api_base,
                          api_key="ollama", temperature=0.0, max_tokens=a.max_tokens,
                          think=False, num_retries=1)
dspy.configure(lm=factory())
_, _, test = load_splits()
recs = instrumented_eval(test[:6], CoTSolver(), factory, workers=3, desc="smoke",
                         max_tokens=a.max_tokens)
for r in recs:
    print(json.dumps({k: r[k] for k in ("predicted", "finish_reason", "prompt_tokens",
                                        "completion_tokens", "parse_error")}))
missing_raw = sum(1 for r in recs if not r["raw_text"])
missing_fin = sum(1 for r in recs if not r["finish_reason"])
print(f"raw_text missing: {missing_raw}/6, finish_reason missing: {missing_fin}/6")
sys.exit(0 if missing_raw == 0 else 1)
