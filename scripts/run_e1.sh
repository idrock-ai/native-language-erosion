#!/bin/bash
# E1 sweep: run on spark-3 (fastest) or the Mac. ~1000 generations/model.
set -euo pipefail
for m in qwen3.5:4b qwen3.5:9b gemma4:e4b qwen3.5:27b qwen3.6:27b gemma4:31b; do
  .venv/bin/python -m src.run --model "$m" --engine ollama \
    --conditions direct,cot,bootstrap,bootstrap_compliant \
    --max-tokens 512 --out-dir results/e1
done
