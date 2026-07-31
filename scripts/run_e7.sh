#!/bin/bash
# E7 sweep: TurkishMMLU generalization check (does the format-tax mechanism travel to a
# second language?). Run on spark-3 (fastest) or the Mac.
set -euo pipefail
for m in qwen3.5:4b qwen3.5:9b gemma4:e4b qwen3.5:27b qwen3.6:27b gemma4:31b; do
  .venv/bin/python -m src.turkish --model "$m" --engine ollama \
    --conditions cot,bootstrap,bootstrap_compliant \
    --max-tokens 512 --out-dir results/e7
done
