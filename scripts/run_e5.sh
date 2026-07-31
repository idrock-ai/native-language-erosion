#!/bin/bash
set -euo pipefail
for m in qwen3.5:4b qwen3.5:27b qwen3.6:27b; do
  .venv/bin/python -m src.run --model "$m" --engine ollama \
    --conditions cot,bootstrap --max-tokens 2048 \
    --subjects ona_tili,matematika --out-dir results/e5/mt2048
done
