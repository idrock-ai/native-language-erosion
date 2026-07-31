#!/bin/bash
# E2: budgets x {cot,bootstrap} x 3 models, ona_tili+matematika only. 512 comes from E1.
set -euo pipefail
for m in qwen3.5:9b gemma4:e4b gemma4:31b; do
  for mt in 256 1024 2048; do
    .venv/bin/python -m src.run --model "$m" --engine ollama \
      --conditions cot,bootstrap --max-tokens "$mt" \
      --subjects ona_tili,matematika --out-dir "results/e2/mt$mt"
  done
done
