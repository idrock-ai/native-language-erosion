#!/bin/bash
set -euo pipefail
for m in qwen3.5:9b gemma4:e4b gemma4:31b qwen3.6:27b; do
  .venv/bin/python -m src.demolab --model "$m" --out-dir results/e3
done
