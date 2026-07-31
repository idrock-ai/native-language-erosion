#!/bin/bash
set -euo pipefail
for m in qwen3.5:4b qwen3.5:9b gemma4:e4b qwen3.5:27b qwen3.6:27b gemma4:31b; do
  .venv/bin/python -m src.residual --model "$m" --out-dir results/e4
done
