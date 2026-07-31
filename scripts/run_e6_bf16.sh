#!/bin/bash
# E6 bf16/f16 coverage of the large models on spark-3 (GB10, 121GB unified).
# Attempt order per model:
#   1) vLLM (NVIDIA container for DGX Spark) serving the HF repo at bf16 -> --engine openai
#   2) fallback: Ollama f16 GGUF tag (e.g. qwen3.5:27b-fp16) -> --engine ollama
# Record the engine actually used; it is written into every item row.
set -euo pipefail
MODEL_HF="$1"      # e.g. Qwen/Qwen3.5-27B
MODEL_TAG="$2"     # e.g. qwen3.5:27b-fp16
if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
  .venv/bin/python -m src.run --model "$MODEL_HF" --engine openai \
    --api-base http://localhost:8000/v1 \
    --conditions cot,bootstrap --max-tokens 512 --out-dir results/e6
else
  ollama pull "$MODEL_TAG"
  .venv/bin/python -m src.run --model "$MODEL_TAG" --engine ollama \
    --conditions cot,bootstrap --max-tokens 512 --out-dir results/e6
fi
