#!/bin/bash
# Pull the six paper models on spark-3. Run detached: nohup bash scripts/spark_pull_models.sh > pulls.log 2>&1 &
set -euo pipefail
for m in qwen3.5:4b qwen3.5:9b qwen3.5:27b qwen3.6:27b gemma4:e4b gemma4:31b; do
  ollama pull "$m"
done
ollama list
