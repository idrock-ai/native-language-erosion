#!/bin/bash
# E7 (TurkishMMLU generalization), large-model half. Same protocol as scripts/run_e7.sh;
# split out because the three large models need a freshly booted spark-3 (the GB10 driver
# orphans device memory on every model unload -- see results/e6/DECISION.md).
#
# Unlike run_e7.sh this does NOT abort the sweep when one model fails: a leak-induced
# load failure on model 2 should not cost us model 3. Per-model status is logged and the
# script exits nonzero if any model failed.
#
# Resumable: a model whose items file already exists is skipped, so re-running after an
# interruption (reboot, killed session) costs only the models that did not finish. Each
# model is ~2.5 h, so this matters. Set FORCE=1 to re-run everything regardless.
set -uo pipefail

OUT_DIR="${OUT_DIR:-results/e7}"
LOG="$OUT_DIR/e7_large.log"
FORCE="${FORCE:-0}"
MODELS="${MODELS:-qwen3.5:27b qwen3.6:27b gemma4:31b}"
mkdir -p "$OUT_DIR"

failed=()
for m in $MODELS; do
  safe="${m//[\/:]/_}"                       # mirrors src/turkish.py's filename rule
  if [ "$FORCE" != "1" ] && [ -s "$OUT_DIR/${safe}_items.jsonl" ]; then
    echo "
=== $(date -Is) SKIP $m (results already present) ===" >> "$LOG"
    continue
  fi
  echo "
=== $(date -Is) START $m (free: $(free -g | awk '/^Mem:/{print $7}')G) ===" >> "$LOG"
  if .venv/bin/python -m src.turkish --model "$m" --engine ollama \
       --conditions cot,bootstrap,bootstrap_compliant \
       --max-tokens 512 --out-dir "$OUT_DIR" >> "$LOG" 2>&1; then
    echo "
=== $(date -Is) OK $m ===" >> "$LOG"
  else
    rc=$?                                    # capture BEFORE $(date) resets $?
    echo "
=== $(date -Is) FAILED $m (rc=$rc) ===" >> "$LOG"
    failed+=("$m")
  fi
  ollama stop "$m" >> "$LOG" 2>&1 || true   # release ollama's own accounting; the
  sleep 10                                  # driver may still orphan the pages
done

echo "
=== $(date -Is) SWEEP DONE failed=[${failed[*]:-none}] ===" >> "$LOG"
touch "$OUT_DIR/e7_large.done"
[ ${#failed[@]} -eq 0 ]
