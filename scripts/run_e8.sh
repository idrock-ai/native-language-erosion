#!/bin/bash
# E8: MIPROv2 sweep over all six models, same protocol as E1 (Q4/Ollama, seed 42,
# max_tokens 512, same 251-item test split). Answers the reviewer question E1 cannot:
# is the native-subject differential specific to BootstrapFewShot's correctness-based
# demonstration selection, or does it survive an optimizer that searches instructions?
#
# Resumable (a model with an items file is skipped) and failure-tolerant (one model's
# failure does not abort the sweep) -- same rationale as scripts/run_e7_large.sh.
# FORCE=1 re-runs everything; MODELS="..." narrows the list.
set -uo pipefail

OUT_DIR="${OUT_DIR:-results/e8}"
LOG="$OUT_DIR/e8.log"
FORCE="${FORCE:-0}"
# small models first: they finish fast, so a config error surfaces in minutes rather
# than after a 3-hour 27B compile
MODELS="${MODELS:-qwen3.5:4b qwen3.5:9b gemma4:e4b qwen3.5:27b qwen3.6:27b gemma4:31b}"
mkdir -p "$OUT_DIR"

failed=()
for m in $MODELS; do
  safe="${m//[\/:]/_}"                       # mirrors src/mipro.py's filename rule
  if [ "$FORCE" != "1" ] && [ -s "$OUT_DIR/${safe}_items.jsonl" ]; then
    echo "
=== $(date -Is) SKIP $m (results already present) ===" >> "$LOG"
    continue
  fi
  echo "
=== $(date -Is) START $m (free: $(free -g | awk '/^Mem:/{print $7}')G) ===" >> "$LOG"
  if .venv/bin/python -m src.mipro --model "$m" --engine ollama \
       --max-tokens 512 --out-dir "$OUT_DIR" >> "$LOG" 2>&1; then
    echo "
=== $(date -Is) OK $m ===" >> "$LOG"
  else
    rc=$?                                    # capture BEFORE $(date) resets $?
    echo "
=== $(date -Is) FAILED $m (rc=$rc) ===" >> "$LOG"
    failed+=("$m")
  fi
  ollama stop "$m" >> "$LOG" 2>&1 || true
  sleep 10
done

echo "
=== $(date -Is) SWEEP DONE failed=[${failed[*]:-none}] ===" >> "$LOG"
touch "$OUT_DIR/e8.done"
[ ${#failed[@]} -eq 0 ]
