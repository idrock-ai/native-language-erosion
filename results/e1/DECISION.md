# E1 decision memo (instrumented decomposition sweep, spark-3/GB10, Ollama Q4_K_M, max_tokens=512)

Run: 6 models x {direct, cot, dspy_bootstrap, dspy_bootstrap_compliant} x 251 paper-split
test items, fully instrumented (raw completions, token usage, finish_reason). Split
verified against the shipped paper traces (analysis/verify_split.py: MATCH).

## Headline numbers

Means over 6 models (ona_tili / matematika):
- direct 32.8 / 53.1
- cot 33.3 / 74.6
- dspy_bootstrap 31.0 / 78.5
- dspy_bootstrap_compliant 33.0 / 80.3

Pooled ona_tili cot->bootstrap: b=65 c=51 (exact McNemar p=0.227 deployment-scored;
p=0.501 knowledge-scored). Flip decomposition: truncation 7 (5 of them gemma4:e4b),
format_drift 0, content 58.

## What held, what changed vs the paper-era run

1. HELD - the format tax is real and demo-length-driven where it appears:
   gemma4:e4b vanilla bootstrap produced 62 truncated / 56 parse-error items across
   subjects (cot: 17/13); its bootstrapped demo payload this run is 3,565 chars
   (paper-era: 3,080). qwen3.5:9b: 22 truncated under vanilla (cot 7), demo payload
   2,528 chars. Models with short demo payloads show ~no format failures.
2. HELD - the one-line fix works and is dominant, not a trade-off: compliant
   bootstrapping recovers ona_tili toward/above cot in every eroding model
   (e4b 25->29, 9b 31->34, 4b 28->31) while math stays equal or better than vanilla
   (mean 80.3 vs 78.5). It also collapses e4b's failure count 62->22.
3. CHANGED - the observational erosion is weaker on this stack and not significant
   at n=600 pooled (paper-era Mac/Ollama: b=75 c=46, p=0.011). The two 27B models
   show no erosion here; qwen3.6:27b erodes -4 via content flips that the format fix
   does not touch (its demos were short).
4. NEW FINDING - run instability is itself evidence for the mechanism: same models,
   same seed, same verified split, but a different serving stack changes which
   demonstrations BootstrapFewShot self-selects (stack-dependent numerics -> different
   correct-sets -> different demo payloads), and the format tax rises and falls with
   the selected payload. The optimizer's harm profile is a property of the
   (model x stack x selected-demos) triple, not of the model alone - strengthening the
   paper's "audit per subject on your deployment stack" recommendation.
5. Interpretation note (dspy behavior, applies to paper-era numbers too): dspy retries
   a failed ChatAdapter parse through JSONAdapter before raising; parse_error rows mean
   BOTH adapters failed, so all failure counts are lower bounds on incidents.

## Protocol consequences

- E2 (budget dose-response) and E3 (demo length x subject) proceed unchanged - they
  manipulate the causal levers directly instead of relying on whatever bootstrap
  happens to select, which this run shows is the right design.
- E4 (powered residual, ~4k paired items) remains the primary endpoint for whether a
  knowledge-side erosion exists beyond the format tax; E1's n=600 content-flip signal
  (58 b vs 50 c) is directionally negative but far from powered.
- The paper must report BOTH runs (paper-era Mac + this GB10 replication) and frame
  the erosion as stack-dependent with a mechanism, not as a universal constant.
