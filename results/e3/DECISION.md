# E3 decision memo (demo length x subject, 4 models, max_tokens=512)

Design: cot reference + 2x2 {reason,native} x {short,long} demos, 4 demos each from the
model's own correct CoT traces (short = brevity-compliant, long = non-compliant).

## Results (pooled exact McNemar on ona_tili, Holm-corrected)

- reason_short vs cot: b=51 c=33, p=0.063 (Holm 0.38) - negative in 4/4 models
- reason_long  vs cot: b=47 c=34, p=0.18 - negative in 4/4 models
- native_short vs cot: b=38 c=33, p=0.64 - mixed (2/4)
- native_long  vs cot: b=42 c=29, p=0.15 - see supply caveat
- long vs short (main effect): b=82 c=79, p=0.87 - NULL in this regime
- native vs reason (main effect): b=78 c=91, p=0.36 - directionally native>reason, n.s.

## Supply caveat (mechanistically informative)

The native/long demo pool was EMPTY for qwen3.6:27b (its native_long condition ran
demo-free, hence b=c=0: identical to cot) and had only ONE demo for gemma4:31b (its
striking -10 rests on a single 513-char tarix demo). Models rarely produce long
native-subject traces at all (pools: 0-7 long native vs 6-42 long reason across
models): the verbose register that inflates outputs is imported from reasoning
subjects. Sensitivity: dropping the two under-supplied cells does not change any
verdict (the degenerate cell contributes zero discordant pairs).

## Interpretation

1. In the MILD-payload regime (model-own demos, <=1.2k chars, budget 512), demo length
   per se has no detectable effect on ona_tili (long-vs-short p=0.87). The format tax
   documented in E1/E2 requires the EXTREME payload x tight budget corner that vanilla
   BootstrapFewShot sometimes lands in (e4b: 3.6k-char demos), plus truncation - which
   E2 manipulates causally.
2. The most consistent depression is reason-subject demos (even short ones): -2..-6 in
   4/4 models, pooled p=0.063 - a small subject-flavored content effect that the
   paper-era controlled experiment also could not distinguish from noise. E4's ~4k
   pairs is the powered test of the content residual.
3. The original paper's "subject mix has no effect" null replicates (native vs reason
   n.s.), now with per-model breakdowns and paired exact tests.
