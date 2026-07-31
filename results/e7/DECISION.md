# E7 decision memo (TurkishMMLU generalization; spark-3/GB10, Ollama Q4_K_M, max_tokens=512)

Run: 6 models x {cot, dspy_bootstrap, dspy_bootstrap_compliant} x 260 test items
(4 subjects x 65), fully instrumented. Native subject =
Turkish_Language_and_Literature; reasoning = Mathematics, Physics; knowledge = History.
Same protocol, seed, budget, and serving stack as E1, so language is the only thing
that changes. Q4 deliberately, NOT bf16: E6 showed precision alone flips these models'
signs, so varying language and precision together would be uninterpretable.

## Verdict in one line

The **harm generalizes; our mechanism does not, and our fix does not repair it.**

## 1. Incidence: null, and more clearly null than on DTM

Pooled native-subject exact McNemar, cot -> vanilla bootstrap:
**b=37, c=36, n=390, p=1.0** (knowledge-scored b=35, c=36, p=1.0). Per-model, only
qwen3.5:4b is nominally significant (p=0.021) and it does not survive Holm (0.128).
There is no absolute Turkish native-language erosion.

## 2. Mechanism: does NOT travel

**35 of 37 native losses across the six models are clean content flips** -- valid parse,
`finish_reason=stop`, completion lengths far under the 512 budget (audited item by
item; qwen3.5:27b's eight losses run 72-373 tokens). Only two are truncation-attributable
and none are format drift.

- qwen3.5:27b erodes 7.7pp on native with **flat** truncation counts (1/1/1 across all
  three conditions) and takes **0%** recovery from the compliant fix.
- gemma4:31b produces **zero** truncations and zero parse errors in every condition.
- The erosion-vs-failure rank correlation across models is rho=-0.27, exact permutation
  **p=0.62**. The earlier "truncations rise in 3/3 models" reading is a sign test at
  **p=0.25** -- three coin flips landing the same way, and now 3 of 6 models are tied at
  zero anyway.

The one place the mechanism *does* fire is exactly where it should: qwen3.5:4b has the
largest selected demo payload (4,387 chars, the largest of any model on either
benchmark), the most failures (17 truncated / 15 unparseable under vanilla), the largest
erosion (-15.3pp), and it is the only model the fix helps (failures 17/15 -> 5/5,
recovery 30.1%). So the mechanism's **scope condition is confirmed** -- format tax where
payloads are long, none where they are short -- but on Turkish that corner is reached by
one model out of six, and the remaining harm is content-side and unexplained.

## 3. What DOES travel: the native-vs-non-native differential

Conditional on an item changing correctness, native items flip harmfully far more often
than non-native ones. Estimated with a Mantel-Haenszel odds ratio (each model its own
stratum) and a bootstrap that resamples **items** as whole clusters, since all models
answer the same items (`analysis/interaction.py`):

| dataset | MH OR | 95% CI (item-cluster) | models agreeing |
|---|---|---|---|
| DTM, original stack (6 models) | 2.27 | [1.37, 3.91] | 5/6 |
| DTM, replication stack (6 models) | 1.63 | [0.97, 2.79] | 4/6 |
| **TurkishMMLU (6 models)** | **2.35** | **[1.38, 4.17]** | **5/6** |

Two of three exclude 1, and the one that does not is the replication stack -- the same
stack on which the observational effect also attenuates. That internal consistency is
what makes this more than a lucky contrast.

The estimate was stable as models were added (n=4: 2.57; n=5: 2.54; n=6: 2.35), with the
CI tightening.

**Why the absolute test misses it:** qwen3.6:27b's native accuracy *rose* (60.0 -> 63.1)
yet its OR is 2.4, because its non-native subjects rose much more. In both benchmarks the
native subject is the only one that fails to benefit (Turkish means: History +4.1,
Mathematics +9.7, Physics +7.2, native -0.3).

## 4. The fix costs reasoning accuracy on Turkish

Unlike DTM, where the compliant metric retained 108% of the math gains, on Turkish it
*reduces* mean reasoning-subject accuracy for five of six models (qwen3.5:4b 68.5->61.5,
qwen3.6:27b 84.6->80.8, qwen3.5:9b 68.5->64.6, gemma4:e4b 63.1->62.3,
gemma4:31b 88.5->87.7). It is not the
free lunch it is on DTM. Recommending it as a default needs this caveat.

## 5. Status of these claims

- Incidence (S1) is the **pre-registered** endpoint and is null.
- The differential (S3) is **NOT pre-registered** -- E7 appears nowhere in the design
  spec and this contrast appears nowhere in it. It is the natural operationalization of
  the original paper's own claim, but it is post-hoc and must be reported as secondary,
  with the null primary endpoint stated first.
- No per-model claim is safe: qwen3.5:27b reverses between benchmarks (DTM OR 0.75,
  Turkish 4.00), and the Turkish pooled estimate is driven hard by qwen3.5:4b (13.87).
  Only the pooled, stratified estimate is stable.

## 6. Consequences for the paper

1. Add "Does it travel?" as the last empirical section (after Precision), so the paper
   closes on generalization rather than the bf16 self-correction.
2. The two-regime Discussion must concede a **second, content-side route** to the harm
   that we neither explain nor repair. Option 1 of the framing choice (concede in
   Discussion + Limitations, keep the mechanism-and-fix framing) was chosen.
3. Limitations: "one benchmark, one language" is now false; and "a cluster-robust
   treatment is future work" is now done and should be replaced with the numbers above.
4. The fix recommendation needs the reasoning-cost caveat from S4.

## Reproduce

    python analysis/turkish_stats.py results/e7
    python analysis/interaction.py  results/e7   --native Turkish_Language_and_Literature
    python analysis/interaction.py  results/main --native ona_tili   # original stack
    python analysis/interaction.py  results/e1   --native ona_tili   # replication stack
