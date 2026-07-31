# E4 decision memo (powered within-subject residual, 6 models, paper + frozen replication sets)

Protocol: bootstrap demos drawn from benchmark ona_tili TRAIN only (within-subject),
evaluated on (i) benchmark ona_tili test (244/model) and (ii) the frozen public
replication set (393/model). Pooled n: paper 1,464 pairs; replication 2,358 pairs.

## Primary endpoint (pre-registered decision rule)

- Paper set: deployment b=137 c=171 (p=0.060); KNOWLEDGE b=129 c=166 (p=0.036) -
  a small nominally significant IMPROVEMENT (c>b; p=0.036 uncorrected — Holm across
  the two test sets gives 0.072), positive delta in 6/6 models (+0.8..+4.9).
- Replication set: deployment b=264 c=260 (p=0.90); knowledge b=250 c=249 (p=1.00) -
  exact null; per-model deltas -2.3..+1.8.
- Style metrics show no assertiveness signature among clean flips (b-flip vs c-flip
  reasoning lengths 354 vs 341 chars; hedge/year/quote rates comparable).

VERDICT: no knowledge-side erosion under within-subject bootstrapping. The spec's
branch taken: the native-language erosion is (almost) entirely a deployment-regime
artifact - it requires the cross-subject demo-import regime (verbose reasoning-subject
demonstrations selected by correctness) meeting fixed decoding budgets (E2's causal
truncation lever), not any intrinsic fragility of the native subject to
demonstrations or optimization.

## Consequence for the original paper's Robustness section

The paper-era vLLM/bf16 result (-7.2, p=0.045, n=250, ONE model, same within-subject
protocol) is overturned by this powered test (n=3,822 pooled, 6 models, two test
sets): treat it as a small-sample fluctuation, to be reported honestly as such.
E6's bf16 runs probe the engine dimension separately.

## Two-regime summary for the paper

- Mixed-subject BootstrapFewShot (deployment default): erosion appears when selected
  demo payloads are extreme and budgets tight; causally budget-sensitive (E2);
  eliminated by the compliant metric (E1, E5) with math gains intact.
- Within-subject bootstrapping: safe to mildly beneficial at scale (this experiment).
