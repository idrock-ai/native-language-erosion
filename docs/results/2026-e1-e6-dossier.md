# Results dossier: erosion mechanism + fix (E1-E6)

Date: 2026-07-28.
Every number below regenerates from committed per-item logs via analysis/ scripts;
per-experiment DECISION.md files under results/e*/ carry the full detail.

## Executive summary

The paper-era finding ("BootstrapFewShot erodes native-language accuracy, p=0.011")
resolves into a sharper, better-supported story:

1. THE EROSION IS A DEPLOYMENT-REGIME ARTIFACT, NOT A PROPERTY OF THE NATIVE
   LANGUAGE. Mixed-subject bootstrapping sometimes selects verbose reasoning-subject
   demonstrations; under fixed decoding budgets these inflate responses into
   truncation/parse failures concentrated on the subject with the longest inputs
   (ona_tili). Within-subject bootstrapping shows NO erosion at scale.
2. THE FORMAT COMPONENT IS CAUSAL AND FIXABLE. Budget dose-response confirms
   truncation causality (E2); a one-line constraint-consistent metric plus rescue
   parsing recovers 85.6% of the erosion while keeping 108% of the math gains (E5).
3. OBSERVATIONAL CELLS ARE UNSTABLE ACROSS STACK AND PRECISION. Same model, same
   seed, same split: the cot->bootstrap delta changes sign across serving stacks
   (E1 vs paper-era) and across Q4/bf16 (E6, 3/3 models) because numerics change
   the correct-set, hence the self-selected demos. Single-cell observational claims
   (including the original paper's) are unreliable at n~100; the paper's claims now
   rest on causal and powered designs.

## Per-experiment verdicts (prediction -> outcome)

- E1 decomposition (6 models x 4 conditions, instrumented): format tax reproduces
  where demo payloads are long (e4b: 62 truncated/56 parse errors under vanilla vs
  22/20 under the fix; demo payload 3.6k chars); pooled erosion directionally
  present but n.s. on this stack (b=65 c=51, p=0.227). Ladder (means): direct 32.8 /
  cot 33.3 / vanilla 31.0 / compliant 33.0 on ona_tili; math 53.1/74.6/78.5/80.3.
- E2 budget dose-response: CONFIRMED causally. e4b truncations 28->16->3->0 across
  256->2048 (trend p<1e-4), 9b p=0.016, control flat 0. Budget-insensitive residual
  -3..-5 remains in eroding models -> isolates the content component.
- E3 demo length x subject (the missing controlled cell): length null in the
  mild-payload regime (long-vs-short p=0.87) - the harm needs the extreme-payload x
  tight-budget corner, which manipulated mild demos do not reach. Small consistent
  reasoning-demos depression (p=0.063, 4/4 models). Native-long demo pools nearly
  EMPTY (0-1 in 2/4 models): models rarely write long native-language traces; the
  verbose register is imported from reasoning subjects. Original subject-null
  replicates with paired tests.
- E4 powered within-subject residual (PRIMARY ENDPOINT; n=3,822 pairs, 6 models,
  benchmark + frozen 393-item replication set): NO knowledge-side erosion.
  Benchmark half: small nominally significant improvement (p=0.036 uncorrected)
  (knowledge-scored b=129 c=166, p=0.036; 6/6 models positive). Replication: exact
  null (p=1.0). OVERTURNS the paper-era vLLM/bf16 -7.2 (p=.045, n=250, one model)
  as small-sample fluctuation.
- E5 fixes shoot-out: compliant+rescue recipe PASSES the pre-registered bar
  (recovery 85.6%, math retention 108.3%); compliant alone 75.6/108.3; 4x budget
  77.2/180.6 (costliest per point). Recipe overshoots cot on 9b (37.0 vs 36.0).
- E6 bf16 coverage (3 large models, one-per-boot due to GB10 driver leak):
  erosion appears at bf16 in 2/3 models (NOT a Q4 artifact); all three models'
  deltas flip sign vs Q4 -> precision extends the instability finding.
  FERTILITY CORRECTION: math outputs cost MORE tokens than Uzbek prose (~0.43 vs
  ~0.38 tok/char; ~130-160 vs ~93-124 tok/item); ona-side truncation pressure comes
  from longest INPUTS (337 vs 156 chars mean) + demo-induced response inflation,
  not tokenizer fertility.

## Resolved claim list for the paper rewrite (spec section 2 disposition)

1. Redistribution framing: KEEP, sharpened - the full prompting stack lands below
   direct answering on ona_tili in 5/6 models (paper-era data) and vanilla bootstrap
   trails the compliant fix on BOTH axes in the replication (E1).
2. Two-part decomposition: REVISED - format/truncation tax (causal, budget- and
   payload-dependent, stack-varying) + small content-side depression from
   reasoning-subject demos (E3 p=0.063, n.s. after correction; honest as suggestive).
   No powered knowledge-side erosion within-subject (E4).
3. Causal completion: budget lever CONFIRMED (E2); demo-length lever null in the
   mild regime (E3) - report both, attribute harm to the extreme corner.
4. Repair: CONFIRMED (E5 recipe passes bar; adopt as recommended deployment default).
5. Practice: per-subject auditing ON THE DEPLOYMENT STACK (stack/precision
   instability makes borrowed audits invalid); dual scoring (deployment vs
   knowledge); parse-failure-aware evaluation.
Honest corrections to publish: the original p=0.011 headline is partly format
failures and does not replicate across stacks; the original -7.2 robustness claim
is overturned by the powered test.

## Assets

- Tables/numbers: results/paper_numbers.json (machine sections e1-e5, incl. the E4
  primary endpoint) + results/e*/DECISION.md (all verdicts) + analysis/ scripts
  (regenerate everything).
- Figures: paper/figures/fig_decomposition.{pdf,png} (E1); fig_subject_delta
  (paper-era, unchanged).
- Benchmark hygiene: loader repairs qids 299/314 from traces, excludes 6
  unrecoverable train/dev rows; split reproduces shipped traces exactly
  (analysis/verify_split.py). Dataport upstream copy still needs the 8 rows fixed.

## Suggested paper title direction

"Anatomy and Repair of a Native-Language 'Erosion': How Prompt Optimization,
Token Budgets, and Serving Stacks Interact in Low-Resource Deployment"
