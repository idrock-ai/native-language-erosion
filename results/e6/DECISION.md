# E6 decision memo (bf16 large-model coverage + empirical fertility)

Protocol: vLLM 0.25.1 (native, ~/vllm-env) on spark-3 (GB10), --dtype bfloat16,
--gpu-memory-utilization 0.80, --enforce-eager, max_model_len 4096; conditions
cot + dspy_bootstrap at max_tokens 512 on the paper split (mixed-subject bootstrap,
identical to E1's protocol). One model per boot (see ops notes). All three large
models COMPLETED at true bf16.

## bf16 vs Q4 (ona_tili, n=100/model; Q4 from results/e1 same protocol/split)

| model            | bf16 cot->boot | Q4 cot->boot |
|------------------|----------------|--------------|
| Qwen3.5-27B      | 37.0 -> 32.0 (-5, trunc 1->6) | 31.0 -> 33.0 (+2) |
| Qwen3.6-27B      | 30.0 -> 37.0 (+7, math +15.8) | 34.0 -> 30.0 (-4) |
| gemma-4-31b-it   | 41.0 -> 36.0 (-5, trunc 1->0) | 38.0 -> 39.0 (+1) |

Verdict:
1. The erosion is NOT a Q4 quantization artifact - it appears at full precision in
   2/3 large models (and in the flagship with a truncation component).
2. Every model's effect CHANGES SIGN between precisions. Precision changes the
   model's correct-set on the train pool, hence which demos BootstrapFewShot
   self-selects, hence the observed effect - extending the run-instability finding
   (E1) to the precision axis. Observational n=100 cells swing +/-5-7; conclusions
   must rest on the causal (E2) and powered (E4) results, which they do.

## Empirical fertility (analysis/fertility.py over e1+e4+e6 logs)

Correction to the working hypothesis: Uzbek prose is NOT more token-expensive than
math output on these tokenizers - the opposite. Math reasoning costs MORE tokens
both per char (~0.43 vs ~0.38 tok/char) and per item (~130-160 vs ~93-124).
The ona_tili-side truncation risk under vanilla bootstrap instead comes from
(i) ona_tili having the LONGEST inputs (mean 337 chars question+options vs 156 math)
and (ii) demo-induced response inflation in the eroding models (E1: parsed-output p90
grew ~2x under long-demo bootstrap). The paper's mechanism section should attribute
the budget pressure to input length + induced verbosity, not tokenizer fertility.
Note: raw_text includes ~71 chars of constant adapter scaffolding per item; this
additive term does not change the subject ordering.

## Ops notes (GB10 platform, driver 595.71.05)

- Unloading any loaded large model orphans its device memory in the NVIDIA driver
  (unconditional: graceful SIGTERM with full-exit wait leaks identically to kill).
  Only a reboot reclaims it -> one-model-per-boot protocol (4 boots total).
- gpu-memory-utilization: 0.60 starves KV for 27B bf16; 0.80 works. enforce-eager
  avoids the fragile compile/graph-capture phase.
- ~/.cache/huggingface is root-owned (stale artifact); HF_HOME=~/hf_home used.
- Firmware upgrades are pending on the box (fwupdmgr) - advisable after the paper.
- A co-tenant GPU service on the shared box was stopped for these runs with its
  owner's consent, and restored afterwards.
