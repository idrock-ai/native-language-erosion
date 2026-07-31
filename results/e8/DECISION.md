# E8 decision memo (MIPROv2 generalization; spark-3/GB10, Ollama Q4_K_M, max_tokens=512)

Run: 6 models x {cot, dspy_mipro} x 251 DTM test items, fully instrumented. Protocol
identical to E1 -- same split, seed, stack, budget, correctness metric, same items -- so
the optimizer is the only thing that changes. CoT was re-run in-process rather than
reused from E1, which doubles as a determinism check (see S4, which turned out to matter).

Search budget set explicitly: num_candidates=5, num_trials=10, minibatch_size=25,
valset=100 items drawn from dev (which no other experiment touches). auto='light' was
avoided because it resolves trial counts internally and would make the cost
irreproducible.

## Verdict in one line

**The native-subject differential is NOT a \textsc{BootstrapFewShot} quirk.** A
structurally different optimizer produces the same differential on the same stack.

## 1. The headline comparison

| optimizer / dataset | MH OR | 95% CI (item-cluster) | models agreeing |
|---|---|---|---|
| \textsc{BootstrapFewShot}, DTM original stack | 2.27 | [1.37, 3.91] | 5/6 |
| \textsc{BootstrapFewShot}, DTM replication stack | 1.63 | [0.97, 2.79] | 4/6 |
| **\textsc{MIPROv2}, DTM replication stack** | **1.75** | **[0.99, 3.14]** | **3/4 treated** |
| \textsc{BootstrapFewShot}, TurkishMMLU | 2.35 | [1.38, 4.17] | 5/6 |

The like-for-like comparison is the two replication-stack rows: **1.63 vs 1.75**, with
near-identical intervals whose lower bounds both sit just below 1 (0.973 and 0.989).
Two optimizers that share almost no machinery land in the same place.

This retires the paper's principal stated limitation ("one optimizer").

## 2. Why MIPROv2 is a real test, not a near-duplicate

\textsc{MIPROv2} searches *instructions* by Bayesian optimization against a validation
set. \textsc{BootstrapFewShot} only keeps training examples the model answers correctly
and never touches the instruction. Concretely, of the four treated models:

| model | demos | payload | instruction rewritten | OR |
|---|---|---|---|---|
| gemma4:e4b | 4 | 4,790 ch | yes | 1.38 |
| qwen3.5:4b | 1 | 700 ch | yes | 4.03 |
| qwen3.5:9b | **0** | 0 ch | yes | 0.69 |
| qwen3.6:27b | 1 | 444 ch | yes | 2.16 |

qwen3.5:9b was optimized with **zero demonstrations** -- instruction only -- and is the
one model whose OR falls below 1. The three models that carried demonstrations all sit
above 1. At n=4 this is a hint, not a finding, but it points the same way as the
mechanism: the harm needs demonstrations to be imported at all.

## 3. Two models were UNTREATED and are excluded

\textsc{MIPROv2} returned the unmodified baseline for **gemma4:31b** and **qwen3.5:27b**
-- original instruction, zero demonstrations. Under our light search budget it found
nothing better than the starting point. These models cannot speak to whether MIPROv2
harms the native subject, so they are excluded rather than scored as nulls. The honest
statement about them is *"our search was too weak to test this model"*, not *"MIPROv2
causes no harm here"*.

Detecting this by "zero discordant pairs" is **not sufficient**, and that near-miss is
the reason S4 exists: qwen3.5:27b produced byte-identical output on all 251 items, but
gemma4:31b produced 132/251 different completions and 14/251 different correctness
*running the identical program twice*. Scored naively, gemma4:31b would have contributed
14 flips of pure noise to the differential. Untreated models are therefore identified
from the saved program (`analysis/mipro_stats.py`), never from the outcome.

## 4. NEW FINDING: the gemma models are not run-to-run deterministic

Same stack, greedy decoding, temperature 0, identical program, two sessions:

| model | E1 vs E8 CoT agreement |
|---|---|
| qwen3.5:4b | 251/251 (100%) |
| qwen3.5:9b | 251/251 (100%) |
| qwen3.5:27b | 251/251 (100%) |
| qwen3.6:27b | 251/251 (100%) |
| gemma4:e4b | 226/251 (**90.0%**) |
| gemma4:31b | 241/251 (**96.0%**) |

Four of four qwen models reproduce exactly. Two of two gemma models do not. This is an
architecture-level split, not a one-model quirk.

Consequences:
- The paper cites gemma4:e4b's failure counts (62 truncated / 56 unparseable under
  vanilla; 17/13 under CoT) as if fixed. They carry run-to-run variance of a few items.
  Limitations must say so. The *claims* survive easily -- run-to-run variance is ~3
  items against gaps of 40+ -- but the precision implied by the numbers is overstated.
- It strengthens the instability thesis. The paper already argues the harm belongs to
  the (model x stack x selected-demos) triple. We can now add that for some models it is
  not stable across sessions on the *same* machine at temperature zero.

## 5. Mechanism: an independent confirmation from an unexpected direction

gemma4:e4b is the paper's worst format-tax case. Compare the two optimizers on it:

| | demo reasoning verbosity | truncated | unparseable | native delta |
|---|---|---|---|---|
| \textsc{BootstrapFewShot} (E1) | mean 631 ch | 62 | 56 | eroded |
| \textsc{MIPROv2} (E8) | mean **755** ch | **15** | **14** | **+3.0** |

MIPROv2's demonstrations are *more* verbose yet produce a quarter of the failures,
so payload size alone cannot explain the format tax. The reason is in the instruction
MIPROv2 wrote, which enforces the format the demos violate:

> "...while **strictly adhering to a structured output format**... providing a mandatory,
> **very brief justification**, and concluding with *only* the correct single-letter choice."

The paper's mechanism is that the compiled prompt **contradicts its own brevity
instruction**. E5's fix removes the contradiction from the *demonstration* side. MIPROv2
removes the same contradiction from the *instruction* side, incidentally, and gets the
same outcome. That is independent support for the mechanism from an optimizer that was
not designed to test it -- and it implies **a second repair the paper does not currently
offer**: strengthen the instruction rather than constrain the demonstrations. No metric
change required.

## 6. Status of these claims

- E8 is **not pre-registered**. It postdates the design spec, like E7.
- The differential contrast is post-hoc and reported as secondary throughout.
- Effective n is **4 treated models**, not 6. The CI includes 1 (barely). E8 shows the
  differential is *reproduced* by a second optimizer at the same magnitude, not that it
  is significant on its own at this n.
- A larger search budget would likely treat all six models. That is the obvious
  follow-up and the honest answer to "why only four".

## Reproduce

    python analysis/mipro_stats.py results/e8 --native ona_tili
    python analysis/interaction.py results/e8 --native ona_tili --cond-b dspy_mipro \
        --exclude gemma4:31b,qwen3.5:27b
