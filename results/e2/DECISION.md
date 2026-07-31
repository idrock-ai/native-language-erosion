# E2 decision memo (max_tokens dose-response)

Prediction tested: if the format tax is truncation-driven, raising the decoding budget
should monotonically remove truncations; any erosion remaining at 2048 is the
budget-insensitive content residual.

Result: CONFIRMED with a two-part decomposition.
- gemma4:e4b: truncations 28 -> 16 -> 3 -> 0 across 256/512/1024/2048
  (Cochran-Armitage trend p < 1e-4); ona delta -10 -> -3.
- qwen3.5:9b: truncations 13 -> 6 -> 3 -> 5 (trend p = 0.016); ona delta -9 -> -5.
- gemma4:31b control: zero truncations at every budget, no erosion (+0/+1/+0/-2).

Interpretation: the budget lever eliminates the truncation component causally, but a
budget-insensitive residual remains in the erosion-prone models (-3 to -5 at 2048 with
~zero truncations). This isolates the content/style residual that E3 manipulates
directly and E4 powers. Also note the 256-budget cells show the deployment-worst case:
tight budgets nearly double the erosion (e4b -10, 9b -9).
