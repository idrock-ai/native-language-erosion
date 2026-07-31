# E5 decision memo (fixes shoot-out, 6 models)

Candidates vs vanilla dspy_bootstrap (sources: results/e1 @512 for compliant/rescue;
results/e2/mt2048 + results/e5/mt2048 for the budget fix).

MEANS (success bar: recovery >= 80% of the cot->vanilla ona erosion AND >= 90% of the
math gain retained):
- compliant metric alone: recovery 75.6, retention 108.3 - just under the bar
  (dragged by qwen3.6:27b, whose -4 is content flips that format fixes cannot touch)
- rescue parsing alone: 57.8 / 116.7
- budget 2048 (4x decode cost): 77.2 / 180.6
- compliant + rescue (the recipe): 85.6 / 108.3 - PASS

Notables: the recipe overshoots cot on qwen3.5:9b (37.0 vs 36.0); every fix retains or
improves math because vanilla's format failures hit math too. Footnotes for the paper:
(i) recovery is defined as 100% when a model shows no vanilla erosion, regardless of
fix quality; (ii) gemma4:31b's -100% retention is a small-denominator artifact (its
vanilla math gain over cot was only 2.6 points); (iii) budget-fix numbers ride on
n=100 ona cells and are the costliest option per point recovered.

Recommendation for the paper: ship the recipe (one-line compliant metric + rescue
parsing) as the default mitigation; budget relief is a complement where latency/cost
permit; none of the fixes address the small content-side depression from
reasoning-subject demos (E3, p=0.063), which within-subject bootstrapping avoids
entirely (E4).
