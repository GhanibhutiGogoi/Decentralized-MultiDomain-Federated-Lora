# Independent final quantity-study audit

All 36 full runs pass the artifact audit: 35 matched quantity runs and one separate equal-size anchor. All 72 recorded best/final inference audits are bound to current checkpoint bytes; 36 peer-state archives are also hashed. All 720 round metrics are recomputed from saved predictions and canonical validation labels.

This script independently checks artifact consistency and calculations. It verifies the recorded inference-audit evidence against the saved models and predictions; it does not rerun model inference or import the experiment/summary implementations.

## Completed five-seed results

Accuracy is percent, mean ± sample SD across seeds 42–46. Best official SST-2 validation accuracy is primary; final accuracy is secondary.

| Arm | Best validation | Final validation | Training GB | Production GB | Mean persistent rank |
|---|---:|---:|---:|---:|---:|
| declora16 | 94.541 ± 0.238 | 94.450 ± 0.320 | 1.893 | 2.011 | 16.000 |
| declora4 | 94.495 ± 0.334 | 94.266 ± 0.372 | 1.185 | 1.259 | 4.000 |
| product16_sample | 94.725 ± 0.181 | 94.427 ± 0.224 | 1.893 | 2.011 | 16.000 |
| fixed_uniform | 94.518 ± 0.262 | 94.197 ± 0.320 | 1.468 | 1.556 | 8.800 |
| fixed_sample | 94.404 ± 0.126 | 94.220 ± 0.238 | 1.468 | 1.556 | 8.800 |
| adaptive_uniform | 94.174 ± 0.221 | 93.807 ± 0.513 | 1.296 | 1.373 | 5.872 |
| adaptive_sample | 94.243 ± 0.249 | 93.899 ± 0.188 | 1.291 | 1.368 | 5.792 |

## Paired comparisons and factorial effects

Differences are percentage points. Intervals are exploratory, unadjusted 95% Student-t intervals across the five paired seeds.

| Contrast | Best difference ± SD | Best 95% interval | Final difference ± SD | Final 95% interval |
|---|---:|---:|---:|---:|
| adaptive_sample_minus_declora16 | -0.298 ± 0.368 | [-0.755, +0.159] | -0.550 ± 0.297 | [-0.919, -0.182] |
| adaptive_sample_minus_declora4 | -0.252 ± 0.348 | [-0.684, +0.180] | -0.367 ± 0.469 | [-0.949, +0.215] |
| adaptive_sample_minus_product16_sample | -0.482 ± 0.318 | [-0.877, -0.087] | -0.528 ± 0.385 | [-1.006, -0.049] |
| adaptation_at_uniform_weights | -0.344 ± 0.181 | [-0.569, -0.119] | -0.390 ± 0.755 | [-1.327, +0.547] |
| adaptation_at_sample_weights | -0.161 ± 0.192 | [-0.399, +0.078] | -0.321 ± 0.221 | [-0.595, -0.047] |
| sample_weighting_at_fixed_ranks | -0.115 ± 0.181 | [-0.340, +0.110] | +0.023 ± 0.221 | [-0.251, +0.297] |
| sample_weighting_at_adaptive_ranks | +0.069 ± 0.192 | [-0.169, +0.307] | +0.092 ± 0.651 | [-0.716, +0.900] |
| adaptation_main_effect | -0.252 ± 0.104 | [-0.382, -0.123] | -0.356 ± 0.392 | [-0.842, +0.131] |
| sample_weighting_main_effect | -0.023 ± 0.104 | [-0.152, +0.106] | +0.057 ± 0.284 | [-0.295, +0.410] |
| adaptation_by_weighting_interaction | +0.183 ± 0.310 | [-0.201, +0.568] | +0.069 ± 0.789 | [-0.910, +1.048] |

## Interpretation

The planned comparison is complete. The proposed adaptive/sample method has lower mean best and final validation accuracy than rank 16, feasible rank 4, and fixed/sample controls. It demonstrates reduced training traffic against rank 16 and fixed/sample, while using more traffic than rank 4. These findings establish a measured accuracy/traffic tradeoff; they do not demonstrate superiority or accuracy parity. A best-endpoint interval crossing zero does not establish equivalence. Neither adaptive ranks nor sample weighting shows an accuracy advantage in the factorial average. The results are scoped to this reconstruction, task, quantities, capacity schedule and budget.

Adaptive training-traffic reductions relative to each comparator:

- declora16: 31.808% (negative means more adaptive traffic).
- declora4: -8.918% (negative means more adaptive traffic).
- fixed_sample: 12.084% (negative means more adaptive traffic).

## Verification limits

- Saved message SHA256 strings and ledger arithmetic checked; raw wire buffers are not retained, so their payloads are not reconstructed.
- Initial model state hashes were not recorded. Common initialization is supported by pinned same-seed code and model unit tests, not a per-run initial-state artifact comparison.
- Confidence intervals are unadjusted exploratory Student-t intervals across five seeds, not across tasks or fresh evaluation samples; no predeclared noninferiority margin exists.
- Best labeled-validation accuracy is primary; final-round accuracy is secondary; hidden test was not used.
- Modeled count setup is 320 bytes. Production traffic excludes base/initial/topology provisioning and network framing. Whole-process CUDA peak is not independent-client memory.
- Single-process peer simulation with visible updates and a gathering root; no enforced privacy guarantee or real multi-machine deployment.

The original disjoint-domain negative study is preserved separately. Final paper integration and PDF validation are delivery tasks beyond this numerical audit.
