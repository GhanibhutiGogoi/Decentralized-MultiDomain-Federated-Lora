# Quantity-skew decentralized LoRA results

Scores use the official labeled GLUE validation split. SST-2 best validation accuracy is primary; final accuracy is secondary. These are exploratory comparisons of our independent implementations, not hidden-test results or wins over printed paper numbers.

Smoke runs are execution checks only. Their scores and paired arithmetic do not establish efficacy or failure of the proposed method.

Runs: 43 complete, 0 incomplete, 0 invalid. In-progress runs are inventoried only and excluded from every aggregate, paired effect, and figure; any partial round rows retain incomplete status in per_round.csv. Complete runs enter only source/model/data/budget-compatible groups. Smoke and full configured budgets, and equal-size anchors and quantity-skew extensions, remain separate.

Metrics were independently recomputed from saved predictions and validation labels. The standalone checkpoint audit separately verifies best/final inference, source/data integrity, and exact raw prediction matches; its documented model/install-helper dependency is shared. Archived checkpoint binaries may be absent when their exact hashes are bound to that audit. All intervals below are unadjusted Student-t intervals of seed-paired differences; small samples are exploratory, and nonsignificance does not establish parity.

## Group 2ec9b556fbc202fe

SST2 · quantity · full_configured_budget; 20 rounds, batch 32, local steps 211, local epochs 1. Total local updates 42,200; training-example exposures 1,350,400. Source `daced6fc0052f4aab6069aea78ce1468bd5c039b76dedb71080155b0276b8d61`.

A full configured budget means all training data are available and at least 20 rounds run; it does not imply an exact reproduction of the publication's budget. Rank-16 reference arms exceed weaker clients' rank caps. The assembly root stores routed adapter records; neither routing nor raw-data locality establishes privacy.

Primary metric: best validation accuracy (%). Mean ± sample SD across seeds; n=1 has no estimated between-seed uncertainty.

| Method | Seeds | Best primary | Final secondary | Training GB | Best + final checkpoint audits |
|---|---:|---:|---:|---:|---:|
| Adaptive ranks, sample weights | 5 | 94.243 ± 0.249 | 93.899 ± 0.188 | 1.291 | 5/5 |
| Adaptive ranks, uniform weights | 5 | 94.174 ± 0.221 | 93.807 ± 0.513 | 1.296 | 5/5 |
| Dec-LoRA reimplementation, rank 16 | 5 | 94.541 ± 0.238 | 94.450 ± 0.320 | 1.893 | 5/5 |
| Dec-LoRA reimplementation, rank 4 | 5 | 94.495 ± 0.334 | 94.266 ± 0.372 | 1.185 | 5/5 |
| Fixed heterogeneous ranks, sample weights | 5 | 94.404 ± 0.126 | 94.220 ± 0.238 | 1.468 | 5/5 |
| Fixed heterogeneous ranks, uniform weights | 5 | 94.518 ± 0.262 | 94.197 ± 0.320 | 1.468 | 5/5 |
| Effective products, rank 16, sample weights | 5 | 94.725 ± 0.181 | 94.427 ± 0.224 | 1.893 | 5/5 |

Paired effects use each seed's own best endpoint, even when best rounds differ. Final-round effects are also recorded in JSON. Main effects average the two within-factor contrasts; the interaction is the difference of those differences.

| Paired effect | Matched seeds | Best endpoint difference (pp) | Exploratory 95% t interval |
|---|---:|---:|---:|
| primary_adaptive_sample_minus_declora16 | 5 | -0.298 ± 0.368 | [-0.755, 0.159] |
| adaptive_sample_minus_feasible_declora4 | 5 | -0.252 ± 0.348 | [-0.684, 0.180] |
| adaptive_sample_minus_product16_sample | 5 | -0.482 ± 0.318 | [-0.877, -0.087] |
| adaptation_at_uniform_weights | 5 | -0.344 ± 0.181 | [-0.569, -0.119] |
| adaptation_at_sample_weights | 5 | -0.161 ± 0.192 | [-0.399, 0.078] |
| sample_weighting_at_fixed_ranks | 5 | -0.115 ± 0.181 | [-0.340, 0.110] |
| sample_weighting_at_adaptive_ranks | 5 | 0.069 ± 0.192 | [-0.169, 0.307] |
| adaptation_main_effect | 5 | -0.252 ± 0.104 | [-0.382, -0.123] |
| sample_weighting_main_effect | 5 | -0.023 ± 0.104 | [-0.152, 0.106] |
| adaptation_by_weighting_interaction | 5 | 0.183 ± 0.310 | [-0.201, 0.568] |

![Validation accuracy versus rounds and training traffic](figures/2ec9b556fbc202fe_accuracy.png)

## Group 6923a7f12dc89d05

SST2 · quantity · smoke; 4 rounds, batch 32, local steps 2, local epochs 1. Total local updates 80; training-example exposures 2,560. Source `db671264d9ff4f23f20cbacd08697632d2b68094043a4bbfa3ae11d51c654332`.

A full configured budget means all training data are available and at least 20 rounds run; it does not imply an exact reproduction of the publication's budget. Rank-16 reference arms exceed weaker clients' rank caps. The assembly root stores routed adapter records; neither routing nor raw-data locality establishes privacy.

Primary metric: best validation accuracy (%). Mean ± sample SD across seeds; n=1 has no estimated between-seed uncertainty.

| Method | Seeds | Best primary | Final secondary | Training GB | Best + final checkpoint audits |
|---|---:|---:|---:|---:|---:|
| Adaptive ranks, sample weights | 1 | 52.523 (one seed) | 52.523 (one seed) | 0.286 | 1/1 |
| Adaptive ranks, uniform weights | 1 | 68.807 (one seed) | 68.807 (one seed) | 0.286 | 1/1 |
| Dec-LoRA reimplementation, rank 16 | 1 | 68.005 (one seed) | 68.005 (one seed) | 0.379 | 1/1 |
| Dec-LoRA reimplementation, rank 4 | 1 | 69.037 (one seed) | 69.037 (one seed) | 0.237 | 1/1 |
| Fixed heterogeneous ranks, sample weights | 1 | 53.096 (one seed) | 53.096 (one seed) | 0.294 | 1/1 |
| Fixed heterogeneous ranks, uniform weights | 1 | 70.298 (one seed) | 70.298 (one seed) | 0.294 | 1/1 |
| Effective products, rank 16, sample weights | 1 | 52.179 (one seed) | 52.179 (one seed) | 0.379 | 1/1 |

Paired effects use each seed's own best endpoint, even when best rounds differ. Final-round effects are also recorded in JSON. Main effects average the two within-factor contrasts; the interaction is the difference of those differences.

| Paired effect | Matched seeds | Best endpoint difference (pp) | Exploratory 95% t interval |
|---|---:|---:|---:|
| primary_adaptive_sample_minus_declora16 | 1 | -15.482 (one seed) | unavailable (one seed) |
| adaptive_sample_minus_feasible_declora4 | 1 | -16.514 (one seed) | unavailable (one seed) |
| adaptive_sample_minus_product16_sample | 1 | 0.344 (one seed) | unavailable (one seed) |
| adaptation_at_uniform_weights | 1 | -1.491 (one seed) | unavailable (one seed) |
| adaptation_at_sample_weights | 1 | -0.573 (one seed) | unavailable (one seed) |
| sample_weighting_at_fixed_ranks | 1 | -17.202 (one seed) | unavailable (one seed) |
| sample_weighting_at_adaptive_ranks | 1 | -16.284 (one seed) | unavailable (one seed) |
| adaptation_main_effect | 1 | -1.032 (one seed) | unavailable (one seed) |
| sample_weighting_main_effect | 1 | -16.743 (one seed) | unavailable (one seed) |
| adaptation_by_weighting_interaction | 1 | 0.917 (one seed) | unavailable (one seed) |

![Validation accuracy versus rounds and training traffic](figures/6923a7f12dc89d05_accuracy.png)

## Group 99281505a3d75619

SST2 · equal · full_configured_budget; 20 rounds, batch 32, local steps 0, local epochs 1. Total local updates 42,200; training-example exposures 1,346,980. Source `daced6fc0052f4aab6069aea78ce1468bd5c039b76dedb71080155b0276b8d61`.

A full configured budget means all training data are available and at least 20 rounds run; it does not imply an exact reproduction of the publication's budget. Rank-16 reference arms exceed weaker clients' rank caps. The assembly root stores routed adapter records; neither routing nor raw-data locality establishes privacy.

Primary metric: best validation accuracy (%). Mean ± sample SD across seeds; n=1 has no estimated between-seed uncertainty.

| Method | Seeds | Best primary | Final secondary | Training GB | Best + final checkpoint audits |
|---|---:|---:|---:|---:|---:|
| Dec-LoRA reimplementation, rank 16 | 1 | 94.151 (one seed) | 94.151 (one seed) | 1.893 | 1/1 |

Paired effects use each seed's own best endpoint, even when best rounds differ. Final-round effects are also recorded in JSON. Main effects average the two within-factor contrasts; the interaction is the difference of those differences.

| Paired effect | Matched seeds | Best endpoint difference (pp) | Exploratory 95% t interval |
|---|---:|---:|---:|
| primary_adaptive_sample_minus_declora16 | 0 | unavailable | unavailable |
| adaptive_sample_minus_feasible_declora4 | 0 | unavailable | unavailable |
| adaptive_sample_minus_product16_sample | 0 | unavailable | unavailable |
| adaptation_at_uniform_weights | 0 | unavailable | unavailable |
| adaptation_at_sample_weights | 0 | unavailable | unavailable |
| sample_weighting_at_fixed_ranks | 0 | unavailable | unavailable |
| sample_weighting_at_adaptive_ranks | 0 | unavailable | unavailable |
| adaptation_main_effect | 0 | unavailable | unavailable |
| sample_weighting_main_effect | 0 | unavailable | unavailable |
| adaptation_by_weighting_interaction | 0 | unavailable | unavailable |

![Validation accuracy versus rounds and training traffic](figures/99281505a3d75619_accuracy.png)

Editorial note: the frozen campaign script mislabeled the final-score table header as primary. This copy corrects that header only. SUMMARY.raw.md preserves the exact remote report; all numerical data are unchanged.
