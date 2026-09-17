# Quantity-skew decentralized LoRA results

Scores use the official labeled GLUE validation split. SST-2 best validation accuracy is primary; final accuracy is secondary. These are exploratory comparisons of our independent implementations, not hidden-test results or wins over printed paper numbers.

Smoke runs are execution checks only. Their scores and paired arithmetic do not establish efficacy or failure of the proposed method.

Runs: 7 complete, 4 incomplete, 0 invalid. In-progress runs are inventoried only and excluded from every aggregate, paired effect, and figure; any partial round rows retain incomplete status in per_round.csv. Complete runs enter only source/model/data/budget-compatible groups. Smoke and full configured budgets, and equal-size anchors and quantity-skew extensions, remain separate.

Metrics were independently recomputed from saved predictions and validation labels. The standalone checkpoint audit separately verifies best/final inference, source/data integrity, and exact raw prediction matches; its documented model/install-helper dependency is shared. Archived checkpoint binaries may be absent when their exact hashes are bound to that audit. All intervals below are unadjusted Student-t intervals of seed-paired differences; small samples are exploratory, and nonsignificance does not establish parity.

## Group 6923a7f12dc89d05

SST2 · quantity · smoke; 4 rounds, batch 32, local steps 2, local epochs 1. Total local updates 80; training-example exposures 2,560. Source `db671264d9ff4f23f20cbacd08697632d2b68094043a4bbfa3ae11d51c654332`.

A full configured budget means all training data are available and at least 20 rounds run; it does not imply an exact reproduction of the publication's budget. Rank-16 reference arms exceed weaker clients' rank caps. The assembly root stores routed adapter records; neither routing nor raw-data locality establishes privacy.

Primary metric: best validation accuracy (%). Mean ± sample SD across seeds; n=1 has no estimated between-seed uncertainty.

| Method | Seeds | Best primary | Final primary | Training GB | Best + final checkpoint audits |
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

## Run status and verification gaps

- `/home/gogoi/ahlora-quantity-20260917/runs/full-v1-equal-declora16-seed42` — incomplete: Training is incomplete; excluded from aggregate estimates and paired effects.
- `/home/gogoi/ahlora-quantity-20260917/runs/full-v1-quantity-adaptive_sample-seed42` — incomplete: Training is incomplete; excluded from aggregate estimates and paired effects.
- `/home/gogoi/ahlora-quantity-20260917/runs/full-v1-quantity-declora16-seed42` — incomplete: Training is incomplete; excluded from aggregate estimates and paired effects.
- `/home/gogoi/ahlora-quantity-20260917/runs/full-v1-quantity-fixed_sample-seed42` — incomplete: Training is incomplete; excluded from aggregate estimates and paired effects.
