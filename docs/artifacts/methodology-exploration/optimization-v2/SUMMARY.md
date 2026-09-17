# Exploratory optimization controls

Post-hoc mechanism controls on the same official CIFAR-100 full test set and frozen ResNet-18 features; no fresh held-out confirmation.

All 21 saved checkpoints were independently reevaluated on gpu003 with a direct CPU fp32 forward pass over all 10,000 test examples; every displayed final correct count matched. The unmodified pooled-reset and FedAvg controls exactly reproduced all 30 original scientific round records for all three seeds, excluding timings.

| Arm | Final full-test accuracy (%) | Paired gap to pooled-reset (pp) |
|---|---:|---:|
| Pooled; Adam reset | 57.253 ± 0.655 | 0.000 ± 0.000 |
| Pooled + epoch SVD | 57.170 ± 0.563 | -0.083 ± 0.136 |
| Pooled + centered SVD | 57.170 ± 0.488 | -0.083 ± 0.211 |
| FedAvg; non-IID | 20.057 ± 0.444 | -37.197 ± 0.431 |
| FedAvg; non-IID + centering | 18.513 ± 1.239 | -38.740 ± 1.890 |
| FedAvg; IID | 53.253 ± 0.442 | -4.000 ± 1.096 |
| FedAvg; IID + centering | 53.297 ± 0.270 | -3.957 ± 0.921 |

| Paired intervention effect | Accuracy difference (pp) |
|---|---:|
| pooled epoch svd effect | -0.083 ± 0.136 |
| pooled centering after svd effect | 0.000 ± 0.082 |
| noniid fedavg centering effect | -1.543 ± 1.632 |
| iid allocation effect | 33.197 ± 0.843 |
| iid centering effect | 0.043 ± 0.197 |
| iid fedavg gap to pooled | -4.000 ± 1.096 |
| iid centered fedavg gap to pooled | -3.957 ± 0.921 |

Mean and sample SD across three seeds; differences calculated within matching seeds; no formal equivalence claim.

## What these controls show

- Roundwise exact reproduction of the unmodified pooled and FedAvg controls ties these interventions to the earlier corrected experiment.
- Pooled SVD refactorization barely changes immediate centered training logits and final accuracy; it does not reproduce the large FedAvg failure by itself.
- IID allocation recovers much of the non-IID FedAvg deficit while preserving sample counts and the data union, implicating statistical heterogeneity/local-update behavior.
- Non-IID FedAvg has large common-logit energy, but removing that mode does not recover its accuracy in these controls; this statistic alone is not a demonstrated remedy.
- The IID FedAvg arms still fall below pooled accuracy; these controls are diagnostic, not a successful adaptive decentralized end-to-end result.

## Immediate SVD and common-mode diagnostics

SVD reconstruction changes the factor representation. Its immediate functional discrepancy here is measured using centered logits on a fixed training probe; the following small values are not a whole-test-set logit bound.

| Pooled intervention | Maximum absolute centered training-logit change | Maximum relative reconstruction energy |
|---|---:|---:|
| pooled_reset_svd | 4.0054321e-05 | 1.177516e-12 |
| pooled_reset_svd_center | 3.8146973e-05 | 1.2982711e-12 |

For common-logit energy, each seed is first averaged over its 30 rounds; uncertainty is then computed across the three seed averages. Measurements are taken before that round's intervention. Centered-tail energy excludes the class-common mode.

| Arm | Mean common-logit energy (%) | Mean centered tail beyond rank16 (%) |
|---|---:|---:|
| pooled_reset_svd | 2.258 ± 0.156 | 0.000 ± 0.000 |
| pooled_reset_svd_center | 0.060 ± 0.002 | 0.000 ± 0.000 |
| fedavg16 | 96.447 ± 0.386 | 0.013 ± 0.001 |
| fedavg16_center | 23.829 ± 0.740 | 0.012 ± 0.001 |
| fedavg16_iid | 2.073 ± 0.065 | 0.003 ± 0.001 |
| fedavg16_iid_center | 0.104 ± 0.012 | 0.003 ± 0.001 |

## Interpretation limits

Same initialization, alpha, total training examples, per-client sample counts and optimizer hyperparameters; pooled and FedAvg optimizer step counts differ.
IID reallocation preserves each client's training sample count, the disjoint pooled training union and every test shard. Its changed split is intentional. These arms use centralized FedAvg at uniform rank16; they do not demonstrate adaptive-rank decentralized accuracy preservation. The interventions were selected after observing the original failure and reuse the official test set, so they are exploratory mechanism evidence. No privacy guarantee or new held-out confirmation is claimed.

The current script hash, raw-input hashes, original-control matching checks, independent checkpoint/prediction hashes, per-seed final metrics, paired effects and diagnostic summaries are in `summary.json`. Publication figures are generated from the same records. The separate first failed launch is preserved by the parent experiment record; this completed directory does not overwrite it.
