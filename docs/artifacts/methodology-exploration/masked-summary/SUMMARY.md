# Partial-gradient exploration: completed validation-only screen and replication

5000 held-out ORIGINAL TRAINING examples; 45000 fit examples; official test not opened

All 15 saved adapters were independently evaluated on gpu003; every fp32 correct count matched the run record. The three replicated seeds are 42, 43 and 44. Seed42 conventional LoRA is reused from the residual screen. The same stratified training holdout, partition and initialization are used for each paired comparison.

| Replicated arm | Validation accuracy, mean ± sample SD (%) |
|---|---:|
| Conventional pooled LoRA | 57.18 ± 0.71 |
| Shared gradients: uniform rank16 | 57.18 ± 0.71 |
| Partial gradients: fixed / domain | 55.90 ± 0.59 |
| Partial gradients: adaptive / domain | 55.03 ± 1.01 |

| Paired effect | Mean ± sample SD (percentage points) |
|---|---:|
| uniform minus pooled | +0.00 ± 0.00 |
| fixed domain minus pooled | -1.28 ± 1.24 |
| adaptive domain minus pooled | -2.15 ± 1.69 |
| adaptive domain minus fixed domain | -0.87 ± 0.86 |

Uniform shared gradients and conventional pooled LoRA have identical final prediction vectors at all three seeds. Their per-epoch correct counts match in 89/90 evaluations; any intermediate differences are explicitly listed in `summary.json`. Endpoint equality does not imply bitwise equality of their optimization trajectories.

## All seed42 screening arms

| Arm | Validation accuracy (%) |
|---|---:|
| Shared gradients: uniform rank16 | 57.64 |
| Partial gradients: fixed / sample | 54.96 |
| Partial gradients: adaptive / sample | 54.68 |
| Partial gradients: adaptive / quality | 54.52 |
| Partial gradients: fixed / domain | 55.26 |
| Partial gradients: adaptive / domain | 54.80 |
| Conventional pooled LoRA | 57.64 |

## Interpretation and limits

Partial factor-gradient coordinates only. Every client requires rank16 factors, rank16 forward computation, full Adam state and padded full-rank gradients. One-process simulation with global minibatch scheduling and a rotating peer-tree collective; not the original model-state gossip protocol.

Sample SD across three paired seeds on one fixed validation holdout; no equivalence margin or formal noninferiority test. Additional screening arms have only seed42.

Gradient-coordinate products are a backward-computation proxy only, excluding common full-rank forward, optimizer work and probe work. Communication is modeled float payload including weight-control traffic, not measured network throughput.

No privacy guarantee. Dense gradient-derived domain signals, training histograms and quality metadata are exchanged.

The shared-gradient control reaches the same endpoint as ordinary pooled LoRA, while partial-gradient candidates remain below it. These are promising changed-method results, not a successful demonstration of the original rank-limited client-memory claim. The adaptive/domain candidate is weaker than its fixed/domain counterpart. All screening variants are retained; the replication plan was recorded before the seed42 screen finished. No official-test evaluation of these candidates was performed.

Transport validation checks each recorded weight-control collective's neighbor edges, root rotation, reduction/broadcast order, message totals and byte ledger. Gradient collective counts are checked against 352 steps × 28 messages per epoch; per-step gradient edge traces were not recorded, so their runtime paths cannot be reconstructed independently from the artifacts. Gradient-rank products, train exposures, ceiling-probe work, normalized sample/quality/domain multipliers, rank ceilings and controller observations are checked for every round.
