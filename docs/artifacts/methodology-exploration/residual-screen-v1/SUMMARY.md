# Residual protocol screen: all candidates failed

5000 held-out ORIGINAL TRAINING examples; 45000 examples used for training. **These are validation results, not official-test results.**

All 11 saved models were independently reevaluated on gpu003 with a direct CPU fp32 forward pass; every correct count matched. Every arm completed 30 epochs with exactly 1,350,000 training sample exposures. Screening used seed42 only; no between-seed SD is available.

| Arm | Validation accuracy (%) | Final local training loss | Maximum local training loss | Protocol payload (MiB) |
|---|---:|---:|---:|---:|
| Pooled LoRA | 57.64 | 1.340 | 2.984 | 0.000 |
| Original fixed / domain | 4.52 | 1.317 | 3.520 | 3145.803 |
| Original adaptive / domain | 4.62 | 1.778 | 3.520 | 3137.259 |
| Residual uniform / sample ×1 | 1.68 | 3.802 | 3.802 | 164.066 |
| Residual fixed / sample ×1 | 1.42 | 3.804 | 3.804 | 164.066 |
| Residual adaptive / domain ×1 | 1.74 | 3.936 | 3.936 | 3287.524 |
| Residual adaptive / domain ×5 | 2.64 | 4.059 | 4.213 | 3287.524 |
| Residual adaptive / domain ×15 | 1.48 | 47.848 | 53.490 | 3287.524 |
| Residual adaptive / sample ×5 | 2.88 | 3.896 | 4.027 | 164.066 |
| Residual adaptive / quality ×5 | 2.46 | 4.027 | 4.166 | 164.248 |
| Residual fixed / domain ×5 | 2.60 | 3.943 | 4.130 | 3287.524 |

All eight retained-base residual recipes failed this screening comparison. The zero-residual preservation invariant does not guarantee that nonzero local learning updates will be useful.

The gain15 recipe remained numerically finite but was unstable: local training cross entropy increased from 3.520 to 47.848, with a maximum of 53.490. Independent final validation cross entropy was 41.463. This is severe finite loss growth, not a NaN crash or a proof of asymptotic divergence.

## Independent checks

The verifier opened only the original `train.pt`, checked its SHA-256, reconstructed the 45,000/5,000 holdout from `holdout.json`, verified exhaustive disjoint indices and 50 validation examples per class, and bound every arm to the same holdout/initialization/partition. The reused baseline `full_test_*` field names refer exclusively to this 5,000-example training holdout.

Every peer round was checked for normalized positive sample/quality/domain weights and trainable-rank ceilings. Adaptive floors and warmup were checked. Residual roots rotate 0–14 twice; all reduction, broadcast and control edges belong to the declared graph. Ledger message counts, per-edge byte sums, full protocol totals, reduction weights, and global projection residuals were verified. Every saved rank16 model and its logits remained finite, including gain15.

## Changed-method and resource scope

Multiple changes: retained dense frozen head, fresh local residual directions, rotating exact tree reduction/broadcast, class centering, rank16 global projection and declared gain. Not a one-variable repair or the original one-hop MH protocol.

Small ranks constrain trainable residuals only; every root performs rank100 reduction factorization and rank16 global projection, and all peers hold the frozen global head. Local function rank can exceed the residual rank.

Each residual round transports 2,867,312 bytes of dense residual reduction plus 2,867,200 bytes of frozen-head broadcast. Quality/domain control traffic is added separately. Domain allgather dominates the reported payload. Original-model-state controls count their training factors, control traffic, and one final assembly; evaluation-only intermediate assemblies remain outside deployment cost, as in the original driver. Packet framing, graph setup and runtime transport overhead remain excluded.

Each peer stores a 204,800-byte frozen head; folding the retained update into that existing weight avoids an extra persistent delta buffer, but aggregation and SVD workspace remain additional. A small trainable residual rank is not a guarantee that the full local function, frozen state, or root's aggregation fits the original rank-only capacity interpretation.

The screen does not establish that retained-base residual methods cannot work. It rejects these eight disclosed recipes at this budget and seed. All failures, the high-gain instability, raw per-round diagnostics and checkpoints remain preserved. No privacy guarantee; domain-control allgather exposes individual dense changes and metadata.
