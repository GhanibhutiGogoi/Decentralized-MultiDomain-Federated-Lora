# Composed end-to-end ablation

This is the first P3 run that composes dynamic rank resizing and bounded domain weighting inside the same decentralized ΔW protocol. It uses real cached CIFAR-100 features on gpu003, heterogeneous ranks `(4,12,32)`, 30 rounds, and seeds 42/43/44 with identical splits, initialization, topology, and optimizer settings.

`adaptive_rank` is a conservative loss-adaptive policy with two-round warm-up; it preserves the effective ΔW when resizing. `weighted` applies the bounded domain signal through a symmetric Sinkhorn reweighting of the MH matrix. `adaptive_weighted_rank` combines both.

Final personalized accuracy (mean ± sample SD):

| Arm | Accuracy |
|---|---:|
| Fixed MH | 24.16 ± 1.04% |
| Adaptive rank | 20.30 ± 2.51% |
| Domain weighted MH | 24.16 ± 1.04% |
| Adaptive rank + domain weighted | 20.30 ± 2.51% |

The tested rank policy reduced effective transmitted floats by 20.5%, but lost 3.85 percentage points of personalized accuracy. The bounded domain signal produced no measurable change in this run. This is a negative result for the tested composition, not a proof that all adaptive policies fail.

## Direct pooled-data conventional baseline

A separate rank-16 LoRA was trained directly on all 50,000 training examples for 30 full passes, with the same initialization, optimizer, batch size, and feature representation. Its full 10,000-image test accuracy was **57.25 ± 0.66%** across seeds 42–44. The decentralized heterogeneous runs scored only **5.11 ± 1.29%** consensus accuracy for fixed MH and **5.35 ± 0.20%** for adaptive rank. This is the direct evidence that the original “match conventional full-data LoRA” claim is not met under the tested heterogeneous decentralized protocol.
