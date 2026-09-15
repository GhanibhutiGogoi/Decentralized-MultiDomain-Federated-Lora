# Corrected integrated experiment: measured results

CIFAR-100; frozen ResNet-18 features and a LoRA classification head; single-process peer-flow simulation.

sample standard deviation across paired seeds (ddof=1), not a confidence interval. descriptive paired comparison; no prespecified noninferiority/equivalence margin or formal equivalence claim.

| Arm | Full-test accuracy (%) | Paired difference vs pooled (pp) | Seeds |
|---|---:|---:|---:|
| Pooled rank 16 | 57.23 ± 0.59 | 0.00 ± 0.00 | 3 |
| Pooled rank 16 Adam reset | 57.25 ± 0.66 | 0.03 ± 0.07 | 3 |
| FedAvg rank 16 | 20.06 ± 0.44 | -37.17 ± 0.40 | 3 |
| Peer rank 16 | 12.69 ± 3.55 | -44.54 ± 4.12 | 3 |
| Peer ranks 4/8/16 sample weights | 7.71 ± 0.81 | -49.51 ± 1.38 | 3 |
| Fixed ranks quality weights | 10.69 ± 0.73 | -46.54 ± 0.47 | 3 |
| Fixed ranks domain weights | 10.78 ± 2.01 | -46.44 ± 2.08 | 3 |
| Adaptive ranks quality weights | 6.48 ± 1.67 | -50.75 ± 2.08 | 3 |
| Adaptive ranks domain weights | 6.90 ± 2.03 | -50.33 ± 2.46 | 3 |

The differences use matching seeds before aggregation. Positive values favor the peer arm; negative values favor pooled training. A sample SD is not an equivalence confidence interval.

| Arm | Training factors (MiB) | Control (MiB) | Final assembly (MiB) | Total deployment (MiB) | Evaluation-only assembly (MiB) |
|---|---:|---:|---:|---:|---:|
| Pooled rank 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Pooled rank 16 Adam reset | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| FedAvg rank 16 | 33.618 | 0.000 | 0.000 | 33.618 | 0.000 |
| Peer rank 16 | 33.618 | 0.000 | 2.734 | 36.353 | 79.300 |
| Peer ranks 4/8/16 sample weights | 19.611 | 0.000 | 2.734 | 22.345 | 79.300 |
| Fixed ranks quality weights | 19.611 | 0.183 | 2.734 | 22.528 | 79.300 |
| Fixed ranks domain weights | 19.611 | 3123.458 | 2.734 | 3145.803 | 79.300 |
| Adaptive ranks quality weights | 11.402 | 0.183 | 2.734 | 14.319 | 79.300 |
| Adaptive ranks domain weights | 11.396 | 3123.458 | 2.734 | 3137.589 | 79.300 |

float32 training factor payloads + float64 modeled domain-control payloads + one final dense assembly; setup/framing/headers excluded; earlier evaluation assemblies separate.

same training sample exposures, alpha, initialization, split and Adam hyperparameters; optimizer step counts and moment persistence differ.

factor bytes and rank-sample products are proxies; probe passes and measured timings are separate; no total-FLOP saving claim.

no privacy guarantee; domain control exchange reveals training histograms and local dense adapter changes to all peers.
