# Conservative domain-weighting iteration

This iteration evaluates a bounded domain-aware redistribution on the recorded
Project 2 Experiment 1 contribution rows. Base weights remain proportional to
`train_samples_seen * quality_score`; standardized domain signals are mapped to
a conservative factor with blend strength `0.10`, clipped to `[0.85, 1.15]`, and
renormalized under the base weights.

On the 75 real-data client-round observations at seed 42, the mean contribution
ranking metrics changed as follows:

| Metric | Quality only | Conservative domain | Change |
|---|---:|---:|---:|
| Spearman correlation | 0.152174 | 0.195652 | +0.043478 |
| Pairwise accuracy | 0.520000 | 0.546667 | +0.026667 |
| Weighted contribution | 0.575943 | 0.577445 | +0.001502 |

These are model-free ranking diagnostics against the measured leave-one-client-
out contribution target. They do not establish an end-to-end training-accuracy
gain. The implementation and reproducible benchmark are in
`framework/aggregation/domain_weighting.py` and
`experiment/experiment2/weight_benchmark.py`.
