# Exact gradient synchronization control

This completed diagnostic tests whether separating data ownership necessarily
causes the pooled-versus-federated accuracy gap. It is **not** the heterogeneous
adaptive-rank solution. Both methods reached **57.23 ± 0.59%** full-test accuracy
across seeds 42/43/44: **57.80%, 56.63%, and 57.25%** respectively. Every one
of the 90 paired epoch evaluations had identical predictions; the final paired
accuracy difference was zero for each seed. Reported SD is sample SD across
the three seeds, not a population equivalence confidence interval.

The 192 float64 gradient checks passed. An independent forward evaluator
reproduced all six final saved-model accuracies and checked checkpoint/cache
hashes, tree edges, and communication totals. Each 30-epoch peer run required
328,440 modeled messages and **12,866,965,440 bytes** (12.87 decimal GB).

The control uses the same frozen CIFAR-100 features, original 15-client
non-IID ownership, initialization, rank 16, alpha 32, learning rate 0.001,
weight decay 0.0001, batch size 128, and persistent Adam as the conventional
pooled reference. Every peer has the same rank-16 factors. There is no
projection, SVD refactorization, domain reweighting, or adaptive rank change.

Each globally scheduled pooled minibatch is divided according to original
client ownership. A peer computes its mean gradient from its own examples.
A spanning tree of the original seeded ring reduces gradient numerators
and sample counts to peer 0, then broadcasts the mean gradient. All modeled
messages follow graph edges; payloads contain gradients and counts, without
raw examples or example indices.

For a minibatch $B_t$, let $B_{i,t}=B_t\cap D_i$ and $m_{i,t}=|B_{i,t}|$.
The reduction implements

$$
g_t=\sum_i\frac{m_{i,t}}{|B_t|}
\nabla_\theta\left[\frac{1}{m_{i,t}}
\sum_{z\in B_{i,t}}\ell(\theta_t;z)\right]
=\frac{1}{|B_t|}\sum_{z\in B_t}\nabla_\theta\ell(\theta_t;z).
$$

Empty contributions are zero. This is the pooled minibatch gradient at the
same factor coordinates. Identical initial factors and Adam state therefore
produce identical iterates in exact arithmetic. Float32 reduction order can
introduce numerical differences, which are measured rather than assumed away.

The simulator collapses identical peer parameter/optimizer replicas to one
object after explicitly simulating the reduction and broadcast. It does not
execute independent physical processes or sockets, demonstrate physical data
isolation, or measure network wall time. The logged runtime includes both
paired training paths, gradient oracles, and evaluations. A fixed tree root
and synchronization after every minibatch are intentional diagnostic choices.

For each seed, the first 64 minibatches also run an independent same-state
float64 gradient oracle with absolute tolerance `2e-11` and relative tolerance
`2e-10`. A separate saved-checkpoint evaluator reconstructs the forward pass
directly, checks full-test counts, verifies cache/checkpoint hashes and tree
edges, and recomputes communication totals. All executions occur on gpu003.

The communication charge includes both tree phases after every minibatch:
28 messages, each carrying 9,792 gradient elements and one int64 count.
Initial model dissemination, global scheduling metadata, and transport
framing are excluded. Every peer must accommodate rank 16. Consequently,
success here would establish a costly uniform-rank positive control, not
accuracy preservation under the original smallest-device rank ceilings.
It also provides no privacy guarantee against information revealed by gradients.

## Why local Adam is a separate issue

The follow-up `adam-order/` diagnostic uses 24 training minibatches (eight per
seed), resetting to the same initial factors and fresh Adam moments for each
batch. It loads no test data. Weighted local raw gradients equal the pooled
gradient to a maximum absolute error of `6.94e-17`. Nonetheless, averaging
the effective updates from independent local Adam steps produces a mean
cosine of **0.2416** against one global Adam step, and a mean relative update
difference of **1.1041**. Mean same-batch cross-entropy falls by **0.01534**
after global Adam and **0.001436** after averaging the local Adam updates.

Adam is nonlinear in its input gradient, so averaging its outputs generally
differs from applying it to an averaged gradient. The positive control above
changes both the synchronization interval and this optimizer aggregation
order. These fresh-initialization one-step probes demonstrate an actual
mechanism in the benchmark; they do not apportion the full training-run gap
or establish a converged optimizer comparison.

Files:

- `driver_snapshot.py`: exact full-run source; the canonical source is
  `project-3-hierarchical-gossip/experiments/diagnose_gradient_equivalence.py`.
- `launch.sh`: exact remote full-run launch settings.
- `smoke/`: completed one-epoch seed-42 attempt and independent verification;
  its source snapshot predates an explicit broadcast-count bookkeeping addition.
- `full/`: per-seed round records, float64 gradient checks, manifests, and saved
  paired models, followed by independent saved-model verification. Its
  `dependency_source_snapshot.tar.gz` preserves 30 source files; the manifest
  explicitly records that dependency capture occurred after launch from the
  untouched isolated snapshot.
- `verify_saved.py`: independent checkpoint/communication verifier.
- `summarize.py`: creates `per_seed.csv`, `curves.csv`, `summary.json`, and
  `gradient_equivalence.pdf` / `.png` from completed recorded runs.
- `adam-order/results.json`: all training-only optimizer-order measurements;
  `driver_snapshot.py` is the exact source and `adam_order.pdf` / `.png` show
  the measurements. `plot_adam_order.py` regenerates these figures.

The test set is used for the fixed diagnostic endpoint and reporting, never
for training gradients or hyperparameter selection within this control.
