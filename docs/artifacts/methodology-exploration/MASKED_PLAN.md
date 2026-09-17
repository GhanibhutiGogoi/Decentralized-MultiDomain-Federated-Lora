# Shared-model partial-gradient exploration

Registered before these runs, after observing poor retained-base residual
screening performance. That failed screen is retained in full.

This explicitly changes the algorithm: all peers retain common rank16 factors
and a synchronized global Adam state. A client computes factor gradients only
for r_i randomly selected rank coordinates per minibatch. Coordinates it does
not compute are missing observations, not zero-valued model components.
Multiply observed gradients by16/r_i before the peer allreduce. Conditional on
the current model/data/rank choice, this is an unbiased estimator of the full
factor gradient under uniform coordinate sampling. Apply one common Adam step
after reduction. No repeated SVD or model-state truncation is used.

This does not satisfy the original complete-model rank cap: full frozen factors
and optimizer state still exist, and the current transport sends zero-padded
rank16 gradient vectors. Any saving is in local gradient computation; total
memory/traffic savings are NOT claimed. A global minibatch schedule is used,
as in the synchronized positive control. Raw examples remain with their
simulated owner; no real multi-host isolation or privacy protection is tested.

Use the same45k/5k training holdout as residual screening. All evaluations are
validation, and only train.pt is opened. Fixed finite seed42 screen,30epochs,
same Adam lr0.001/weight_decay0.0001, alpha32, and batch128. Arms: fixed and
adaptive rank × sample weighting; adaptive quality-only; fixed and adaptive
rank × dynamic domain weighting. No intermediate checkpoint selection.
An additional uniform16/sample arm verifies this driver against the pooled
reference; it was added before the first numerical run of this variant.
The six full screening arms are split between two identical V100 GPUs on
gpu003 in separate output directories to shorten wall time; device/configuration
and source hashes are recorded. The one-epoch smoke checks remain separate.

Fixed ranks are4/8/16. Adaptive ranks use the canonical P1 controller and
training-only gradient/quality probes on a deterministic ceiling-sized subset
of shared coordinates (changed probe integration is declared). Dynamic P2
weights use training histograms, training-probe quality, and the local probe
gradient's effective first-order update as the domain signal. This differs
from P2 inputs obtained after a whole local epoch; it must not be called the
unchanged original pipeline. Weights refresh once per epoch.

Tests: partial gradients match autograd on active coordinates; enumeration of
all coordinate subsets recovers the full gradient in expectation; no missing
model coordinates are erased; allreduce matches a direct test oracle; train/
validation disjointness and all exposure/rank/count invariants.
