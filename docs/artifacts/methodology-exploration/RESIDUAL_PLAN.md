# Prospective residual-protocol exploration

Specified before launching these candidate trials. This is a method variant:
rank caps restrict trainable local residuals, while each peer retains a larger
frozen global head. The previous state-averaging result remains valid.

Use a fixed class-stratified 45,000/5,000 split of the original training data
(50 validation examples/class, split seed 20260915). Candidate code opens only
the cached training tensor; official test tensors are not opened. All candidates
use the same split, local partitions, 30 epochs, original Adam settings, alpha32,
and rank16 deployment endpoint. Screening seed42 is exploratory. No selection
based on the official test set or intermediate stopping/checkpoints.

Finite initial candidate list:

- Conventional pooled LoRA, original fixed-rank/domain and adaptive-rank/domain.
- Retained-base residual LoRA: uniform16/sample weights/gain1;
  fixed4/8/16/sample/gain1; adaptive/domain/gains1,5,15;
  adaptive/sample/gain5; adaptive/quality-only/gain5;
  fixed4/8/16/domain/gain5.

The quality-only arm was added following independent design review, before the
screening launch. It isolates the domain multipliers from quality weighting.

Gains1,5,15 are a coarse, disclosed exploration of aggregation step size,
motivated by five disjoint label domains and fifteen separate local trajectories.
They are not theoretically derived optimal rates. One local epoch is used per
synchronization in every residual candidate; no extra training exposure.

At each synchronization peers begin with a common retained frozen head, train
zero-initialized adaptive/fixed-rank residual adapters, then perform an exact
neighbor-tree weighted reduction of the residual updates. The root rotates
through peers by a fixed rule. Add the aggregate times the declared gain to the
retained global update, center across output classes (softmax invariant), and
project once to deployment rank16. Broadcast the resulting frozen head on the
same tree. This is exact peer collective communication, not one-hop MH gossip.
Every peer, including one with a small trainable rank, must be able to perform
the aggregation SVDs when acting as root. The local function can have rank up
to16+r before the final global projection. These capabilities exceed the
original interpretation that the complete client adapter fits in rank r.

Track the extra frozen-head memory, dense tree messages, P2 control traffic,
training exposure, actual rank trajectories, and final validation accuracy.
Returning zero residuals must preserve the retained model regardless of local
rank. This invariant and the validation split are checked remotely before runs.

Any follow-up candidate or replication choice will be recorded separately before
launch. A promising outcome would establish a direction for repair; it would
not validate a privacy guarantee, clinical applicability, formal equivalence,
or the original smaller model-state memory assumption.
