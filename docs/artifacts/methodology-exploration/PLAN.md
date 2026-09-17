# Methodology exploration, 2026-09-15

Status: running. This is an exploratory follow-up to the completed corrected
experiment, not a replacement for its negative result. User question: does the
methodology necessarily fail, or are correctable protocol choices responsible?

All numerical tests, training, and figure generation run on gpu003. The launch
uses an isolated source snapshot at `~/ahlora-exploration-20260915`, based on
commit `eed323a`; it does not change the existing remote working tree.

## First diagnostic stage (specified before these runs)

Use the same frozen CIFAR-100 features, 50,000 training examples, 10,000 test
examples, initialization, alpha 32, rank-16 reference, Adam hyperparameters,
30 epochs/rounds, and seeds 42/43/44 as the corrected experiment.

1. Reproduce pooled-reset and centralized FedAvg rank16.
2. Add exact rank16 SVD after every pooled-reset epoch; isolate changing the
   LoRA factor coordinates without intentionally reducing model rank.
3. Center the classifier update across output classes before SVD, in paired
   pooled-reset-SVD and FedAvg controls. A common logit shift is irrelevant to
   softmax but may dominate the Frobenius norm used by SVD.
4. Redistribute the same training examples IID while preserving every client's
   sample count; repeat FedAvg with and without centering. This diagnoses severe
   label skew rather than changing the official test set.
5. With no training, distribute an already-trained pooled adapter to peers at
   uniform16, fixed4/8/16, and adaptive-floor2/4/8 ranks, then repeat the current
   project-and-mix protocol. This tests information attrition. Pooled
   initialization is an oracle stress test, not a federated training result.
6. Verify that neighbor-exchanged gradients at a common model state reconstruct
   the pooled gradient; test synchronized rank16 training where feasible. This
   is a high-communication positive control, not the full proposed method.

Every attempted run, failure, configuration, source hash, final checkpoint, and
per-round observation must be retained. No test-dependent early stopping,
checkpoint selection, or undisclosed hyperparameter search. Later interventions
will be described as exploratory and logged before their runs. Reusing the
official test set for this investigation prevents treating improvements as a
fresh confirmatory equivalence result; promising variants need new validation.

Potential subsequent repair: retain an accumulated frozen global model, train
adaptive-rank *residual* adapters, and aggregate local changes rather than
capacity-truncated total model states. This changes the memory/capacity model
and must be reported as a variant, not a silent fix to the old result.
