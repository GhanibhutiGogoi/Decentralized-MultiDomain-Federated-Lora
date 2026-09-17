# Methodology exploration archive

This preserves the completed investigation of the original disjoint-domain
protocol, including its negative results. It is separate from the subsequently
requested comparison with published decentralized methods under unequal client
dataset sizes. All training, numerical analysis, checkpoint verification and
figure generation ran on gpu003. The September 17 archival pass performed no
new training.

| Artifact | What it establishes | Evaluation scope |
|---|---|---|
| `optimization-v2/` | Seven controlled optimization interventions; three seeds | Original 50,000 train / 10,000 official test |
| `optimization-failed-attempt/` | Preserved earlier failed attempt | Historical debugging evidence |
| `rank-retention/` | No-training propagation of an already-trained model through heterogeneous rank projection | Diagnostic, not a trained replacement |
| `gradient-equivalence/` | Exact peer-gradient control and independent saved-model verification | Original full-data task |
| `residual-screen-v1/` | Eleven final checkpoints: eight residual recipes and three controls; all residual recipes unsuccessful | Seed42; 45,000 fit / 5,000 validation |
| `masked-screen-a/`, `masked-screen-b/` | Six partial-gradient variants | Seed42; same fixed training holdout |
| `masked-replication-43/`, `masked-replication-44/` | Uniform shared-gradient, fixed/domain and adaptive/domain replications | Seeds43/44; same fixed training holdout |
| `pooled-validation-replication/` | Ordinary conventional LoRA references | Seeds43/44; seed42 reference reused from residual screen |
| `masked-summary/` | Independent 15-checkpoint evaluation, paired statistics, curves, source/data hashes and report | Validation only; official test unopened |

The new independent summaries are
[`residual-screen-v1/SUMMARY.md`](residual-screen-v1/SUMMARY.md) and
[`masked-summary/SUMMARY.md`](masked-summary/SUMMARY.md). Their JSON/CSV files
contain machine-readable results, resource accounts and verification records.
All per-round records, checkpoints, plans and stdout logs are retained. A
record field named `full_test_accuracy` in a reused residual/pooled driver refers
to the explicitly identified 5,000-example training holdout in these folders;
it does not mean the official test set was opened.

## Completed validation findings

| Partial-gradient comparison | Validation accuracy, mean ± sample SD (%) |
|---|---:|
| Conventional pooled LoRA | 57.18 ± 0.71 |
| Uniform shared-gradient control | 57.18 ± 0.71 |
| Fixed partial gradients + domain weights | 55.90 ± 0.59 |
| Adaptive partial gradients + domain weights | 55.03 ± 1.01 |

These are three paired seeds, each trained for 30 epochs on 45,000 examples.
Adaptive/domain minus conventional LoRA is −2.15 ± 1.69 percentage points;
adaptive/domain minus fixed/domain is −0.87 ± 0.86 points. These SDs describe
between-seed variability on one common validation holdout, not confidence bounds
or an equivalence claim. The three final uniform-control prediction vectors
equal their pooled counterparts. Intermediate correct counts match in 89/90
evaluations; the exception is recorded in `masked-summary/summary.json`.

The partial-gradient method still requires full rank16 factors, full-rank
forward computation, full Adam state and padded full-rank gradient transport.
Its rank controls computed gradient coordinates only. It therefore does not
satisfy the original whole-adapter memory-cap claim. It also uses synchronized
global minibatches and rotating peer-tree collectives, replacing the original
one-hop model-state gossip protocol. No privacy guarantee is established.

All eight retained-base residual recipes failed the seed42 screen, with
validation accuracy between 1.42% and 2.88%. The high-gain recipe produced severe
but finite loss growth. These failures reject the disclosed recipes at this
budget, not all possible retained-state methods.

## Independent verification and reproducibility

The residual verifier checked 11 saved checkpoints and the partial-gradient
verifier checked 15, with one reused pooled checkpoint in both: 25 unique saved
models. Every independent CPU float32 correct count matched. The verifiers
import no project model/forward/evaluator code, open only `train.pt`, check its
tensor hash, and validate the exhaustive disjoint 45,000/5,000 indices with 50
validation examples per class. They also check initialization/partition pairing,
rank ceilings, sample exposures, weight formulas and recorded transport ledgers.
Per-step gradient edges were not logged; their paths cannot be independently
reconstructed from these artifacts.

`masked-summary/source_snapshot.tar.gz` preserves 47 source files verified
against all six launch manifests plus the canonical P1/P2 dependency hashes.
`source_manifest.json` records those checks. One AppleDouble metadata file is
included solely because the original launch provenance included it; it is not
executed Python. `data_manifest.json` hashes inputs and residual outputs in the
six validation-run directories. Dataset tensors are not committed; checkpoints
contain adapters and their initial model parameters, not training examples.

Run from the repository `scripts/` directory on gpu003, substituting the actual
archive and feature-cache paths:

```sh
python summarize_residual_exploration.py \
  --input "$EXPLORATION_RUNS/residual-screen-v1" \
  --feature-cache "$FEATURE_TENSOR_DIRECTORY" \
  --paper-figure "$FIGURE_DIRECTORY/exploration_residual"
python summarize_masked_exploration.py \
  --input "$EXPLORATION_RUNS" \
  --feature-cache "$FEATURE_TENSOR_DIRECTORY" \
  --paper-figure "$FIGURE_DIRECTORY/exploration_masked"
```

The feature directory must directly contain the original `train.pt` and cache
`manifest.json`. These commands perform validation and plotting, not training.
PDF/PNG figure pairs are also preserved in `paper/figures/`. Paper text/PDF
integration is tracked separately; this archive does not assert that an older
paper PDF already includes these additions.
