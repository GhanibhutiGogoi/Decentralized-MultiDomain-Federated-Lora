# Experiment 05: direction-aware domain signatures

This experiment asks whether trained adapters reveal the five known CIFAR-100 domain groups. It compares the original spectral features with two signatures built from the scaled effective update `DeltaW = (alpha / rank) B A`:

| Signature | Clustering input | Algorithm |
|---|---|---|
| `spectral_baseline` | Original standardized singular values of A, B and BA, plus factor norms | Ward agglomerative clustering |
| `row_norms` | L1-normalized L2 norm of each DeltaW output row; cosine affinity | Average linkage on `1 - affinity` |
| `inverse_delta_l2` | `1 / (1 + distance)` between flattened DeltaW vectors | Average linkage on `1 - affinity` |

All three use a prespecified cluster count of five. Domain labels enter only the benchmark partition and ARI/NMI scoring. They do not influence signature construction, clustering or the flat gossip matrix. Client order is shuffled before constructing the flat topology because consecutive client IDs share domains in this testbed. Clustering order is also shuffled at each stage so ties do not inherit those ordered labels.

Training uses the same cached-head helpers as [Experiment 04](EXPERIMENT04.md). The feature cache has a frozen ResNet-18 in evaluation mode, deterministic resizing and ImageNet normalization, with no training augmentation. This differs from the original experiments 01–03. Shared initial head parameters, data splits and paired client minibatch random streams are held fixed across training modes within each seed. The original spectral formula is preserved, but this is a new training protocol rather than a numerical reproduction of experiment 03.

Run tests and experiments on the documented SSH GPU machine. From `project-3-hierarchical-gossip`:

```bash
python experiments/05_signature_validation.py \
  --output results/experiment_05_signature_validation \
  --methods local mh --seeds 42 43 44 --stages 2 5 10 20 \
  --ranks 16 --alpha 32 --consensus-rank 16 \
  --data-dir data --feature-cache data/features --device cuda
```

The output directory must be empty. Shared options from Experiment 04 control learning rate, local epochs, batch size, cache location and sample caps. Experiment 05's `--stages` determines total training rounds. Homogeneous ranks are required because the original spectral baseline has rank-dependent feature length. A single seed, short training or capped data must be labeled accordingly.

At stages 2, 5, 10 and 20, signatures are extracted after local training and, for MH, after mixing. Each stage also records **personalized accuracy** (mean client accuracy on its own test shard) and **consensus accuracy** (a uniformly merged adapter at the configured reference rank, evaluated on the union of test shards). These quantify the models being clustered but are never used to choose a signature or modify training.

The output files are:

- `manifest.json`: status, exact configuration, environment, cache and protocol provenance.
- `seed{seed}_{method}.json`: every training round, stage evaluations, signatures/affinity matrices, predicted assignments, scoring labels and hashes.
- `results.jsonl`: one record for each completed seed/mode run.
- `summary.csv`: ARI/NMI and both accuracy metrics per seed, mode, stage and signature.
- `gate_g1.json`: stage-10 ARI mean and sample standard deviation, with descriptive candidate bands (0.7 for hard clustering, 0.4 for soft affinity).
- `splits_seed{seed}.json`: exact client partition indices and training class counts.

ARI is adjusted for chance; one indicates exact agreement up to cluster-label permutation and zero is near chance. NMI lies between zero and one. Negative ARI is valid. Report all signatures, stages and seeds, including failures; candidate bands are a screening heuristic, not a statistical acceptance test.

This is an offline diagnostic that can inspect every adapter. A high ARI does not complete a decentralized discovery protocol, measure its communication overhead, establish privacy, or show that predicted clusters improve downstream accuracy. The oracle hierarchy benchmark remains explicitly oracle until those steps are implemented and evaluated.
