# Experiment 04: reproducible completion benchmark

Run this experiment on the documented SSH GPU machine. It compares local-only,
client-uniform centralized ΔW FedAvg, flat Metropolis–Hastings gossip, and
hierarchical gossip with known CIFAR-100 superclass domains. A result does not
require hierarchy to win: this is an empirical, reproducible system benchmark.

## Run the complete battery with one command

On the documented SSH GPU machine, use a new empty output directory:

```bash
cd project-3-hierarchical-gossip
python experiments/run_completion.py --output-root results/completion \
  --data-dir data --feature-cache data/features --device cuda
```

This runs the fixed completion protocol serially: uniform rank 16 with all four
methods; heterogeneous ranks 4/12/32 with all four ΔW methods, the three factor
merging methods, and MH/oracle error feedback; then aggregate plots and CSVs;
then local/MH signature validation at rounds 2, 5, 10, and 20. Experiment 04 uses
50 rounds, seeds 42/43/44, shared alpha 32, and consensus rank 16. All jobs reuse
the same verified 224×224 feature cache and the full official dataset.

`battery_status.json` records the exact child arguments, job states, selected
compute environment, timestamps, exit codes, and output paths. Each job has its
own `logs/*.log`. A nonzero exit or missing completion marker stops the battery;
completed evidence remains intact. The command never resumes or overwrites an
existing output root. The aggregate evidence is in `report/`; signature results
and the descriptive G1 screen are in `signatures/`.

## Run individual experiments

```bash
cd project-3-hierarchical-gossip
python experiments/04_protocol_benchmark.py --output results/exp04-main \
  --data-dir data --feature-cache data/features --device cuda \
  --seeds 42 43 44 --rounds 50 --ranks 16 --alpha 32 --consensus-rank 16
```

The first command downloads verified CIFAR-100 and ImageNet ResNet-18 weights,
then caches the official 50,000/10,000 train/test representations. To separate
feature preparation from training, run with `--prepare-only --output results/preparation`.
Every invocation requires a new empty output directory to prevent stale mixing
or accidental replacement. Completed method/seed results survive later failures.

For the completed real-data smoke protocol, replace the round/seed options with
`--rounds 2 --seeds 42` and keep the rank/alpha/reference-rank settings above.
Optional `--max-train-per-client 128 --max-test-per-domain 100` caps are useful
for smaller execution checks, but were not used in the full-data battery. They
still build the full reusable feature cache. Smoke and subset runs are labeled
in the manifest and cannot support multi-seed full-data conclusions.

For heterogeneous ranks, run separate paired batteries:

```bash
python experiments/04_protocol_benchmark.py --output results/exp04-heterogeneous-delta \
  --data-dir data --feature-cache data/features --device cuda \
  --seeds 42 43 44 --rounds 50 --ranks 4 12 32 --alpha 32 --consensus-rank 16 \
  --methods local fedavg mh oracle
python experiments/04_protocol_benchmark.py --output results/exp04-heterogeneous-factor \
  --data-dir data --feature-cache data/features --device cuda \
  --seeds 42 43 44 --rounds 50 --ranks 4 12 32 --alpha 32 --consensus-rank 16 \
  --methods fedavg mh oracle --merge factor_zero_pad
python experiments/04_protocol_benchmark.py --output results/exp04-heterogeneous-feedback \
  --data-dir data --feature-cache data/features --device cuda \
  --seeds 42 43 44 --rounds 50 --ranks 4 12 32 --alpha 32 --consensus-rank 16 \
  --methods mh oracle --error-feedback
```

These are the settings of the active completion battery: uniform rank 16 has
total rank `15 × 16 = 240`; the heterogeneous cycle has total rank
`5 × (4 + 12 + 32) = 240`. Every method uses alpha 32 and consensus rank 16.
The heterogeneous ranks are fixed assignments, not the adaptive rank policy.
Output/cache paths may differ on the SSH machine; its manifests record the exact
commands and verified cache identity. Results are still running as of
12 September 2026; consult the [handoff](../docs/artifacts/claude_handoff.json)
for completion status before citing numbers.

## Exact protocol and limits

- A shared pretrained ResNet-18 backbone runs once in evaluation mode; all
  BatchNorm buffers remain fixed. Images are deterministically resized to 224×224
  (bilinear, antialias) and ImageNet normalized, without augmentation. Cached
  feature and weight SHA-256 hashes identify the actual representation. This is
  a frozen-feature classification benchmark, distinct from experiments 01–03.
- The shared random frozen 100-class linear head and LoRA initialization are
  paired within each seed. Larger ranks share the first rows of the same initial
  A; every B starts at zero. All methods use the same fixed alpha and rank cycle.
- Each of five known superclass domains has three clients. Training uses a
  within-domain Dirichlet split; testing uses disjoint domain test shards. Full
  indices and class counts are saved. Positive sample caps explicitly label a
  subset run. The flat topology permutes client IDs by seed before ring creation,
  so sequential IDs do not give it a domain-ordered topology.
- Adam resets once per local-training round in every arm; each client has its own
  paired minibatch generator. Local-only skips communication and SVD
  reparameterization. All communication arms use the same runner kernels.
- FedAvg uses a uniform client objective to match doubly stochastic gossip. It is
  not the sample-weighted legacy `FedAvgServer`. The full matrix is applied after
  local updates; each receiver refactorizes at its own rank.
- Oracle hierarchy has complete within-domain exchange and uniform representative
  bridging every five rounds. Domains are supplied, never discovered; there are
  no learned transfer weights. Its spectral gap is measured over a full period.
- Personalized accuracy averages each client's accuracy on its own disjoint test
  shard. Its separately named sample-weighted version counts all examples.
  Per-domain accuracy is sample-weighted. Consensus accuracy evaluates the
  uniform mean effective update, compressed at `--consensus-rank` (default maximum
  client rank), against the union of test shards. Oracle domain membership is not
  supplied to the consensus model. Evaluation does not mutate client models.
- Fixed-round final metrics are reported. No test-set hyperparameter selection,
  best-round winner selection, or expected ordering is a pass/fail condition.

## Communication and compression measurements

`effective_messages/floats` charge every off-diagonal contributor in the applied
mixing matrix, sending the source's low-rank factors. `operational_messages/floats`
describe an exact transport schedule: FedAvg gathers factors and broadcasts the
dense average ΔW; hierarchy exchanges factors initially, then dense exact ΔW
intermediates in representative and second intra stages. Exact dense intermediates
avoid introducing extra SVD compression into the effective mixing matrix. The
factor baseline instead transports padded factor intermediates. These are
simulated payload counts, excluding setup and evaluation; fp32 bytes are four
times floats. No actual distributed network latency or throughput is measured.

For ΔW merging, `mean_tail_mass` is relative squared SVD residual and
`mean_residual_energy` is absolute squared residual. Error feedback may increase
these while retaining residual memory for later rounds. For naive factor merging,
both SVD quantities are null; `mean_merge_error_energy` and
`mean_relative_merge_error` measure its discrepancy from the effective ΔW target.
Factor error is not SVD tail mass and need not satisfy its bounds.

## Data for artifact creation

- `manifest.json`: status, run classification, exact configuration, protocol,
  Git/environment/feature provenance, total runtime, completed-run index.
- `seed{seed}_{method}.json`: atomic per-round checkpoint, initial metrics,
  paired split/initialization hashes, rank map, topology order, spectral gap,
  complete personalized/consensus/domain/communication/compression history.
- `splits_seed{seed}.json`: all train/test indices, domain IDs, training class
  counts, and split checksum.
- `results.jsonl`: completed runs only, one complete JSON record per method/seed.
- `summary.csv`: final-round metrics and total costs, one row per completed run.

Use the exact seed-level rows to compute mean and standard deviation across
seeds, and paired method differences within seeds. Preserve unsuccessful or
contradictory findings; the benchmark is useful even when hierarchy does not
improve consensus accuracy or when personalization requires more communication.

Generate the aggregate handoff on the SSH machine after the runs finish:

```bash
python experiments/summarize_completion.py \
  --inputs results/exp04-main results/exp04-heterogeneous-delta \
           results/exp04-heterogeneous-factor results/exp04-heterogeneous-feedback \
  --output results/completion-report
```

The report writes `aggregate.json`, raw and aggregated seed/round/domain CSVs,
all signed paired differences, and PNG/SVG figures. It keeps incompatible
training/cache/source protocols in separate contexts and checks split and
initialization hashes before pairing seeds. Duplicate evidence is rejected.
Means, sample standard deviations, missing runs, and pairing exclusions remain
explicit; there is no favorable-result filtering or automatic winner claim.
