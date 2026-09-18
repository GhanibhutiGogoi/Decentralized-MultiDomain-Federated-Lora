# Quantity-skew decentralized LoRA comparison

**Complete.** All 36 planned full runs finished on gpu003 at 14:50 Shanghai on 2026-09-18 and passed independent best/final checkpoint verification. The study contains seven methods across five matched seeds (42–46), plus one separate equal-size Dec-LoRA anchor. Seven smoke runs and 119 remote tests are supporting execution checks. They are not additional scientific replications.

This RoBERTa-base/SST-2 study compares our documented independent reconstruction of published Dec-LoRA with adaptive ranks and sample-size-weighted peer training. It is separate from the completed CIFAR-100 disjoint-domain diagnostics in [`../methodology-exploration/`](../methodology-exploration/). The [prospective protocol](../../research/2026-09-17-quantity-skew-protocol.md) and [literature audit](../../research/2026-09-17-paper-baseline-shortlist.md) record the design and reconstruction assumptions.

## Final five-seed results

Values are mean ± sample standard deviation. Best accuracy on the official 872-example labeled validation split is primary, following the selected paper's evaluation endpoint. Final-round accuracy is secondary. There is no hidden-test result.

| Method | Best validation accuracy | Final-round accuracy | Mean training traffic |
|---|---:|---:|---:|
| Dec-LoRA reconstruction, rank 16 | 94.541 ± 0.238% | 94.450 ± 0.320% | 1.893 GB |
| Dec-LoRA reconstruction, rank 4 | 94.495 ± 0.334% | 94.266 ± 0.372% | 1.185 GB |
| Effective products, rank 16, sample weights | 94.725 ± 0.181% | 94.427 ± 0.224% | 1.893 GB |
| Fixed heterogeneous ranks, uniform weights | 94.518 ± 0.262% | 94.197 ± 0.320% | 1.468 GB |
| Fixed heterogeneous ranks, sample weights | 94.404 ± 0.126% | 94.220 ± 0.238% | 1.468 GB |
| Adaptive ranks, uniform weights | 94.174 ± 0.221% | 93.807 ± 0.513% | 1.296 GB |
| Adaptive ranks, sample weights | 94.243 ± 0.249% | 93.899 ± 0.188% | 1.291 GB |

The proposed adaptive/sample method is **0.298 percentage points below rank 16** on the primary endpoint. Its exploratory paired 95% Student-t interval is **[-0.755, +0.159] points**. Against rank 4 the gap is -0.252 points, interval [-0.684, +0.180]. Against fixed/sample it is -0.161 points, interval [-0.399, +0.078]. The final-round gap from rank 16 is -0.550 points, interval [-0.919, -0.182]. These are unadjusted, small-sample intervals, not a prespecified noninferiority test.

Adaptive/sample saves **31.81%** of training traffic versus rank 16 and **12.08%** versus fixed/sample, but uses **8.92% more** than rank 4. The rank-4 control therefore has both lower communication and higher mean accuracy. Across the heterogeneous 2×2 experiment, adaptation has a mean best-accuracy effect of -0.252 points (exploratory interval [-0.382, -0.123]); sample weighting has a mean effect of -0.023 points ([-0.152, +0.106]). The weighting-by-adaptation interaction is +0.183 points ([-0.201, +0.568]). Weighting changes the mixing kernel as well as the objective, so it is not an objective-only intervention.

**Conclusion:** the planned experiment is complete, but it does not demonstrate a benefit over the simpler decentralized alternatives or establish performance parity. The result is a measured accuracy/communication tradeoff against rank 16, with a stronger practical challenge from rank 4. This is not an impossibility theorem for adaptive decentralized LoRA. No privacy guarantee follows from these results.

## Evidence and interpretation

- [`final-summary-20260918/SUMMARY.md`](final-summary-20260918/SUMMARY.md), JSON, CSVs and plots contain the complete campaign report. Its 43 complete entries mean 36 full runs plus seven smoke runs. The three protocol groups stay separate. The final-score column label is corrected to secondary; `SUMMARY.raw.md` preserves the exact frozen-script report.
- [`full/`](full/) contains all 36 completed run records, source snapshots, predictions, membership hashes, transfer ledgers and checkpoint audits. Best/final checkpoint binaries remain on gpu003, with their hashes bound to the verification records; they are not included in Git. A 1.6 GB archive of all full-run checkpoints was also transferred to the ignored local `checkpoints/quantity-skew/full-checkpoints-20260918.tar`; [`checkpoint-backup.sha256`](checkpoint-backup.sha256) records its remote source hash.
- [`final-audit/`](final-audit/) contains the independent final evidence and statistical audit. [`final-figures/`](final-figures/) records the manuscript figure data and generation provenance.
- [`campaign-status.json`](campaign-status.json) and [`campaign-events.jsonl`](campaign-events.jsonl) preserve final scheduler state. [`monitoring-events.jsonl`](monitoring-events.jsonl) records the temporary SSH outage and later successful retrieval. Earlier progress snapshots, including `pilot-progress.json`, are historical and are superseded by these completed results.
- [`CLAUDE_HANDOFF.md`](CLAUDE_HANDOFF.md) explains how to update the external artifact. The manuscript and its source are in [`../../../paper/`](../../../paper/).

The single equal-size anchor reaches 94.151% best/final validation accuracy. It uses one local epoch per round and is not pooled with the quantity experiment. Missing author implementation details and a dataset-count discrepancy prevent claims of an exact numerical reproduction. Do not compare our changed-partition scores directly against the paper's printed 93.81%.

## Protocol and resource scope

All 67,349 training examples are allocated once across ten disjoint, class-stratified client shards in size ratios `[1,1,1,1,2,2,2,4,4,4]`. Independently shuffled rank capacities are `[4,4,4,4,8,8,8,16,16,16]`. Each quantity run performs 20 rounds, 211 local updates per client per round, and batches of 32: exactly 1,350,400 training-example exposures. Adaptive probes add separately recorded work. All arms use identical partitions, sample streams, optimizer conventions and nominal training budgets for each seed.

Dec-LoRA mixes factors on a ring. The effective-product arms mix `(alpha/rank) B A` and apply compact QR/SVD truncation. Sample weights are the static empirical-risk weights `n_i/N`, implemented through reversible weighted Metropolis-Hastings gossip and final assembly. Sample weighting and effective-product aggregation are established ideas; they are not claimed as novel mechanisms.

Training executes as a **single-process simulation of peer transport**, with actual serialized messages on declared graph edges. It is not a physical multi-host deployment. A preselected capacity-16 peer assembles the final adapter through neighbor-tree routing. Weak peers do not install a global rank-16 adapter. The assembly peer sees routed adapters, and decentralized routing provides no cryptographic privacy guarantee.

Reported training GB are actual serialized training payloads, including classifier traffic. Final assembly and repeated validation assemblies are separate. Production accounting includes modeled count setup (320 bytes), training messages and one final assembly; base-model, initial-state and topology provisioning are excluded. Lower adapter rank is not a demonstrated reduction in total GPU memory: the shared backbone, classifier, ceiling-rank probes and assembly workspace remain. Whole-process CUDA peaks are not independent-client peaks. Rank-16 reference arms exceed weaker clients' stated adapter-rank caps; rank 4 is feasible for every client.

## Reproduction and delivery

Use each run's config and source manifest with the pinned model/data revisions in the protocol. All tests, scientific analyses and figure generation execute on gpu003. The campaign was run in `~/ahlora-quantity-20260917/repo` and completed without a metric gate: unfavorable results were retained. The scheduler verifies each run's best/final checkpoints and pins source, interpreter, packages and analysis code. Do not restart the completed campaign to manufacture a favorable result.

This experiment now has its own [standalone paper](../../../paper/quantity-skew/main.pdf), [LaTeX source](../../../paper/quantity-skew/main.tex), and local figure collection. The original multi-domain experiment is a [different paper](../../../paper/multidomain/main.pdf). The [split-paper build records](../paper-split-20260919/) describe the current deliverables. The earlier 38-page combined version is retained in the [combined archive](../../../paper/archive/combined-20260918/); its [historical build record](paper-build/README.md) still identifies that exact PDF.

Final artifacts are delivered directly to `main`. Training checkpoints remain in the corresponding `~/ahlora-quantity-20260917/runs/` directories on gpu003. Large model weights and private machine access details are not committed.
