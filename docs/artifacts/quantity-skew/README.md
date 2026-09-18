# Quantity-skew decentralized LoRA comparison

This is the new RoBERTa/SST-2 study against an independent reimplementation of published Dec-LoRA. It is separate from the completed CIFAR-100 disjoint-domain diagnostics in `../methodology-exploration/`.

The prospective protocol and all reconstruction assumptions are in [the protocol](../../research/2026-09-17-quantity-skew-protocol.md); [the primary-source literature audit](../../research/2026-09-17-paper-baseline-shortlist.md) explains paper selection. Model/dataset revisions, run source snapshots, exact client memberships, actual serialized graph transfers, adaptive-rank probes, predictions, checkpoints and metrics are recorded by the runner.

## Current evidence

- Remote numerical/software checks: 119 tests passed on gpu003.
- Seven matched real-model smoke runs: completed, with four rounds, two local steps and a 640-example training subset. All seven best/final checkpoint pairs reproduced exactly using the independent evaluator.
- Smoke results establish execution and auditability only. They do not establish scientific efficacy, equivalence or a win over Dec-LoRA.
- Full-data comparison: running under the prospective protocol in a durable four-GPU queue. The fixed plan is 36 runs: seven quantity arms across five seeds plus one equal-size paper-setting anchor. Completed full-budget, replicated results must replace this status before any scientific success claim.

### Latest access attempt: 2026-09-18, 15:18 Shanghai

SSH to gpu003 reset the connection before authentication on four consecutive attempts. Live campaign completion could not be checked. This is an access failure, not evidence that training failed or stopped. The last confirmed experimental snapshot remains the 13:18 record below; its remaining-time estimate is historical. No jobs were restarted or modified. See [`monitoring-events.jsonl`](monitoring-events.jsonl). Final five-seed analysis, evidence transfer, and paper integration await access to the remaining remote records.

### Last confirmed progress: 2026-09-18, 13:18 Shanghai

Thirty-two of36 full runs have completed and passed independent best/final checkpoint audits. The last four seed46 runs are training, with roughly1.5 training hours remaining plus verification/report overhead. No campaign error is recorded.

The table compares only the same four completed seeds42–45 across all seven methods. Best labeled-validation accuracy is the registered primary endpoint; final-round accuracy is secondary. Values are mean ± sample standard deviation.

| Method | Best validation accuracy | Final-round accuracy |
|---|---:|---:|
| Dec-LoRA reimplementation, rank16 | 94.50 ± 0.25% | 94.38 ± 0.32% |
| Dec-LoRA reimplementation, rank4 | 94.47 ± 0.38% | 94.27 ± 0.43% |
| Effective products, rank16, sample-size weights | 94.75 ± 0.20% | 94.41 ± 0.25% |
| Fixed heterogeneous ranks, uniform weights | 94.58 ± 0.25% | 94.24 ± 0.36% |
| Fixed heterogeneous ranks, sample-size weights | 94.44 ± 0.11% | 94.29 ± 0.20% |
| Adaptive ranks, uniform weights | 94.27 ± 0.09% | 93.89 ± 0.55% |
| Adaptive ranks, sample-size weights | 94.29 ± 0.25% | 93.92 ± 0.21% |

The proposed adaptive/sample method trails rank16 Dec-LoRA by0.201 percentage points on mean best accuracy, with an exploratory paired95% t interval of[-0.746,+0.344] points. It trails fixed/sample by0.143 points and rank4 by0.172 points. At the final round its mean gap from rank16 is0.459 points. These interim four-seed results show no accuracy advantage; they establish neither superiority nor parity. The final fifth seed remains pending.

Mean adaptive training payload is1.292GB versus1.893GB for rank16 Dec-LoRA (31.74% lower),1.468GB for fixed/sample (12.00% lower), and1.185GB for rank4 (adaptive uses9.02% more). Mean persistent adaptive training rank is5.8125. Rank and payload reductions do not establish lower total memory: the backbone, classifier and capability-rank probes remain. Final deployment and evaluation assembly costs are separate from these training-byte totals.

Completed records and source snapshots are in [`full/`](full/). The dated [progress report and figures](progress-summary-20260918-1318/SUMMARY.md) separate completed runs from smoke and ongoing runs. Its available per-arm seed counts differ; only matched seeds enter paired comparisons. This README and [`pilot-progress.json`](pilot-progress.json) explicitly restrict the current main comparison to the four seeds completed by every arm. The report's final-score column label is editorially clarified as secondary; numerical data are unchanged. The running remote campaign retains its pinned analysis script until completion.

Work is delivered directly to `main`. New paper conclusions await the completed five-seed comparison. Earlier dated snapshots are preserved.

[`campaign-status.json`](campaign-status.json) and [`campaign-events.jsonl`](campaign-events.jsonl) are dated repository snapshots, not a live service. The authoritative live records are `~/ahlora-quantity-20260917/campaign-full-v1/` on gpu003. The initial scheduler estimate was about21 remaining training hours, excluding verification/report overhead. Four existing runs were adopted without restarting or changing their source. Remote lifecycle checks verified successful and failed worker exit receipts, duplicate-worker rejection, and the single-scheduler lock.

The scheduler automatically verifies each completed run, stops new launches on execution/verification/report errors, and generates CPU-only tables and plots in `~/ahlora-quantity-20260917/campaign-summary/`. It launches the remaining seed42 controls first; seeds43–46 start only after all eight screen runs pass verification. There is no accuracy threshold for continuing: negative outcomes are retained and replicated. Source, interpreter, packages, and analysis code are pinned. Final results still require research review and paper integration.

To inspect progress **on gpu003**:

```bash
cat ~/ahlora-quantity-20260917/campaign-full-v1/status.json
tail -n 20 ~/ahlora-quantity-20260917/logs/campaign.log
```

If the scheduler has stopped normally, resume with the same pinned scripts and environment:

```bash
~/ahlora-venv/bin/python ~/ahlora-quantity-20260917/repo/scripts/run_quantity_campaign.py \
  --base ~/ahlora-quantity-20260917 --python ~/ahlora-venv/bin/python \
  --summary-script ~/ahlora-quantity-20260917/repo/scripts/summarize_quantity_benchmark.py
```

The lock prevents a second scheduler. A blocked campaign requires diagnosis; do not clear its failure record or overwrite a failed output directory to manufacture a successful run.

The `smoke/` folders preserve the actual smoke source snapshots and records. They precede final hardening/provenance additions in source commit `8ae7c2e`; full-budget runs archive their own source and must not be pooled with smoke results. Small saved models remain on gpu003; checkpoint SHA256 values and verification records identify them. Source assets are public, pinned Hugging Face RoBERTa/GLUE revisions; the large foundation-model weights are not committed.

The training runtime is a single-process peer-transport simulation on gpu003. It is not a multi-machine deployment and provides no cryptographic privacy guarantee. The final adapter is assembled at a preselected rank16-capable peer; weaker peers do not install it. Whole-process peak GPU memory is not a per-client memory measurement.

## Reproduce and interpret

Run commands and comparisons must use the recorded config and source snapshot. The main quantity experiment fixes 211 local updates of exactly32 examples per peer per round, for20 rounds. Sample-size weighting is static `n_i/N`. The equal-size, one-local-epoch Dec-LoRA anchor is a separate reconstruction setting. The seven quantity arms include uniform rank16 and rank4 factor-gossip baselines, an unconstrained product-space control, and the fixed/adaptive by uniform/sample-weighted factorial.

The primary endpoint follows the paper's best labeled-validation accuracy; final accuracy and resource costs are secondary. Canonical GLUE counts and several unspecified paper hyperparameters are disclosed assumptions. Our changed-partition scores cannot be compared directly with the publication's printed93.81%. No hidden test result or performance-parity conclusion is implied.
