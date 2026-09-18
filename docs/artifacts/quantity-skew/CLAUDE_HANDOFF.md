Update the project artifact from the completed quantity-skew experiment. Preserve the earlier negative disjoint-domain results as a separate study. Do not portray the project as still waiting for training: all 36 planned full runs completed at 14:50 Shanghai on 2026-09-18, including 35 matched quantity runs and one equal-size anchor. All 72 best/final full checkpoints passed independent verification. The seven smoke runs are execution checks, not extra scientific seeds.

Read these files first:

- `docs/artifacts/quantity-skew/README.md`
- `docs/artifacts/quantity-skew/final-summary-20260918/SUMMARY.md`, `summary.json`, `per_run.csv`, and `per_round.csv`
- `docs/artifacts/quantity-skew/final-audit/`
- `docs/artifacts/quantity-skew/final-figures/` and `paper/figures/quantity-skew-*`
- `docs/research/2026-09-17-quantity-skew-protocol.md` and the paper-baseline shortlist
- `paper/main.tex`, `paper/main.pdf`, and the source-pinned per-run records under `docs/artifacts/quantity-skew/full/`

Earlier dated progress folders and `pilot-progress.json` are superseded historical snapshots. The final generated report counts 43 complete entries because it includes seven smoke runs; keep its three protocol groups separate. The frozen report accidentally labeled its final column primary; `SUMMARY.md` corrects that label, while `SUMMARY.raw.md` preserves the exact original report. The data are unchanged.

Explain the revised question: clients train on disjoint unequal-size shards of the same SST-2 task, with almost identical class proportions, independently assigned rank ceilings, a ten-peer ring, and sample-size-weighted global adapter assembly. This is not the original different-topic/domain partition. The competitor is an independent reconstruction of published Dec-LoRA on RoBERTa-base, with unspecified author details and a dataset-count discrepancy explicitly disclosed. Sample weighting and effective-product aggregation are existing ideas; the study measures their combination with adaptive ranks and peer training.

Show all seven methods over the same five seeds (42–46). Best official labeled-validation accuracy is primary, final accuracy secondary; use mean ± sample SD. Key results:

| Method | Best validation | Final validation | Training GB |
|---|---:|---:|---:|
| Dec-LoRA rank 16 | 94.541 ± 0.238% | 94.450 ± 0.320% | 1.893 |
| Dec-LoRA rank 4 | 94.495 ± 0.334% | 94.266 ± 0.372% | 1.185 |
| Product rank 16 / sample | 94.725 ± 0.181% | 94.427 ± 0.224% | 1.893 |
| Fixed / uniform | 94.518 ± 0.262% | 94.197 ± 0.320% | 1.468 |
| Fixed / sample | 94.404 ± 0.126% | 94.220 ± 0.238% | 1.468 |
| Adaptive / uniform | 94.174 ± 0.221% | 93.807 ± 0.513% | 1.296 |
| Adaptive / sample | 94.243 ± 0.249% | 93.899 ± 0.188% | 1.291 |

Adaptive/sample minus rank 16 is -0.298 percentage points at the primary endpoint, exploratory paired 95% t interval [-0.755, +0.159]. Its final-round difference is -0.550 points [-0.919, -0.182]. Against rank 4 its best-score gap is -0.252 points [-0.684, +0.180]. A confidence interval containing zero does not establish parity; no noninferiority margin was prespecified. These exploratory intervals are unadjusted across comparisons.

Traffic savings are 31.81% versus rank 16 and 12.08% versus fixed/sample, but adaptive uses 8.92% more than rank 4. The rank-4 control is cheaper and has higher mean accuracy. Across the 2×2 heterogeneous-rank design, adaptation's mean best-score effect is -0.252 points; sample weighting's is -0.023 points. Do not claim that adding sample weights solved accuracy or that adaptive rank is necessary to achieve these scores.

Use the checked figures to explain accuracy over rounds, paired seed differences, the traffic/accuracy tradeoff, and rank trajectories. Show the fixed work budget: 20 rounds × 10 clients × 211 local steps × 32 examples = 1,350,400 example exposures per quantity run, plus adaptive probes. Clarify classifier traffic, ceiling-rank probe work, final assembly versus training messages, and the separate one-seed equal-size anchor. Do not compare the changed-partition scores to the paper's printed 93.81% as a win.

Keep terminology consistent: adaptive rank, static sample-size weighting, factor gossip, effective-update gossip, final assembly. Put mathematical explanations naturally into the method explanation. Source formulas are in the protocol and manuscript. Report a single-process simulation with serialized graph transfers, not a physical multi-machine deployment. Lower persistent rank is not a demonstrated reduction of total GPU memory. The assembly peer sees adapter records; no privacy guarantee or hospital-readiness claim was established.

End with the actual conclusion: the planned experimental project is completed and reproducible, but the proposed combination has not demonstrated an advantage or parity over the decentralized controls. It reduces traffic relative to rank 16 at a small observed accuracy cost, while rank 4 is the stronger practical control. Preserve the historical pooled-versus-domain failure and its diagnostics. These scoped negative findings do not prove that all adaptive decentralized LoRA methods must fail.
