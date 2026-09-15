# Prompt for updating the project explainer

Update the existing AH-LoRA artifact from the corrected research evidence in this repository. Read `docs/artifacts/claude_handoff.json`, then `docs/artifacts/integrated-corrected/README.md`, `RESULTS.md`, `summary.json`, `validation.json`, and the updated paper in `paper/` before changing the artifact.

The original goal is to **match ordinary pooled-data LoRA accuracy**, while supporting resource-limited clients through adaptive ranks and decentralized training. Be explicit about whether the measured experiment achieves that goal. Outperforming pooled training is not required. Matching a weak federated baseline is insufficient.

Explain the implemented sequence: each client trains its own adapter under a 4/8/16 ceiling; the P1 controller changes ranks using training-only probes; the P2 policy computes bounded domain factors; reversible weighted Metropolis gossip mixes scaled effective updates; an actual simulated neighbor-tree reduction constructs the rank-16 deployment adapter. Compare that adapter with one conventional rank-16 LoRA trained directly on all 50,000 examples, using the same 10,000-example test set, scaling, frozen features, and initialization. State that this is a single-process simulation of peer communication on gpu003, not a tested multi-host hospital deployment.

Use the nine-arm table and paired seed-level differences to separate pooled-versus-federated behavior, uniform-versus-heterogeneous ranks, adaptive-versus-fixed ranks, and domain-versus-quality weighting. Use the final round for the main endpoint. Draw curves and error bars from the recorded data; three-seed error bars are sample standard deviations. Keep personalized accuracy secondary and clearly distinguish its local-model endpoint.

Include the corrected graphs in `paper/figures/integrated_*` or recreate them directly from the accompanying CSV files. Show domain-control traffic and final assembly alongside factor traffic, and label rank/sample savings as a proxy rather than measured total compute savings. Explain the mathematical operations naturally within the methodology.

The old `p3-e2e-ablation` comparison is superseded: Sinkhorn normalization erased its domain factors, its rank arm substituted a median-loss heuristic, its pooled baseline used a different alpha, and its assembly was a centralized diagnostic. Do not reuse that comparison as evidence of success, failure, or weighting neutrality. Keep it only as an explicitly labeled audit history. Preserve the valid isolated P1/P2 and historical P3 results with their own protocols.

Use the actual completed results even if negative. Do not imply that implementation completion means the accuracy objective succeeded. Do not infer privacy from local data or LoRA: updates and training histograms are exposed, no privacy guarantee was established, and this experiment does not validate hospitals, medical data, or arbitrary AI training.

Finish with a clear evidence-based conclusion and links to the paper, raw records, independent verification, and source snapshot. Remove stale conflicting completion statements throughout the artifact.
