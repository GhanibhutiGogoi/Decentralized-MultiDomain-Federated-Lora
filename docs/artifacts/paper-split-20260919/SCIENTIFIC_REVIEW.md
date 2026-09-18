# Independent review of the two standalone manuscripts

Date: 2026-09-19. Reviewer: the independent literature/source-review agent, separate from both manuscript authors.

## Scope and decision

The source review covers `paper/multidomain/main.tex` and `paper/quantity-skew/main.tex`, their local inputs, figure references, bibliographies, methodological equations, results, and limits on their conclusions. Both manuscripts pass this review with no unresolved scientific blocker. This review did not rerun experiments or model inference. It compares the manuscripts with frozen implementation sources and already completed remote verification artifacts; all numerical execution remains on `gpu003`.

The papers answer separate questions. The multidomain manuscript reports the original CIFAR-100 study and its completed mechanism investigations. The quantity-skew manuscript reports the new RoBERTa/SST-2 comparison with reconstructed Dec-LoRA. Neither paper requires the other manuscript to define its method, interpret its endpoint, resolve its equation labels, or compile its figures and bibliography. The previous combined manuscript is historical material, not the current scientific presentation.

## Original multidomain study

The standalone source was compared with the historical manuscript at commit `56869fc`. The original numerical results, core methodology, and corrected diagnostic findings are preserved. The changes add a focused title and related-work context, clarify the historical probe rule, and improve presentation. No SST-2, quantity-skew, or Dec-LoRA campaign result is imported as evidence for the original hypothesis.

The corrected comparison remains nine methods at seeds 42–44, with all 50,000 CIFAR-100 training examples and all 10,000 test examples, frozen ResNet-18 features, 15 clients, and 30 rounds. Its pooled rank-16 result is 57.23 ± 0.59% and its adaptive/domain result is 6.90 ± 2.03%. The distinction between equal example exposures and unequal optimizer-step counts is retained. The reported reduction of about 41.9% in factor traffic is not presented as an equivalent saving in total traffic: the domain-control exchange reduces the total saving to about 0.26%.

The method is self-contained: the training objective, scaling of LoRA products, capability controller, domain features and weights, weighted Metropolis mixing, local compression, assembly, and traffic accounting are defined locally. In particular, the historical gradient-probe description was checked directly against `docs/artifacts/integrated-corrected/source_snapshot.tar.gz`, member `project-3-hierarchical-gossip/experiments/protocol_benchmark.py`. This implementation includes a factor gradient in the stable-rank median only when its squared spectral norm exceeds `1e-12`, and returns 1 when no gradient qualifies. The manuscript now states that rule accurately. It is distinct from the new transformer's zero-inclusive probe and is not silently replaced by it.

The completed follow-up evidence remains appropriately separated:

- IID redistribution, pooled refactorization, no-training rank-retention stress tests, and Adam-order diagnostics are mechanism-specific interventions, not additive causal explanations of the full performance gap.
- The full-data synchronized-gradient control matches all recorded prediction vectors at all 90 seed/epoch endpoints, under full rank-16 state and common optimizer assumptions.
- The later residual and partial-gradient studies use the original-training holdout rather than reopening the official test. All eight unsuccessful residual recipes are retained.
- Adaptive partial gradients reach 55.03 ± 1.01% versus 57.18 ± 0.71% for the paired pooled validation reference, while retaining full rank-16 model and optimizer state. This does not satisfy the original complete-model resource claim.
- The masked driver's final predictions match at all three seeds; the single intermediate correct-count discrepancy is retained rather than confused with the separate 90/90 full-data control.
- Historical personalized/consensus, oracle hierarchy, automatic discovery, and isolated policy results remain qualified as supporting experiments with different assumptions and endpoints.

The conclusion correctly rejects the tested combination's original accuracy/resource claim without declaring decentralized LoRA impossible. It does not infer privacy from local data ownership or prediction agreement. LoRA, federated averaging, gossip, and decentralized optimization citations provide context without claiming that their theory proves this composed protocol converges or preserves accuracy. All referenced equation, figure, and table labels are supplied by the local manuscript and its local result-table input.

## Quantity-skew study

The standalone manuscript was checked against the frozen seed-42 adaptive/sample source under `docs/artifacts/quantity-skew/full/full-v1-quantity-adaptive_sample-seed42/source/` and the completed independent audit under `docs/artifacts/quantity-skew/final-audit/`. Its values agree with those remote audit results; another reviewer independently checked the quantity-study numerical presentation.

The campaign scope is stated correctly: seven quantity-skew methods at five paired seeds, plus one separate equal-size anchor, for 36 full-budget runs. The 35 matched runs supply the five-seed aggregates. The source-pinned model and dataset revisions, canonical 67,349/872 train/validation sizes, label-stratified disjoint shards, ring topology, independently shuffled quantity/capacity assignments, and common fixed local-step budget are explicit. The paper does not present the one-seed equal-size anchor as a causal estimate of quantity skew.

The methodology can be read independently of the original study. Its local equations define the mean cross-entropy objective, scaled query/value updates and frozen biases, full trainable classification head, factor and product aggregation, weighted Metropolis kernel and detailed balance, disagreement, compact QR/core-SVD factorization, projection error, rank controller, peer assembly, traffic, endpoints, paired intervals, and factorial effects.

Specific source checks confirm:

- The transformer probe includes all 48 LoRA factor gradients, including zero gradients, and divides by the squared spectral norm clamped below at `1e-30`. Its zero-gradient value is 0. Classifier gradients do not enter this median.
- The menu, capability mapping, minimum ranks, two warm-up rounds, demand EMA, quantization, hysteresis, two-request patience, quality EMA/alarm, and rank-restoration rule agree with the frozen controller. Residual-demand modulation is disabled.
- Expansion rescales existing factor coordinates to preserve the scaled effective update, adds seeded nonzero A directions with zero B columns, and reduction uses truncated product-space SVD.
- Static sample-size weights affect both the objective weights and the graph's mixing matrix. They are not a new dynamic domain-allocation method.
- The compact QR/core-SVD equations preserve the correct LoRA scaling and acknowledge workspace growth with contributing ranks. Product averaging is not confused with separate factor averaging.
- Final assembly occurs at a preselected capable peer, has charged transfer costs, is not broadcast to weaker clients, and does not feed the evaluated assembly back into training.
- Model-message bytes include the full classifier and metadata. The 320-byte count setup is modeled, provisioning/framing are excluded, and measured process peaks are not client-device memory measurements.

Best official validation accuracy is the primary endpoint and final-round validation accuracy is a separate required secondary endpoint. Each method selects its own best round. Adaptive/sample obtains 94.243 ± 0.249% best accuracy versus 94.541 ± 0.238% for rank-16 Dec-LoRA. The paired best-score difference is −0.298 points with an unadjusted 95% interval of [−0.755, +0.159]; this is not called parity or noninferiority. Its final-round deficit of 0.550 points is also retained. Feasible rank-4 Dec-LoRA obtains 94.495 ± 0.334% best accuracy with lower training traffic than the proposal. The negative weighting and adaptation findings and their uncertain exploratory contrasts are not hidden.

The comparator citation is the pinned REALM 2025 publication, DOI `10.18653/v1/2025.realm-1.24`. The baseline is explicitly an independent reconstruction, with no verified author implementation. The published-versus-canonical dataset-count discrepancy prevents a direct claim to beat the printed score. DeCAF is acknowledged as prior work for consensus over effective products followed by factorization; the product-space control is not labeled a faithful DeCAF reproduction or a novel merge operator. The local bibliography also identifies LoRA, FedAvg, RoBERTa, and GLUE.

The paper correctly distinguishes the 72 fresh-process best/final inference checks from the final artifact audit, which binds them to checkpoint bytes and reconstructs 720 stored round metrics without rerunning inference. Shared model/install helpers, absent historical raw wire buffers, absent per-run initialization hashes, validation selection, five-seed uncertainty, single-process execution, and lack of privacy guarantees are disclosed. No pooled RoBERTa result, medical validation, hidden-test result, or independent-device memory saving is invented.

The only editorial suggestion was to replace an ambiguous results-paragraph pronoun with “Product16/sample's final mean.” The author has accepted that clarification; it does not alter any result.

## Visual review

The reviewer individually inspected the remote-rendered images for multidomain pages 21–30 at `/tmp/dmfl-papers-20260919/multidomain/rendered/main-21.png` through `main-30.png`. All pass: plots, axes, legends, mathematical notation, captions, tables, and references are readable, with no clipping, overlap, or accidental blank page. These pages retain clear separation between corrected results, later holdout studies, and supporting historical figures. The historical heterogeneous-rank and isolated-policy figures are legible and their different experimental scopes are explained in their captions.

The parent reviewer inspected original-study pages 1–10, and its author inspected pages 11–20. The parent reports that the final remote rebuild's 30 original-study page images match those reviewed images exactly.

The reviewer also individually inspected all final quantity-paper pages 7–14 at `/tmp/dmfl-papers-final-20260919/quantity-skew/rendered/main-07.png` through `main-14.png`. The PDF identified by the parent has SHA256 `bc28e87bd1d2db3fb7feb34ce420c12d21fb1fd89912f305a9e8302061eba419` and a warning-free remote build. Every assigned page passes. Both tables, all four figures, endpoint/factorial equations, negative conclusion, provenance paths, source fingerprint, and all bibliography entries are readable and within page bounds. Figure error bars and labels distinguish best from final validation accuracy and show the feasible rank-4 comparison. There is no clipping, overlap, missing plot label, or accidental blank page. The results-paragraph pronoun clarification appears correctly in the final render. The parent reviewer covers quantity pages 1–6.

No local rendering or scientific execution was performed for this review. The scientific source review and these assigned visual reviews are complete with no outstanding findings.
