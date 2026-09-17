# Decentralized LoRA baseline shortlist and reproduction audit

Research date: 2026-09-17. Read-only primary-source and public-repository review; no training or numerical software tests were run for this note. All subsequent experiments belong on gpu003.

## Recommendation

Use **Dec-LoRA, “Decentralized Low-Rank Fine-Tuning of Large Language Models” (REALM 2025)** as the principal published comparator, initially on **RoBERTa-base + SST-2**, and add **DeCAF-style effective-update aggregation with truncated SVD** as an essential methodological control. Dec-LoRA is an actual peer-to-peer method, uses a manageable transformer on V100-class hardware, and evaluates the same task at all clients rather than assigning different topical tasks. The choice is based on fit and reproducibility, not on choosing a weak reported result.

**No official author implementation was verified for these shortlisted papers.** The sources support transparent independent reimplementations, not a claim to have executed the authors' exact code. Several public repositories mention these algorithms, but a matching name or method description does not establish author ownership. This is a material limitation and must accompany the comparison.

There must be two separately named stages:

1. Published-setting reconstruction: retain the paper's model, task, topology, rank, and metric, and document every unspecified implementation choice.
2. Quantity-skew comparison: run the reconstructed baseline and the proposed method on **exactly the same new unequal-size client partition**. This is a new controlled benchmark on a paper's dataset, not an exact reproduction of the paper's partition.

Sample-size weighting `p_i = n_i / sum_j n_j` is conventional empirical-risk/FedAvg weighting. It should be an explicit control, not presented as a new contribution by itself. Rank adaptation, decentralized preservation of the intended sample-weighted objective, and measured resource/accuracy tradeoffs are the potentially research-relevant parts.

## 1. Dec-LoRA — preferred published comparator

- Title: **Decentralized Low-Rank Fine-Tuning of Large Language Models**.
- Authors: Sajjad Ghiasvand, Mahnoosh Alizadeh, Ramtin Pedarsani.
- Publication: Proceedings of the 1st Workshop for Research on Agent Language Models (REALM 2025), pages 334–345, July 2025.
- Publisher record: <https://aclanthology.org/2025.realm-1.24/>
- Published PDF: <https://aclanthology.org/2025.realm-1.24.pdf>
- DOI: <https://doi.org/10.18653/v1/2025.realm-1.24>
- Later arXiv version: <https://arxiv.org/html/2501.15361v5>
- Official public code: **not verified**. The ACL PDF's GitHub citation is to Hugging Face PEFT, not the authors' experiment repository. The arXiv primary page does not establish an author code repository either.
- License: ACL's published paper is CC BY 4.0. No author-code license can be claimed without an author repository. Hugging Face PEFT is a separate dependency, not Dec-LoRA's official implementation.

### Actual decentralized method

All clients start from the same pretrained model and common adapter initialization. Each client performs `K` local updates, sends its A and B factors to graph neighbors, and independently averages each factor using a fixed symmetric doubly stochastic mixing matrix. No parameter server is part of Algorithm 1. Ring and static Erdős–Rényi (ER) graphs are used. For ER, the stated matrix is

`Q = I - 2 L / (3 lambda_max(L))`,

where `L` is the graph Laplacian. The paper describes ring neighbors but does not give numerical ring coefficients in the published text. Self/left/right weights of 1/3 are a standard, explicitly disclosed reconstruction choice.

Evaluation of an averaged client model is an **evaluation endpoint**, separate from training's neighbor communication. Our implementation must state whether final factor averages are assembled by neighbor-only reduction or collected by an evaluator; extra assembly communication must be counted when claiming decentralized deployment.

### Published experimental facts

The ACL PDF, inspected directly, states:

- Backbone: **RoBERTa-base**, rather than a frozen image feature extractor or an adapter solely on a classifier.
- Tasks: MRPC, SST-2, QNLI, QQP, MNLI from GLUE.
- Learning rate: **1e-3**; batch size: **32** across BERT-family tasks/methods.
- Published Table 1: **10 clients**, ring and ER, Dec-LoRA rank **16**; `K=1, T=20` and `K=5, T=10`.
- Published Table 2: **10 or 20 clients**, ring, ranks **2/4/8**, comparison against conventional pooled-data LoRA, **100 communication rounds**.
- Metric: **best validation accuracy across rounds**, evaluated using averaged client models. MRPC's designated metric is **F1**.
- Published Table 1 SST-2 Dec-LoRA ring values: **93.81%** for `K=1`, **94.61%** for `K=5`. These are published reference values, not our measurements and not guaranteed reproducible targets.
- Three-client label-skew experiment: binary task client class proportions `[.15,.85]`, `[.85,.15]`, `[.5,.5]`; separate from the main 10-client setup and separate from our quantity-only proposal.
- Hardware includes NVIDIA A6000 and **V100**, so the model family is a sensible fit to gpu003. Runtime and memory have not been measured in this literature review.

### Material ambiguities and version differences

- **Optimizer not specified.** The ACL PDF contains no optimizer/Adam declaration. Algorithm 1 is written as gradient descent. “Default hyperparameters” for PEFT baselines does not determine an optimizer.
- **Unit of a local update unclear in the experiment description.** Algorithm 1 expresses stochastic-gradient updates; the text also uses “communication rounds/epochs” for larger-model experiments. Do not silently call one whole local epoch one minibatch update. Record the exact chosen semantics, number of minibatches, and example exposure. A short sensitivity comparing the two interpretations may be needed before asserting numerical reproduction.
- **LoRA alpha, dropout, target modules, sequence length, classifier handling, optimizer-state handling, data-split seed, training seeds, and replicate count are not specified in the inspected published setup.** Rank-16 Q/V adapters across 12 RoBERTa-base layers contain 589,824 parameters, consistent with the rounded `0.60M` count, but this does not prove the target configuration or classifier handling.
- **SST-2 split discrepancy:** Table 3 lists train **66,675**, development **674**, while canonical GLUE SST-2 has 67,349 training and 872 development examples. A 1% training holdout would explain the first pair, but the paper does not say this. Some reported accuracy values are consistent with 872-example development evaluation; without code/seeds that is only a clue. No exact split reconstruction claim is justified.
- The later arXiv v5 expands the theoretical/empirical material and changes presentation and some numeric results. **Pin the ACL publication** for the principal published comparator; do not combine v5 table values with ACL settings as if they were one release.
- “Best validation” is selection on a public development set, not held-out test accuracy. Report final-round accuracy too, and use a separate training-only tuning split for our methodological decisions.

### Concrete feasible reconstruction to preregister

Start with RoBERTa-base, SST-2, 10-client fixed ring, batch 32, LR 1e-3, rank16 baseline. Match `T=20,K=1` as the first published-setting check **after explicitly specifying whether K is minibatches or epochs**; retain the 100-round rank comparison as a separate protocol if used. Use the full canonical dataset and standard development set unless author provenance resolves the discrepancy, and mark this as a disclosed difference. Pick and log target modules, head training, alpha, sequence length, optimizer/moments, and seeds before comparing methods. A practical explicit reconstruction can use Q/V LoRA and AdamW, but those must be labeled our choices rather than attributed to the paper. Run at least three paired seeds; five is preferable if runtime allows.

For quantity skew, keep the model/task/evaluation fixed and stratify the labels while varying client counts, so size is the deliberate source of statistical heterogeneity. Compare equal-size and quantity-skew conditions on all methods. Preserve an unconstrained rank16 reference, but also include a fixed-rank baseline that fits every client's stated resource cap. Do not describe the unconstrained reference as resource matched.

### Smaller first study: MRPC

MRPC is a legitimate paper task and offers a substantially smaller first complete multi-seed experiment. The ACL Table 3 explicitly lists **3,301 training / 367 development** examples and **F1**. These counts sum to the canonical 3,668-example MRPC training split, but do not identify a split seed; canonical GLUE MRPC development has 408 examples. As with SST-2, an internal holdout is plausible but not established. Table 4's **10-client ring, full-precision versus 4-bit** experiment reports full-precision MRPC F1 values **89.31, 89.20, 89.16** for ranks **2,4,8**, respectively. MRPC is absent from the paper's main Table 1 and Table 2 multi-method comparisons. Use it as a first **paper-task matched-method benchmark**, with honest split/optimizer/steps assumptions and both F1 and accuracy; do not claim exact reproduction of the main published comparison. Retain SST-2 as a larger confirmatory task before generalizing a positive result.

## 2. DeCAF — essential aggregation control and related work

- Title: **DeCAF: Decentralized Consensus-And-Factorization for Low-Rank Adaptation of Foundation Models**.
- Authors: Nastaran Saadati, Zhanhong Jiang, Joshua R. Waite, Shreyan Ganguly, Aditya Balu, Chinmay Hegde, Soumik Sarkar.
- Primary source: <https://arxiv.org/abs/2505.21382>
- Full text inspected: <https://arxiv.org/html/2505.21382>
- Status established here: May 2025 arXiv preprint; no venue claim is inferred.
- Official code/license: **not verified**. No implementation link in the inspected HTML. First author public repositories at <https://github.com/nsaadati> did not contain a DeCAF repository at inspection. Absence from this search is not proof that code does not exist elsewhere.

DeCAF performs neighborhood consensus on effective LoRA products and uses truncated SVD to obtain new low-rank factors. This substantially overlaps the existing project's product-space merge design. Comparing only to independent A/B factor averaging would omit a particularly relevant prior algorithm.

Published setup:

- CLIP with ViT-B/16 on Flowers, UCF, Food101; rank2, stated scaling factor eta1, dropout.25, batch32, cosine schedule; **500 epochs**, communication every epoch.
- Ten clients and **40 shots per class total** in IID experiments, giving four examples/class/client. Their centralized reference uses 16 shots from prior work, so it is not a fair same-data comparator to reuse unchanged.
- Ring, complete, and bipartite topology experiments. Complete mixing uses 1/N; ring uses 1/3 on self and the two neighbors.
- LLaMA-2-7B on WiC and BoolQ: rank4, stated scaling eta8, dropout.1, batch32, **150 batches**, communication every three batches; reported F1.
- Seeds, exact split files, and a complete optimizer/learning-rate specification were not established by this review.

Feasibility: CLIP few-shot training is plausible on 32GB V100s but introduces a new dataset stack; 500 epochs and full-layer SVD deserve profiling. LLaMA2 carries model-access/license and memory overhead. Best immediate role: implement the **DeCAF algorithm as a matched control on the selected Dec-LoRA task**, explicitly distinguishing this from a reproduction of DeCAF's own published benchmark numbers.

## 3. ADF-LoRA / TAD-LoRA — stronger newer alternating baseline, code unavailable

- **ADF-LoRA: Alternating Low-Rank Aggregation for Decentralized Federated Fine-Tuning**, Xiaoyu Wang, Xiaotian Li, Zhixiang Zhou, Chen Li, Yong Liu, November 2025 preprint: <https://arxiv.org/html/2511.18291v1>.
- Follow-up by the same authors: **Stabilizing Decentralized Federated Fine-Tuning via Topology-Aware Alternating LoRA** (TAD-LoRA), February 2026 preprint: <https://arxiv.org/html/2602.00451v1>.
- Official code/license: **not verified**. Neither inspected primary HTML links author code; GitHub exact-name/title searches did not establish an author implementation. The arXiv hosting license is not a code license.

These methods train one LoRA factor at a time according to a common phase schedule, while mixing **both** factors to limit mismatch of frozen blocks. TAD-LoRA chooses phase duration with topology in mind.

Common experimental setup:

- RoBERTa-large335M, LoRA on Q/V, r8, alpha16, dropout.1; **classification head frozen**.
- GLUE SST-2, QNLI, QQP, MNLI; ten clients.
- 150 rounds × 20 local steps, batch32, sequence length128, AdamW with stated Hugging Face defaults.
- Learning-rate search ADF: `{5e-4,1e-3,2e-3,5e-3}`; TAD adds `2e-4`.
- Binary-task label skew: three clients90/10, three10/90, four50/50. MNLI uses class-skewed triples. This is **not quantity-only IID**.
- ADF communication: pair encounters with probability.1 per round; symmetric averaging. TAD: independently activated ER edges with p in `{.5,.2,.1,.05,.02,.01}`; ring results also reported.
- Evaluation: evaluate each of ten client models, average client accuracies, then summarize seeds. This is **not accuracy of one assembled adapter**.
- TAD appendix says three random seeds but IDs are unspecified, and inconsistently calls its reported dispersion “variance” while other text says standard deviation. Do not silently reinterpret.

Feasibility: RoBERTa-large is plausible sequentially on V10032GB but costlier than RoBERTa-base. Missing code, evaluation mismatch, and the label-skew protocol make it a later independent reconstruction rather than the cleanest first baseline.

## 4. RW-LoRA — relevant communication comparator, substantially different protocol

- Title: **RW-LoRA: Communication-Efficient Decentralized LoRA Fine-Tuning via Random Walks**.
- Authors: Xingran Chen, Rohit Bhagat, Ghadir Ayache, Rawad Bitar, Yanmin Gong, Salim El Rouayheb.
- Primary source: <https://arxiv.org/html/2609.00078v1>.
- The paper states acceptance at **IEEE ITW 2026**; this review did not independently verify conference proceedings.
- Official code/license: **not verified**, with no author-code link in inspected HTML.

One adapter/model token travels through the graph. Only the current client updates it and forwards it to a neighbor. This avoids simultaneous model averaging entirely, so it tests a different communication/parallelism tradeoff.

- RoBERTa-base125M, default rank16, ablation ranks4/8/16/32.
- MRPC, SST-2, QNLI, MNLI, QQP. MRPC uses F1; others accuracy.
- Thirty clients; ring and complete graphs; Metropolis–Hastings walk with uniform target stationary distribution.
- AdamW LR1e-3, ten local steps per token visit, batch32.
- Five runs hold initialization/data seed42 fixed and vary only random-walk seed1–5. This is not five fully independent training/data seeds.
- Approximately2500 token steps versus180 gossip rounds; this means approximately25,000 versus54,000 total local updates, not matched work. Both final and best observed scores are reported.
- QNLI communication curve explicitly uses IID data; exact quantity allocation and split files are not provided in the inspected text.

Good potential later control for accuracy per edge activation or bytes, but synchronized-round comparison is inappropriate. It also does not directly implement heterogeneous client ranks.

## Public repository encountered but not verified as paper code

<https://github.com/LuYuPromax/DFL-LORA>, inspected commit `df9da4eaccf9ab4a67c7d68fda261dd497d36e92`, has an **Apache-2.0** license and real decentralized-factor code. Its README proposes freezing A to reduce consensus error. However:

- No paper title, citation, or author link establishes it as an official implementation of any shortlisted paper.
- BERT launch defaults are actually DistilBERT, seven clients, r4, alpha32, and training type `ConLoRA`; README terminology differs.
- Topology generator hardcodes seven-client graphs while the data-splitting example defaults to four clients.
- Local absolute model/data paths must be replaced; target list contains `classifer` spelling.
- `finetune/llama/client.py` is empty at that commit.
- Requirements pin Torch2.0.1, Transformers4.42.3, PEFT0.11.1, datasets2.20.0, among many packages.

It may be useful as an attributed implementation reference after inspection, but cannot be labeled the official Dec-LoRA baseline or run uncritically as a publication reproduction.

## Fair-comparison requirements

1. Preserve dataset IDs, split-index files, tokenizer/model revision, preprocessing, maximum length, adapter sites, scaling, classifier policy, optimizer, moments policy, schedules, and seeds.
2. Preserve identical graph and communication opportunities across matched methods. Log modeled and actual bytes separately, including metadata and final assembly.
3. Count examples and optimizer steps rather than equating a local epoch across unequal datasets with equal work.
4. Evaluate the paper's endpoint and the user's assembled-global-adapter endpoint separately when they differ.
5. Include fixed-rank equal weights, fixed-rank sample weights, adaptive-rank equal weights, adaptive-rank sample weights, and effective-product/TSVD controls where budgets permit.
6. Do not make resource savings claims from rank alone: measure adapter/optimizer state, peak memory, forward/backward work, merge cost, wall time, and transmitted bytes.
7. Freeze a tuning protocol before the final paired-seed experiment; tune every comparator fairly. A result on the new quantity-skew split should be stated as a result under that tested condition, not as generally beating the paper.
8. Keeping raw data on clients is a data-locality property. These comparisons alone do not establish formal privacy or protection against update leakage.
