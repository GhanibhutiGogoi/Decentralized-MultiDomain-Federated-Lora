# Literature Reading List

Priority: **[Must Read]** > **[Important]** > **[Useful]**

---

## 1. LoRA and Parameter-Efficient Fine-Tuning

- **[Must Read]** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
  - The foundational paper. Understand the math: W' = W + BA, rank selection, scaling factor alpha/r.
  - Notes: _____

- **[Important]** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023)
  - 4-bit quantization + LoRA. Relevant for reducing communication cost further.
  - Notes: _____

## 2. Federated Learning Foundations

- **[Must Read]** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
  - The FedAvg paper. Our centralized baseline.
  - Notes: _____

- **[Important]** Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
  - FedProx: handles heterogeneity with proximal term. Compare aggregation strategies.
  - Notes: _____

- **[Important]** Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (ICML 2020)
  - Variance reduction for FL. Relevant for convergence analysis.
  - Notes: _____

## 3. Heterogeneous LoRA in Federated Learning

- **[Must Read]** Cho et al., "Heterogeneous LoRA for Federated Fine-tuning" (2024)
  - HetLoRA: different ranks per client. Key baseline. Understand their aggregation via zero-padding.
  - Notes: _____

- **[Must Read]** Sun et al., "FedALoRA: Adaptive Low-Rank Adaptation for Federated Learning" (2024)
  - Personalized aggregation weights for LoRA. Another key baseline.
  - Notes: _____

- **[Important]** Zhang et al., "FFA-LoRA: Federated Feature Alignment LoRA" (2024)
  - Feature-aligned aggregation. Related to our domain-aware approach.
  - Notes: _____

## 4. Decentralized / Gossip-Based Federated Learning

- **[Must Read]** Lian et al., "Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for D-PSGD" (NeurIPS 2017)
  - D-PSGD: the foundational decentralized FL paper. Understand mixing matrices.
  - Notes: _____

- **[Must Read]** Koloskova et al., "Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication" (ICML 2019)
  - Gossip SGD with compression. Key for convergence proofs.
  - Notes: _____

- **[Important]** Assran et al., "Stochastic Gradient Push for Distributed Deep Learning" (ICML 2019)
  - SGP: push-based gossip for directed graphs. Alternative gossip protocol.
  - Notes: _____

- **[Important]** Tang et al., "Communication Compression for Decentralized Training" (NeurIPS 2018)
  - DeepSqueeze: gradient compression in decentralized settings.
  - Notes: _____

## 5. Clustered / Personalized Federated Learning

- **[Must Read]** Ghosh et al., "An Efficient Framework for Clustered Federated Learning" (NeurIPS 2020)
  - IFCA: iterative federated clustering. Our clustering is related but works without central server.
  - Notes: _____

- **[Important]** Marfoq et al., "Federated Multi-Task Learning under a Mixture of Distributions" (NeurIPS 2021)
  - FeSEM: EM-based clustering for FL. Compare clustering approaches.
  - Notes: _____

- **[Useful]** Li et al., "Ditto: Fair and Robust Federated Learning Through Personalization" (ICML 2021)
  - Fairness in personalized FL. Relevant for our fairness metrics.
  - Notes: _____

## 6. Convergence Theory for Decentralized FL

- **[Important]** Koloskova et al., "A Unified Theory of Decentralized SGD with Changing Topology and Local Updates" (ICML 2020)
  - Unified convergence framework. Essential for our theoretical analysis.
  - Notes: _____

- **[Useful]** Neglia et al., "Decentralized Gradient Methods: Does Topology Matter?" (AISTATS 2020)
  - Impact of topology on convergence. Directly relevant to our topology design.
  - Notes: _____

---

## Reading Schedule

| Week | Papers | Focus |
|------|--------|-------|
| 1 | LoRA, FedAvg, D-PSGD, HetLoRA | Foundations |
| 2 | IFCA, Gossip SGD, FedALoRA, Koloskova unified | Core techniques |
| 3 | Remaining important papers | Depth |
| 4 | Useful papers as needed | Breadth |
