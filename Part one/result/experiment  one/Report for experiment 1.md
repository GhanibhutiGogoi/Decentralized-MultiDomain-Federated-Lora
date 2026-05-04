\# Adaptive LoRA Rank for Heterogeneous Federated Learning:  

\## A Study on Capacity–Computation Mismatch



\---



\## Abstract



Federated Learning (FL) enables collaborative model training across distributed clients with heterogeneous computational resources. Low-Rank Adaptation (LoRA) has recently emerged as an efficient fine-tuning method that significantly reduces the number of trainable parameters. However, existing approaches typically assume a \*\*uniform LoRA rank across all clients\*\*, disregarding variations in computational capability.  



This paper investigates the limitations of fixed LoRA rank in heterogeneous federated environments. Through extensive experiments across multiple modalities (image, text, tabular, and audio), we demonstrate that uniform rank allocation leads to \*\*capacity–computation mismatch\*\*, resulting in inefficient resource utilization and aggregation bias. Specifically, low-capability clients under-utilize model capacity, while high-capability clients suffer from representational bottlenecks despite increased computation.  



We analyze these effects both mathematically and empirically, showing that client influence is primarily dictated by computational effort rather than parameterization. Based on these findings, we argue that \*\*adaptive LoRA rank allocation is necessary\*\* to align model capacity with client resources, improve fairness, and enhance global model performance.  



\---



\## 1. Introduction



Federated Learning (FL) enables distributed clients to collaboratively train models without sharing raw data. However, FL systems are inherently \*\*heterogeneous\*\*, with clients differing in computational power, data availability, and communication constraints.



Low-Rank Adaptation (LoRA) reduces training complexity by decomposing weight updates into low-rank matrices, significantly lowering parameter count and communication overhead. Despite this advantage, most implementations assume a \*\*fixed LoRA rank across clients\*\*, which is unrealistic in heterogeneous environments.



This paper investigates:



> How does a fixed LoRA rank affect federated learning performance under heterogeneous client computational capabilities?



\---



\## 2. Methodology



\### 2.1 Federated Learning



The global model is updated using Federated Averaging (FedAvg):



\\\[

w^{(t+1)} = \\sum\_{k=1}^{K} \\frac{n\_k}{\\sum\_{j=1}^{K} n\_j} \\, w\_k^{(t)}

\\]



where:

\- \\( w\_k^{(t)} \\): local model  

\- \\( n\_k \\): number of processed samples  



\---



\### 2.2 LoRA Parameterization



\\\[

W' = W + BA

\\]



\- \\( A \\in \\mathbb{R}^{r \\times d} \\)  

\- \\( B \\in \\mathbb{R}^{d \\times r} \\)  

\- \\( r \\ll d \\)  



Parameter reduction:

\\\[

d^2 \\rightarrow 2rd

\\]



\---



\### 2.3 Heterogeneous Clients



Clients are assigned different computational budgets:



\- Client 1: \\( E\_1 = 1 \\)  

\- Client 2: \\( E\_2 = 2 \\)  

\- Client 3: \\( E\_3 = 3 \\)  



All clients share:

\- identical LoRA rank  

\- identical architecture  



\---



\### 2.4 Experimental Setup



Modalities:

\- Image (CNN, MLP)  

\- Text (LSTM, Transformer)  

\- Tabular (MLP)  

\- Audio (1D CNN)  



Metrics:

\- Accuracy  

\- Contribution %  

\- Aggregation weights  

\- Cumulative samples  



\---



\## 3. Results and Analysis



\### 3.1 Contribution Imbalance



\\\[

\\text{Contribution}\_k \\propto E\_k

\\]



High-compute clients dominate aggregation.



\---



\### 3.2 Capacity–Computation Mismatch



\#### Low-capability clients

\- under-trained parameters  

\- low contribution  



\#### High-capability clients

\- limited by low rank  

\- diminishing returns  



\---



\### 3.3 Accuracy Trends



\- steady improvement  

\- dominated by high-capability clients  



\---



\### 3.4 Aggregation Bias



\\\[

\\text{Influence}\_k \\propto n\_k \\propto E\_k

\\]



Leads to unfair contribution distribution.



\---



\## 3.5 Computational Inefficiency (NEW CONTRIBUTION)



\### Definition



\\\[

\\text{Efficiency}\_k = \\frac{\\Delta \\mathcal{L}\_k}{E\_k}

\\]



\\\[

\\text{Computational Loss}\_k = E\_k - \\alpha \\cdot \\Delta \\mathcal{L}\_k

\\]



\---



\### Case 1: High Compute + Low Rank



\\\[

\\lim\_{E\_k \\to \\infty} \\Delta \\mathcal{L}\_k \\rightarrow \\text{constant}

\\]



\- performance saturates  

\- extra computation wasted  



\---



\### Case 2: Low Compute + High Rank



\\\[

\\Delta \\mathcal{L}\_k \\propto \\frac{E\_k}{r}

\\]



\- insufficient training  

\- unused parameters  



\---



\### System-Level Effect



| Client Type | Problem | Waste |

|------------|--------|------|

| High compute + low rank | Bottleneck | Computation |

| Low compute + high rank | Under-training | Parameters |



\---



\## 4. Discussion



Findings show:



\- FL is \*\*resource-driven\*\*, not parameter-driven  

\- uniform rank causes imbalance  

\- mismatch leads to inefficiency  



Key issue:



> Misalignment between model capacity and computational capability



\---



\## 5. Conclusion



This paper demonstrates:



\- fixed LoRA rank creates \*\*capacity–computation mismatch\*\*  

\- high-resource clients waste computation  

\- low-resource clients waste parameters  

\- aggregation becomes biased  



\### Final Statement



\*\*Adaptive LoRA rank is necessary for efficient, fair, and scalable federated learning.\*\*



\---



\## 6. Future Work



\### 6.1 Adaptive Rank



\\\[

r\_k \\propto \\text{compute capacity}

\\]



\---



\### 6.2 Joint Optimization



\\\[

\\alpha\_k = f(n\_k, r\_k)

\\]



\---



\### 6.3 Dynamic Rank Scheduling



\- adjust rank during training  



\---



\### 6.4 Real-World Constraints



\- bandwidth  

\- energy  

\- device availability  



\---



\### 6.5 Non-IID Data



\- evaluate robustness  

\- improve fairness  



\---



\## Keywords



Federated Learning, LoRA, Adaptive Models, Distributed Systems, Heterogeneous Computing

