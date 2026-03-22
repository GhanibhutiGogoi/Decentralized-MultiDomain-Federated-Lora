# Hierarchical Gossip Aggregation for Decentralized Federated Learning with Domain Clustering
                                                               

Samir Kassymov

March 22, 2026
____________________________________________________________________________________________

## ABSTRACT

Decentralized federated learning has emerged as a promising paradigm for collaborative model training without relying on centralized data collection. However, existing approaches often struggle with two critical challenges: inefficient peer-to-peer communication and the presence of domain heterogeneity across clients. Traditional gossip-based methods typically treat all clients equally, ignoring differences in data distributions, which leads to suboptimal model performance and slow convergence.

This project proposes a novel hierarchical gossip aggregation framework that explicitly accounts for domain heterogeneity in decentralized environments. Clients are first clustered based on domain similarity using representations derived from LoRA matrices. Within each cluster, gossip aggregation is performed to efficiently exchange information among similar clients. A cross-domain transfer mechanism is then introduced, where transfer weights are learned to control knowledge sharing between different clusters.

The proposed approach preserves domain-specific knowledge while enabling beneficial cross-domain learning. It is expected to improve communication efficiency, accelerate convergence, and enhance overall model performance in decentralized federated learning systems.

## 1 OBJECTIVES

The primary objective of this project is to design a scalable and efficient decentralized federated learning framework that can effectively handle domain heterogeneity. Specifically, the project aims to:

- Develop a hierarchical aggregation strategy for decentralized environments.
- Improve communication efficiency in gossip-based learning systems
- Preserve domain-specific knowledge during model aggregation
- Enable controlled and beneficial cross-domain knowledge transfer
- Ensure convergence stability under decentralized settings

Additionally, this project provides a foundation for future research in adaptive clustering, personalized federated learning, and large-scale distributed systems.

## 2 Problem Context

Decentralized federated learning eliminates the need for a central server by allowing clients to communicate directly through peer-to-peer mechanisms such as gossip protocols. While this approach improves system robustness and scalability, it introduces significant challenges in coordination and convergence.

A key limitation of existing methods is their assumption that all clients are homogeneous. In practice, however, clients often operate on data from different domains, leading to domain heterogeneity. Standard gossip aggregation methods perform uniform averaging across clients, ignoring these differences. As a result, domain-specific knowledge may be diluted, and irrelevant information may negatively impact model performance.

This highlights the need for a domain-aware aggregation mechanism that can both preserve local specialization and enable meaningful collaboration across clients.

## 3 Dataset

This project uses the CIFAR-10 dataset, a standard benchmark dataset for image classification

- Total imgaes: 60.000
- Traininп set: 50.000
- Test set: 10.000
- Image size: 32 * 32 RGB
- Number of classes: 10

The data set includes categories such as airplanes, automobiles, birds, cats, and trucks.

## 4  Methodology

## 4.1  System Design

The proposed framework introduces a domain-aware hierarchical gossip aggregation mechanism for decentralized federated learning. The system consists of three key components:

1. **Client Clustering by Domain Similarity**
Clients are grouped into clusters based on the similarity of their learned representations, derived from LoRA matrices. This ensures that clients with similar data distributions are aggregated together.
2. **Within-Domain Gossip Aggregation**
Gossip-based communication is performed within each cluster, allowing clients to exchange information efficiently with similar peers. This helps preserve domain-specific knowledge and reduces noise from unrelated domains.
3. **Cross-DomainTransferLearning**
To enable knowledge sharing across clusters, the system learns transfer weights that determine how much influence one domain should have on another. These weights are computed based on performance evaluations across domains and normalized to ensure stable learning.

This hierarchical structure enables both local specialization and global collaboration.

## 4.2  OperationalProcess

The overall workflow of the system can be described as follows:
1. Initialize clients with local datasets and model parameters
2. Extract domain-specific features from LoRA representations
3. Cluster clients based on domain similarity
4. Perform gossip aggregation within each cluster
5. Compute cross-domain transfer weights between clusters
6. Apply cross-domain knowledge transfer
7. Update local models and repeat the process over multiple rounds

This iterative process allows the system to progressively improve model performance while maintaining efficient communication.

## 4.3 Expected Outcomes

The proposed framework is expected to achieve the following outcomes:

- Reduced communication overhead compared to traditional gossip methods
- Improved model accuracy through domain-aware aggregation
- Faster convergence due to structured information exchange
- Enhanced scalability for large decentralized networks
- Better preservation of domain-specific knowledge
  
Overall, the framework aims to provide a more efficient and robust solution for decentralized federated learning under realistic, heterogeneous conditions.

## Conclusion

This project presents a structured and domain-aware approach to decentralized federated learning through hierarchical gossip aggregation. By integrating domain clustering and adaptive cross-domain transfer, the proposed method addresses key limitations of existing approaches. It balances local specialization with global knowledge sharing, resulting in improved performance and scalability. The framework opens new directions for future research in intelligent aggregation strategies and decentralized AI systems.
