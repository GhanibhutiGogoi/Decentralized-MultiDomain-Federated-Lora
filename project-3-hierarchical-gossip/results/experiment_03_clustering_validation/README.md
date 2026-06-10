# Experiment 03: Clustering Validation

## Problem

Hierarchical Gossip Aggregation relies on grouping similar clients into clusters. Before using clustering for decentralized federated learning, we must verify whether the clustering algorithm can successfully recover the underlying domain structure of the data.

The goal of this experiment is to evaluate the quality of client clustering and measure how well predicted clusters align with true domain labels.

## Approach

We extracted client representations and applied Agglomerative Clustering with:

* Number of Clusters: 5
* Linkage: Ward
* Random Seed: 42

Clustering quality was evaluated at multiple training stages:

* Stage 2
* Stage 5
* Stage 10
* Stage 20

The following metrics were used:

* Adjusted Rand Index (ARI)
* Normalized Mutual Information (NMI)
* Silhouette Score

Additionally, t-SNE visualizations and clustering confusion matrices were generated to compare predicted clusters against true domain assignments.

## Results

Generated files:

* clustering_metrics.json
* clustering_quality.png
* clustering_confusion_matrix.png
* tsne_visualization.png

Final clustering metrics (Stage 20):

| Metric           | Value   |
| ---------------- | ------- |
| ARI              | -0.1395 |
| NMI              | 0.2903  |
| Silhouette Score | 0.0704  |

Metric evolution:

| Stage | ARI     | NMI    | Silhouette |
| ----- | ------- | ------ | ---------- |
| 2     | 0.0308  | 0.4511 | 0.2218     |
| 5     | -0.0410 | 0.3922 | 0.1588     |
| 10    | -0.0426 | 0.3641 | 0.0804     |
| 20    | -0.1395 | 0.2903 | 0.0704     |

The clustering confusion matrix shows substantial overlap between domains, with clients from different domains frequently assigned to the same cluster.

The t-SNE visualization further demonstrates that predicted clusters do not align well with the true domain structure.

## Findings

The clustering method failed to recover the true domain structure of the clients.

All three clustering metrics deteriorated as training progressed. ARI became negative, indicating that the final clustering assignments were worse than random agreement with the ground-truth domains. NMI also decreased steadily from 0.4511 to 0.2903, while the silhouette score dropped from 0.2218 to 0.0704.

These results suggest that the current feature representation does not provide sufficient separation between domains for reliable clustering. Consequently, clustering-based client grouping requires either improved representations, alternative similarity metrics, or different clustering strategies before it can be effectively integrated into Hierarchical Gossip Aggregation.
