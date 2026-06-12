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

## Findings

This experiment should be interpreted as a pipeline-validation run rather than a final research conclusion.

The purpose of this run was to verify that client feature extraction, clustering, metric computation, and visualization generation work end-to-end.

Because these results are based on short smoke-test runs, the clustering metrics should be treated as diagnostic outputs only.

The current results are useful for checking the implementation and identifying potential issues in the clustering pipeline, but they should not be used as final evidence about clustering quality.

Full-length experiments with stable representations and more complete training are required before drawing conclusions about the effectiveness of the clustering strategy.
