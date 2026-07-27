# Clean analysis package: ledger

## What is included
- Core numeric trend outputs (year-level)
- Header/account evolution outputs
- Embedding clustering comparison by unit:
  - year-level document clustering
  - page-level document clustering

## Embedding setup for comparison
- Embedding: TF-IDF + SVD
- Clustering: KMeans with silhouette-based k selection

## Quick compare
unit  n_units  k  silhouette embedding
year      196  2    0.006574 tfidf_svd
page     1580 12   -0.028779 tfidf_svd

## Files to look at first
- embeddings/unit_comparison.csv
- embeddings/year_cluster_timeline.png
- embeddings/page_cluster_timeline.png
- yearly_amount_and_volume.png
- change_points_amount_sum.png
