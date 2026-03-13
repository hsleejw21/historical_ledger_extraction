# Embeddings output guide

This folder compares text-embedding clustering by analysis unit.

## Comparison summary
- `unit_comparison.csv`: Side-by-side metrics for year-level vs page-level clustering (`n_units`, chosen `k`, silhouette).

## Year-level outputs
- `year_docs.csv`: One aggregated document per year (input to embedding).
- `year_embedding_meta.csv`: Method metadata (`tfidf_svd`, selected `k`, silhouette).
- `year_embedding_clusters.csv`: Cluster label for each year.
- `year_cluster_profile.csv`: Cluster size and year range summary.
- `year_embedding_pca.png`: PCA 2D projection of year embeddings.
- `year_cluster_timeline.png`: Cluster labels across time (x-axis = year).

## Page-level outputs
- `page_docs.csv`: One aggregated document per sheet/page (input to embedding).
- `page_embedding_meta.csv`: Method metadata (`tfidf_svd`, selected `k`, silhouette).
- `page_embedding_clusters.csv`: Cluster label for each page/sheet.
- `page_cluster_profile.csv`: Cluster size and year-range summary.
- `page_embedding_pca.png`: PCA 2D projection of page embeddings.
- `page_cluster_timeline.png`: Cluster labels across time (x-axis = page reference year).

## Interpretation note
- If silhouette is low or negative, clusters are weakly separated in that feature space.
- Current run: year-level separation is better than page-level under TF-IDF+SVD.
