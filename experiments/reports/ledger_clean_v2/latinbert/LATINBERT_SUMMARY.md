# LatinBERT Embedding Analysis

## Model
- **bowphs/PhilBerta**
- Device: mps

## Silhouette scores (year-level clustering)
| Method | Model | Best k | Silhouette |
|--------|-------|--------|------------|
| LatinBERT | bowphs/PhilBerta | 2 | 0.4555 |
| TF-IDF + SVD (v2) | TfidfVectorizer + TruncatedSVD | 4 | 0.2642 |

## Interpretation notes
- The corpus spans Latin-dominant (pre-1760), transitional (1760-1840), and
  English-dominant (post-1840) text. A Latin-specific model may capture semantic
  structure in early entries better than bag-of-words TF-IDF.
- Check **latinbert_vs_tfidf_timeline.png** to see whether the two methods
  agree on where historical transitions occur.
- Check **latinbert_pca.png** (right panel, coloured by era) to see whether
  the Latin model separates eras more cleanly than TF-IDF.
- A higher silhouette for LatinBERT on pre-1760 years would support the claim
  that it captures meaningful semantic variation in the early Latin entries.

## Output files
- year_latinbert_embeddings.csv  — raw embedding vectors (n_years × hidden_dim)
- year_latinbert_clusters.csv    — cluster assignments + era + amount metadata
- era_cluster_profile_latinbert.csv
- era_cluster_profile_tfidf.csv
- embedding_comparison.csv       — silhouette score comparison table

## Figures
- latinbert_pca.png              — PCA of LatinBERT embeddings (cluster + era)
- tfidf_pca.png                  — PCA of TF-IDF embeddings (for comparison)
- latinbert_cluster_timeline.png — cluster assignments across 1700-1900
- latinbert_vs_tfidf_timeline.png — side-by-side timeline comparison
