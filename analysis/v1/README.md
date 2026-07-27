# Clean analysis package: ledger

This is the **single source of truth** for this ledger analysis.
Old timestamped folder names were removed.

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

## Root file guide
- `yearly_numeric_summary.csv`: Year-by-year numeric summary (counts, totals, median, p90, max).
- `yearly_change_points.csv`: Change-point candidates from year-over-year deltas.
- `yearly_amount_and_volume.png`: Time series of yearly amount and row volume.
- `yearly_median_p90.png`: Distribution-level yearly trend (median and p90).
- `change_points_amount_sum.png`: Yearly amount with detected structural jump points.
- `header_account_first_appearance.csv`: First/last year each header/account text appears.
- `header_account_yearly_counts.csv`: Number of header rows and unique header texts per year.
- `REANALYSIS_SUMMARY.md`: Narrative summary from the year-level analysis stage.

## Embeddings folder guide
See `embeddings/README.md` for per-file details.
