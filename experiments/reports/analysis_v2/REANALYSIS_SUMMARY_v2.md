# Reanalysis Summary v2 (Year-level)

## What is new in v2

### Issue 0 — Double-counting fix for late-era double-entry pages (NEW)
Late-era pages (~1880–1900) use a double-entry layout where both income (left/Dr) and
expenditure (right/Cr) columns appear on the same physical page. Summing all row amounts
without correction double-counts activity on these pages.

Fix: the `side` field in supervisor JSON files marks each row as `"left"` (income) or
`"right"` (expenditure). When this field is present, the row's amount is multiplied by 0.5
before aggregation. This eliminates the double-count.

- Post-1840 vs pre-1760 nominal ratio  BEFORE fix: ~18.9×  →  AFTER fix: **14.9×**
- Post-1840 vs pre-1760 real ratio     BEFORE fix: ~11.4×  →  AFTER fix: **8.9×**

### Issue 1 — Price deflation
Nominal pound amounts have been deflated using the **Phelps Brown-Hopkins price
index** (base: 1700 = 100) to produce constant-price (real) equivalents.
This allows testing whether the observed growth in transaction scale persists
after removing inflation.

- Post-1840 vs pre-1760 nominal ratio : 14.9×
- Post-1840 vs pre-1760 real ratio    : 8.9×

A real ratio substantially smaller than the nominal ratio indicates that part
of the apparent expansion was driven by rising prices (especially the
post-1800 Napoleonic-era spike to ~270 on the index), not real activity growth.

### Issue 2 — Extended stop-word list
The following word categories are now excluded from all TF-IDF and
CountVectorizer analyses:
- **Time template**: year, years, yr, yrs
- **Accounting format**: total, balance, carried, forward, brought, account,
  acct, received, paid, payment, payments, sum, amounts
- **Currency units**: pound, pounds, shilling, shillings, pence, penny, sterling
- **Generic abbreviations**: ob, viz, ie, per

A token pattern `[a-z][a-z]+` also eliminates pure-digit tokens (e.g., "10").

### Issue 3 — Relative variables
Three new relative variables are added to the yearly summary:
- **entry_ratio** = n_entry_rows / n_data_rows
  Separates real transaction activity from additional subtotal rows.
- **data_rows_per_page** = n_data_rows / n_pages
  Normalises row volume by the number of pages covered each year.
- **header_diversity** = n_unique_header_texts / n_header_rows
  Tests whether header growth reflects new account categories or repetition.

Both entry_ratio and data_rows_per_page are included in change-point detection.

---

## Dataset overview
- Years covered: 196
- Data rows (entry+total): 49,187
- Header rows: 6,516
- Stop words added beyond sklearn English: 36

## Era-level summary (nominal and real)
                 era  years  mean_amount_sum_nominal  mean_amount_sum_real  median_amount_median_nominal  median_amount_median_real  mean_entry_ratio  mean_data_rows_per_page
industrial_1760_1840     81             25268.816152          14094.021778                      8.200000                   4.593723          0.915901                22.736086
           post_1840     55             84545.571941          49304.382820                      7.229167                   4.132795          0.876540                31.890855
            pre_1760     60              5673.049531           5540.156320                      2.057292                   1.991129          0.935525                28.393550

## Change points — yearly amount_sum (nominal)
- 1891: delta=217,759.51, robust_z=39.64
- 1880: delta=-215,516.93, robust_z=39.23
- 1892: delta=-194,020.40, robust_z=35.24
- 1879: delta=189,288.84, robust_z=34.36
- 1876: delta=-171,890.20, robust_z=31.13
- 1873: delta=163,485.56, robust_z=29.57
- 1893: delta=151,079.02, robust_z=27.26
- 1894: delta=-137,663.45, robust_z=24.77
- 1883: delta=68,050.17, robust_z=11.85
- 1889: delta=60,692.39, robust_z=10.48

## Change points — entry_ratio (accounting structure)
- 1885: delta=0.1585, robust_z=12.91
- 1883: delta=-0.1403, robust_z=11.32
- 1889: delta=-0.1369, robust_z=11.02
- 1884: delta=-0.1068, robust_z=8.38
- 1894: delta=0.0899, robust_z=6.90

## Key outputs
- yearly_numeric_summary_v2.csv   (includes entry_ratio, data_rows_per_page,
                                    price_index_1700base, amount_sum_real,
                                    amount_median_real)
- yearly_change_points_v2.csv     (now includes entry_ratio and
                                    data_rows_per_page metrics)
- year_top_terms_tfidf_v2.csv
- year_embedding_clusters_v2.csv
- era_top_terms_v2.csv            (stop-word-filtered)
- era_emerging_terms_v2.csv
- header_account_yearly_counts_v2.csv  (includes header_diversity)

## Key figures
- nominal_vs_real_amount.png      (NEW: deflation comparison)
- yearly_amount_and_volume.png
- yearly_median_p90.png           (now includes real median)
- change_points_amount_sum.png
- entry_ratio_over_time.png       (NEW)
- data_rows_per_page.png          (NEW)
- header_diversity_over_time.png  (NEW)
- year_embedding_clusters_timeline.png
- era_top_terms_heatmap.png       (clean terms, no noise words)
