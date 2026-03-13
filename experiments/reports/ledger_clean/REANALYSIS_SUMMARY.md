# Reanalysis Summary (Year-level)

## Scope understanding applied
- Unit of analysis: **year** (not page)
- Workbook structure recognized: sheet name = year_page_image or year-year_page_image
- Row types used: header / entry / total
- Text & numeric analysis performed on entry + total rows
- Industrial Revolution window highlighted: 1760-1840

## Dataset overview
- Years covered: 196
- Data rows (entry+total): 49,187
- Header rows: 6,516
- Unique normalized header/account texts: 2,917

## Era-level numeric summary
                 era  years  mean_amount_sum  median_amount_median  mean_data_rows
industrial_1760_1840     81     26864.152032              9.000000      213.864198
           post_1840     55    111881.290095              8.183333      238.981818
            pre_1760     60      5914.424983              2.073958      191.516667

## Change points (yearly amount_sum)
- 1891: delta=435,519.02, robust_z=72.65
- 1892: delta=-388,040.79, robust_z=64.64
- 1894: delta=-273,316.23, robust_z=45.29
- 1880: delta=-217,770.44, robust_z=35.92
- 1879: delta=189,947.10, robust_z=31.23
- 1876: delta=-178,735.51, robust_z=29.34
- 1893: delta=172,662.94, robust_z=28.31
- 1873: delta=170,740.51, robust_z=27.99
- 1883: delta=127,545.89, robust_z=20.70
- 1889: delta=121,384.77, robust_z=19.66

## Key outputs
- yearly_numeric_summary.csv
- yearly_change_points.csv
- year_top_terms_tfidf.csv
- year_embedding_clusters.csv
- year_cluster_profile_by_era.csv
- era_top_terms.csv
- era_emerging_terms.csv
- recurring_descriptions_first_appearance.csv
- header_account_first_appearance.csv
- header_account_yearly_counts.csv
- header_account_era_top_terms.csv

## Key figures
- yearly_amount_and_volume.png
- yearly_median_p90.png
- change_points_amount_sum.png
- year_embedding_clusters_timeline.png
- era_top_terms_heatmap.png
