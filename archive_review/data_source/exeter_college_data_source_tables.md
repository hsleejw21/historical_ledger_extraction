# Draft Tables

## Table 1. Source Corpus Summary

| Item | Value | Notes |
| --- | --- | --- |
| Archive / institution | Exeter College, Oxford University | Annual account volumes / ledger-like account books |
| Raw digital format | Page-image PDFs | Scanned handwritten historical account pages, not born-digital text |
| Time span | `1700-1900` | Long-run historical archive |
| Number of PDF volumes | `193` | Based on local raw-source audit |
| Total raw pages | `1,581` | Matches the repository's repeated reference to `1,581` enriched pages |
| Multi-year volumes | `33` | Cross-year file labels present in raw corpus |
| Missing raw years | `1874`, `1875`, `1887`, `1888`, `1890` | True raw-source gaps |
| Typical early time convention | Accounting year | Often approximately `2 Nov. (t-1)` to `2 Nov. t` |
| Typical later time convention | More mixed | Some nineteenth-century files retain accounting-year logic; late nineteenth-century samples look closer to year-end general accounts |
| Page structure | Ledger-like columnar format | Dates or time markers on left, descriptions in center, monetary values on right |
| Language | Historically evolving | Early Latin / Latinized forms, later increasingly English, with mixed-language intermediate pages |

## Table 2. Source Coverage and Archival Conventions by Period

| Period | Typical time convention | Typical language pattern | Typical page format | Implication for data preparation |
| --- | --- | --- | --- | --- |
| Early 18th century | Accounting-year labeling centered on the ending year | Predominantly Latin or Latinized accounting forms with some English elements | Highly regular ledger-like list structure | Year labels should not be read as strict calendar years |
| Late 18th century | Accounting-year logic still prominent; some cross-year volumes | Mixed Latinized headings and increasingly vernacular entries | Similar columnar structure with recurring headers and amounts | Extraction remains structurally feasible, but language coding should allow mixed cases |
| Early 19th century | Mixed regime; accounting-year formulas sometimes move off page `1` | Strongly mixed, with growing English usage | Sectional account pages and more varied heading placement | Time interpretation may require checking later title pages, not just first pages |
| Mid 19th century | Less uniform from first-page inspection alone | Predominantly English entries, with residual traditional forms | More sectional and report-like account pages | Page layout remains structured, but temporal conventions should not be inferred mechanically from file names alone |
| Late 19th century | Samples increasingly resemble year-end general accounts | Largely English headings and descriptions | General-account presentation with balance and revenue/payment sections | Later-period files may be more compatible with calendar-year interpretation, but this should still be verified case by case |

## Suggested Appendix Table

Suggested appendix table title:
- `Appendix Table A1. File-Level Inventory of Raw Source Volumes`

Suggested columns:
- `File name`
- `Labeled year(s)`
- `Single-year / multi-year`
- `Page count`
- `Best-guess time interpretation`
- `Needs recheck`
- `Notes`

Suggested source:
- [annual_accounts_pdf_page_audit.csv](/Users/EthanJoo/PhD/Research/Archieve/annual_accounts_pdf_page_audit.csv)
- [annual_accounts_file_time_audit.csv](/Users/EthanJoo/PhD/Research/Archieve/annual_accounts_file_time_audit.csv)
