# Annual Accounts 1700-1900
## A3 Page-Count Audit

Audit target:
- Raw source folder: [Annual_accounts_1700-1900](/Users/EthanJoo/PhD/Research/Archieve/Annual_accounts_1700-1900)

Generated outputs:
- PDF-level audit: [annual_accounts_pdf_page_audit.csv](/Users/EthanJoo/PhD/Research/Archieve/annual_accounts_pdf_page_audit.csv)
- Year-level summary: [annual_accounts_year_page_summary.csv](/Users/EthanJoo/PhD/Research/Archieve/annual_accounts_year_page_summary.csv)

Method:
- Counted PDF pages directly from the local raw files using `pypdf`.
- Parsed file names as either:
  - single-year volumes: `YYYY.pdf`
  - multi-year volumes: `YYYY-YYYY.pdf`
- For year-level summaries, two different concepts are kept separate:
  - `single_year_pdf_pages_exact`: exact page count for files labeled with one year only
  - `page_exposure_equal_split`: a conservative approximation that divides a multi-year volume equally across the years in its file name

Important limitation:
- Exact raw page counts by calendar year are **not fully identifiable** from file names alone when a PDF spans multiple years.
- For example, a file such as `1721-1722.pdf` has an exact total page count, but not an exact year-by-year page split unless the pages are inspected internally.

## Headline Findings

- Total PDFs in raw corpus: `193`
- Total raw PDF pages: `1,581`
- Multi-year PDFs: `33`
- Minimum PDF length: `1` page
- Maximum PDF length: `17` pages
- Median PDF length: `8` pages

Most important consistency check:
- The raw corpus total of `1,581` PDF pages matches the repository's repeated reference to `1,581 enriched pages`.
- This strongly suggests that the current analysis corpus is very close to the full raw PDF page universe, rather than a small sampled subset.

## Coverage Over Time

### Years with no raw PDF volume at all

- `1874`
- `1875`
- `1887`
- `1888`
- `1890`

These are true gaps at the raw-source level, not merely downstream processing gaps.

### Years without a standalone single-year PDF

These years are represented only through multi-year volumes, so year-level page counts are approximate unless pages are manually assigned:

- `1721`, `1722`
- `1759`, `1760`
- `1768`, `1769`, `1770`, `1771`, `1772`, `1773`, `1774`, `1775`, `1776`, `1777`, `1778`, `1779`, `1780`, `1781`, `1782`, `1783`, `1784`, `1785`, `1786`, `1787`, `1788`, `1789`, `1790`, `1791`, `1792`, `1793`, `1794`, `1795`, `1796`, `1797`, `1798`, `1799`

This matters for any paper claim about uneven yearly source density.

## Examples

- `1700.pdf` has `12` pages and cleanly maps to year `1700`.
- `1721-1722.pdf` has `13` pages in total.
  - Exact total is known.
  - An equal-split approximation gives `6.5` pages to `1721` and `6.5` pages to `1722`.
- `1799-1800.pdf` interacts with both a cross-year volume and a standalone `1800.pdf`.
  - This means `1800` has a mix of exact single-year pages and approximate carryover exposure from a multi-year source.

## Paper-Relevant Interpretation

- The raw source corpus is not evenly distributed across time.
- Temporal unevenness comes from two distinct sources:
  - true missing years with no source PDF
  - years represented only through cross-year volumes
- A paper should not present annual source density as if every year's page count were equally direct and equally observed.

## Recommended Use in the Paper

- Report `193` raw annual-account PDFs and `1,581` total raw pages.
- Explicitly list the years with no raw volume: `1874`, `1875`, `1887`, `1888`, `1890`.
- State that some years are represented only via cross-year volumes, so year-level raw page exposure is partially approximated unless manual page-level year assignment is performed.
- Use the year-level summary table as the basis for a coverage figure or appendix table.

## Recommended Next Step

- If yearly source density becomes substantively important in the paper, manually inspect the multi-year PDFs and assign their internal pages to specific years.
- If yearly source density is only a control or background issue, the equal-split approximation is acceptable as long as it is clearly disclosed.
