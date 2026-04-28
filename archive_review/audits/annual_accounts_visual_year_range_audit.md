# Annual Accounts 1700-1900
## Visual Audit of Multi-Year Volume Date Ranges

Audit target:
- Raw source folder: [Annual_accounts_1700-1900](/Users/EthanJoo/PhD/Research/Archieve/Annual_accounts_1700-1900)

Generated outputs:
- Audit table: [annual_accounts_visual_year_range_audit.csv](/Users/EthanJoo/PhD/Research/Archieve/annual_accounts_visual_year_range_audit.csv)
- Header contact sheets:
  - [headers_contact_1.jpg](/Users/EthanJoo/PhD/Research/Archieve/tmp/pdfs/header_crops/headers_contact_1.jpg)
  - [headers_contact_2.jpg](/Users/EthanJoo/PhD/Research/Archieve/tmp/pdfs/header_crops/headers_contact_2.jpg)
  - [headers_contact_3.jpg](/Users/EthanJoo/PhD/Research/Archieve/tmp/pdfs/header_crops/headers_contact_3.jpg)
  - [headers_contact_4.jpg](/Users/EthanJoo/PhD/Research/Archieve/tmp/pdfs/header_crops/headers_contact_4.jpg)

Method:
- For each multi-year PDF, I inspected the first-page image or a cropped header image derived from the raw PDF.
- The objective was narrow: verify the **volume-level year range** written on the page, not assign each internal page to a single calendar year.
- This is therefore a check on file-name reliability, not a full page-by-page dating exercise.

Important limitation:
- A title page can usually confirm which years a volume covers.
- It usually cannot tell us how many internal pages belong to the first year versus the second year.
- So this audit helps validate the source inventory, but it does **not** solve the annual page-allocation problem for cross-year volumes.

## Headline Findings

- Multi-year PDFs inspected: `33`
- Apparent file-name matches: `30`
- Apparent file-name mismatches: `2`
- Still unclear from inspected first-page images: `1`

Likely mismatches:
- `1721-1722.pdf` appears to show a title range of `1720-1721`
- `1759-1760.pdf` appears to show a title range of `1758-1759`

Still unclear:
- `1780-1781.pdf`
  - The visible crop confirms a start date of `2 Nov. 1780`
  - The terminal year is not visible in the inspected image set

## Why This Matters

- The raw corpus is generally well named, but not perfectly.
- For most late-eighteenth-century overlapping volumes, the file names appear trustworthy.
- For at least two earlier volumes, the file name may be offset by one year relative to the title page.

This matters because any paper statement about annual coverage can be wrong in two distinct ways:
- by treating a cross-year volume as if it were a precise calendar-year source
- by taking file names at face value when a small number of them may be misdated

## Paper-Relevant Interpretation

- For source-description purposes, the corpus can still be described as a `1700-1900` annual-accounts PDF archive with both single-year and cross-year volumes.
- For precise annual exposure claims, especially in the earlier period, the paper should state that multi-year file names were audited visually and that a small number appear misaligned with the title page.
- If the substantive analysis depends strongly on year-by-year source density, the safest next step is to manually verify the ambiguous or mismatched cross-year volumes before presenting fine-grained annual coverage claims.

## Recommended Use in the Memo or Paper

- In the main text:
  - state that the corpus contains both single-year and cross-year volumes
  - state that cross-year volume dates were visually spot-checked from title pages
  - note that most matched the file names, with a small number of earlier exceptions
- In an appendix:
  - include the audit table or summarize it in one sentence
  - explicitly distinguish `file-name year range` from `visually verified title-page range`

## Bottom Line

- The file-name layer is mostly reliable.
- It is not reliable enough to treat all multi-year volume names as unquestionable ground truth.
- For most purposes, this is a disclosure and documentation issue.
- It becomes a substantive issue only if annual source density or very fine-grained year timing is central to the empirical claims.
