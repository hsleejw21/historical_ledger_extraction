# Freeze Notes

These points appear sufficiently stable to freeze for the `Data Source` subsection:

- The source archive is a corpus of scanned annual account volumes from **Exeter College, Oxford University**.
- The usable research input is a set of page-image PDFs derived from handwritten account books.
- The raw corpus spans `1700-1900`.
- The raw corpus contains `193` PDF files and `1,581` pages.
- The page universe appears to align closely with the downstream processed universe referenced in the repository.
- The archive is structurally regular enough to describe as a ledger-like columnar source with dates or time markers on the left, entries in the center, and monetary amounts on the right.
- Early records are best interpreted through an accounting-year convention rather than a strict calendar-year convention.
- A recurring early formula is approximately `2 Nov. (t-1)` to `2 Nov. t`.
- Some later files still preserve this convention, but the relevant title formula may appear on later pages instead of page `1`.
- At least some late nineteenth-century files look more like year-end general accounts.
- The language of recordkeeping appears to shift gradually from Latin / Latinized forms toward English.
- Missing raw years are limited and can be stated explicitly rather than treated as a central sampling problem.

These points are probably better left outside the narrow `Data Source` subsection and moved to later parts of `Data Preparation`:

- Detailed discussion of file-name mismatches
- Page-level year allocation for cross-year volumes
- Source limitations beyond a brief transition sentence
- Consequences for variable construction and aggregation
