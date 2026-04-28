# Draft
## Data Source

The empirical material is drawn from a corpus of scanned annual account volumes from **Exeter College, Oxford University**, covering the period `1700-1900`. The research corpus consists of page-image PDFs derived from handwritten account books rather than born-digital text files.

At the raw-source level, the corpus contains `193` PDF volumes and `1,581` total pages. The raw total of `1,581` pages matches the repository's repeated reference to `1,581` enriched pages, indicating that the processed corpus is close to the full raw page universe rather than a narrow sampled subset.

The archive should not be treated as a simple sequence of calendar-year observations. In the earlier part of the corpus, the file label functions most plausibly as an **accounting-year identifier** rather than as a strict calendar-year tag. Title formulas in early volumes repeatedly indicate account periods running approximately from `2 November` of one year to `2 November` of the next. This convention remains visible into part of the nineteenth century, although in some later files the relevant formula appears on a later page rather than on page `1`.

The corpus is therefore better understood as a sequence of evolving account volumes than as a uniform yearly text panel. Most files are labeled with a single year, but `33` volumes are explicitly cross-year files. Some apparent overlap is likely a consequence of accounting-year conventions rather than simple duplication. Raw-source gaps are limited: `1874`, `1875`, `1887`, `1888`, and `1890` do not appear as raw PDF volumes in the current corpus.

Despite historical variation in time labeling, the underlying **page structure is comparatively regular**. A typical page contains a heading or year label at the top, a date field on the left, a central block of entry descriptions, and a right-side set of pre-decimal currency columns for pounds, shillings, and pence. Many pages also conclude with a subtotal or total. This recurring structure provides the basis for later row extraction and variable construction.

The language of recordkeeping also changes over time. Early material relies heavily on Latin or Latinized accounting forms, while later material shows increasing use of English descriptions, headings, and account labels. The transition appears gradual rather than discrete, with many intermediate pages combining Latinized headings and English entries. The source material is therefore more accurately described as undergoing a **progressive vernacularization of accounting practice** than as exhibiting a binary switch from Latin to English.

Taken together, these features indicate that the archive is both structured and historically evolving. The corpus is sufficiently regular to support systematic extraction, but conventions concerning time labeling, page organization, and language use remain part of the data-generating process and are carried forward into later stages of data preparation.

## Suggested Figure

Suggested figure title:
- `Typical Structure of an Account Page`

Suggested example image:
- Annotated version: [exeter_college_typical_page_annotated.png](/Users/EthanJoo/PhD/Research/Archieve/exeter_college_typical_page_annotated.png)
- Source page: [1830_p1.jpg](/Users/EthanJoo/PhD/Research/Archieve/tmp/later_period_samples/1830_p1.jpg)

Suggested callouts:
- `Headings / Year Label`
- `Dates`
- `Entry Descriptions`
- `Amounts (£ | s. | d.)`
- `Subtotal / Total`

Suggested caption:
- `Example of a typical account-page layout from the Exeter College corpus. The illustrated page shows the recurring components used throughout much of the archive: headings and a year label at the top, dates on the left, entry descriptions in the center, and pre-decimal currency columns on the right for pounds, shillings, and pence, with a subtotal or total at the bottom. This regular structure underlies the subsequent extraction and structuring of page-level observations.`
