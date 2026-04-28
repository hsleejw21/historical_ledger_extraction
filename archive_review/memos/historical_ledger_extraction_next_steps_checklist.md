# Historical Ledger Extraction
## Next-Step Checklist

Use this as the practical follow-up to:
- [historical_ledger_extraction_archive_pipeline_review_draft.md](/Users/EthanJoo/PhD/Research/Archieve/historical_ledger_extraction_archive_pipeline_review_draft.md)

Goal:
- Move from a broad repository review to a concrete paper-preparation workflow.
- Prioritize the tasks that matter most for a substantive paper using the archive-derived data.

Core principle:
- Do not optimize for "best extraction framework" unless it directly changes the credibility of the substantive findings.
- Optimize for:
  - clarity of archival source and coverage,
  - reliability of the key variables,
  - robustness of the main historical/economic patterns,
  - and paper-ready documentation.

---

## 1. Immediate Priority: Freeze the Measurement Story

- **Task**: Write down the final canonical pipeline in one page.
- **What to do**:
  - State the archive/source being studied.
  - State the exact unit progression:
    - PDF/account book
    - page image
    - extracted row
    - enriched entry
    - year/decade aggregate
  - State the final scoring rule used for extraction evaluation.
  - State which semantic variables matter for the paper.
- **Output**:
  - A short internal methods note.
- **Why first**:
  - This prevents the project from drifting between inconsistent descriptions.

- **Task**: Freeze the final metric/version language.
- **What to do**:
  - Treat [src/evaluation/scorer.py](/tmp/historical_ledger_extraction_review/src/evaluation/scorer.py:395) as the canonical scorer.
  - Treat the `v6` rescored interpretation as the current unified evaluation layer for `v2` family comparisons.
  - Stop relying on the outdated metric description in `README.md`.
- **Output**:
  - A short "metric version note" for the team.
- **Why first**:
  - Otherwise every pipeline comparison remains unstable or ambiguous.

---

## 2. Source and Coverage: Clarify the Dataset Before Re-Running Anything

- **Task**: Get the raw data inventory from collaborators.
- **What to collect**:
  - archive/source name
  - institution scope
  - number of PDFs/books
  - time span
  - what each file represents
  - whether this is the full surviving series or a subset
- **Output**:
  - A source inventory table.
- **Why now**:
  - The biggest current paper-facing weakness is source definition, not code.

- **Task**: Build a coverage table.
- **What to include**:
  - years covered
  - pages per year/decade
  - missing or sparse periods
  - blank/non-transactional pages if relevant
  - excluded materials if any
- **Output**:
  - One CSV or spreadsheet and one simple figure.
- **Why now**:
  - This will shape how aggressively you can interpret long-run trends.

---

## 3. Reproducibility Check: Confirm the Pipeline Actually Runs

- **Task**: Reconstruct the missing directories once data arrive.
- **What to check**:
  - `data/images/`
  - `data/annual_accounts_1700-1900/` if used
  - `data/ground_truth/*.json` generated from `ground_truth.xlsx`
  - extraction cache location
- **Output**:
  - A short run log stating what exists and what had to be generated.
- **Why now**:
  - You need to know whether the paper is backed by a rerunnable pipeline or only by committed outputs.

- **Task**: Run a tiny smoke test.
- **What to do**:
  - Run the extraction pipeline on 1-2 pages.
  - Then run enrichment on those pages.
  - Then run one analysis script on those outputs.
- **Output**:
  - A minimal "smoke test passed" note.
- **Why now**:
  - This isolates path/config issues before you touch the full corpus.

- **Task**: Reproduce the 33-page evaluation subset.
- **What to do**:
  - Regenerate or load the evaluation pages.
  - Confirm that results are in the same neighborhood as the repository outputs.
- **Output**:
  - A replication note: exact match, close match, or mismatch.
- **Why now**:
  - This is the bridge between GitHub outputs and your actual paper evidence.

---

## 4. Variable Triage: Decide What the Paper Truly Depends On

- **Task**: Classify variables by evidentiary importance.
- **Suggested buckets**:
  - Tier A: central and relatively reliable
    - amount
    - direction
  - Tier B: useful but should be validated more carefully
    - category
    - language
    - is_arrears
  - Tier C: exploratory or appendix-only unless strengthened
    - payment_period
    - english_description
    - embedding-based variables
    - modernity/innovation indices
- **Output**:
  - A one-page variable-priority sheet.
- **Why now**:
  - The robustness burden should be proportional to the variable's role in the paper.

- **Task**: Mark each main result by dependency depth.
- **What to do**:
  - For each planned empirical claim, note whether it relies on:
    - raw transcription only,
    - semantic coding,
    - post-processing/aggregation,
    - or text-derived indices.
- **Output**:
  - A result-to-variable dependency map.
- **Why now**:
  - This tells you where reviewer skepticism will be highest.

---

## 5. Reliability Work: Validate the Variables That Matter Most

- **Task**: Summarize existing robustness evidence field-by-field.
- **What to pull from the repo**:
  - inter-model reliability
  - confidence-threshold sensitivity
  - sparse-year sensitivity
  - arrears sensitivity
  - year-weight sensitivity
- **Output**:
  - A compact internal validation table.
- **Why now**:
  - You already have a lot of evidence; it needs to be organized around the paper's main variables.

- **Task**: Add a manual audit for key semantic variables.
- **What to sample**:
  - by era
  - by page type
  - by confidence level
  - by category boundary difficulty
- **Priority variables**:
  - direction
  - category
  - arrears
- **Output**:
  - A hand-check table with agreement/error notes.
- **Why now**:
  - Manual spot-checking is often more persuasive to reviewers than more model-vs-model comparisons alone.

- **Task**: Inspect long pages and header-heavy pages.
- **What to look for**:
  - section-header propagation errors
  - ditto/Eidem cross-row dependence
  - batch-boundary issues in enrichment
- **Output**:
  - A short memo on likely systematic enrichment failure modes.
- **Why now**:
  - These are plausible sources of directional coding error.

---

## 6. Substantive Robustness: Stress-Test the Claims, Not Just the Pipeline

- **Task**: Re-run your main substantive patterns under conservative variants.
- **Minimum set**:
  - exclude arrears
  - confidence threshold filter
  - sparse-year exclusion
  - alternative year weighting for multi-year pages
  - alternative era cut
- **Output**:
  - A "main result stability" table.
- **Why now**:
  - This is the most important evidence if the paper's contribution is substantive.

- **Task**: Decide whether deflated values are central or supplementary.
- **What to do**:
  - Resolve the deflator naming/provenance issue.
  - Check whether the main conclusions depend on nominal or real amounts.
- **Output**:
  - A deflator note with one final recommended treatment.
- **Why now**:
  - Real-value claims are often reviewer magnets.

- **Task**: Separate robust claims from fragile claims.
- **What to do**:
  - Label each planned claim as:
    - robust enough for main text
    - acceptable with caveat
    - exploratory only
- **Output**:
  - A claim-priority list.
- **Why now**:
  - This prevents overclaiming.

---

## 7. Paper-Preparation Outputs to Build

- **Task**: Build a pipeline flowchart.
- **Output**:
  - one figure for appendix or methods section

- **Task**: Build a variable dictionary.
- **Output**:
  - one table with:
    - variable name
    - level
    - observed/inferred/constructed
    - brief definition
    - used in main text? yes/no

- **Task**: Build a validation summary table.
- **Output**:
  - one table with:
    - extraction quality summary
    - inter-model agreement
    - manual audit results
    - key robustness checks

- **Task**: Build a limitations paragraph.
- **Output**:
  - one short subsection covering:
    - surviving-record bias
    - model-based inference
    - weaker variables
    - non-public raw data if applicable

---

## 8. Recommended Working Order

If you want the shortest practical order, do this:

1. Freeze source definition and metric version.
2. Get raw data and confirm directory structure.
3. Run a 1-2 page smoke test.
4. Reproduce the 33-page evaluation subset.
5. Decide which variables are Tier A / B / C.
6. Build a compact validation table from existing robustness outputs.
7. Add manual audit for `direction`, `category`, and `arrears`.
8. Re-test main findings under conservative variants.
9. Draft the paper's Data Preparation / Validation section.

---

## 9. What Not to Prioritize Yet

Do not spend early time on:
- making the code elegant
- broad refactoring
- adding new extraction architectures
- squeezing out small extraction-score improvements

Unless:
- those changes materially alter the reliability of the variables used in the paper.

---

## 10. Minimum Deliverables for the Next Round

If you want a concrete target for the next work cycle, aim to leave it with these 6 outputs:

1. source inventory table
2. coverage over time table/figure
3. smoke-test run log
4. 33-page replication note
5. validation summary table
6. claim-stability table

If you have those, the project moves from "interesting repository" to "paper-preparation workflow with credible measurement support."

