# Historical Ledger Extraction Repository
## Archive Pipeline Review Draft

Prepared for paper-development purposes using the checklist in "Archive Pipeline Review Checklist for Codex".

Review target:
- Repository: `Ethanjoo/historical_ledger_extraction`
- Reviewed clone: `/tmp/historical_ledger_extraction_review`
- Commit reviewed: `e205d66c7a07abf724b15feeaed725e508f3d0c9`

Review framing:
- This review treats the repository primarily as a data-construction pipeline for a management / OM / economic-history style paper.
- The main question is not whether the repository proposes the best OCR / extraction framework.
- The main question is whether the repository provides a transparent, reproducible, and methodologically defensible basis for archival-data construction and downstream substantive analysis.

Working interpretation of paper position:
- The paper's core contribution is more likely to be the substantive historical/economic patterns inferred from the ledger data than a methodological contribution in document extraction.
- Accordingly, the most important standard is not "state-of-the-art extraction performance" by itself, but whether the extraction/enrichment process yields variables that are reliable enough for the main substantive claims.

---

## A. Source and Coverage

- **Item**: A1. Archival source definition
- **Status**: Partially addressed
- **Evidence in repo**:
  - The repository-level description states that the project concerns "18th–19th century English parish accounting ledgers" in [README.md](/tmp/historical_ledger_extraction_review/README.md:1).
  - The enrichment script instead frames the source as "historical Oxford college financial ledgers" in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:51).
  - `run_all.py` expects archival PDFs in `data/annual_accounts_1700-1900/` and converts them to page images in `data/images/`, suggesting a full annual-accounts archive in [run_all.py](/tmp/historical_ledger_extraction_review/run_all.py:4).
- **Why it matters**:
  - A reviewer will want to know exactly what archive the paper studies, whether it covers a single institution, which account books are included, and whether the sample represents the full financial record or only a surviving subset.
  - The "parish ledgers" versus "Oxford college ledgers" inconsistency is not cosmetic; it changes the implied institution, coding perspective, and external validity.
- **What still needs to be done**:
  - Define the archive unambiguously in paper-ready language.
  - State institution, account-book type, time span, and whether the corpus includes the full surviving series or a selected subset.
  - Add a provenance table linking archive source, PDF inventory, page images, and final usable pages.
- **Risk type**: substantive validity, reproducibility

- **Item**: A2. Unit of observation
- **Status**: Well addressed
- **Evidence in repo**:
  - The extraction pipeline is page-based in [pipeline/run_pipeline.py](/tmp/historical_ledger_extraction_review/pipeline/run_pipeline.py:143).
  - Extracted outputs are row-based JSON objects in [src/agents/standalone_extractor.py](/tmp/historical_ledger_extraction_review/src/agents/standalone_extractor.py:25) and [src/agents/supervisor.py](/tmp/historical_ledger_extraction_review/src/agents/supervisor.py:24).
  - Enrichment is row-level and explicitly distinguishes `header`, `entry`, and `total` rows in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:140).
  - Downstream analysis expands enriched rows to year-level observations in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:151).
- **Why it matters**:
  - The paper will need to distinguish clearly between page, row, enriched entry, and year/era aggregate.
  - This is already reconstructable from code, which is a strength.
- **What still needs to be done**:
  - Add an explicit unit-of-observation paragraph in the paper.
  - Include one flowchart or figure showing transitions from page to row to enriched row to year/decade aggregates.
- **Risk type**: documentation/reproducibility

- **Item**: A3. Coverage over time
- **Status**: Partially addressed
- **Evidence in repo**:
  - The README reports `1,581 enriched pages (1700–1900)` in [README.md](/tmp/historical_ledger_extraction_review/README.md:25).
  - The enrichment outputs span 1700–1900 and are visibly uneven by decade.
  - The robustness scripts explicitly test sparse-year sensitivity in [experiments/robustness/measurement_validation.py](/tmp/historical_ledger_extraction_review/experiments/robustness/measurement_validation.py:1) and summarize sparse-year exclusion effects in [experiments/reports/robustness/measurement/measurement_summary.txt](/tmp/historical_ledger_extraction_review/experiments/reports/robustness/measurement/measurement_summary.txt:103).
- **Why it matters**:
  - Long-run time-series claims can be distorted if some decades are much better represented than others.
  - The repo recognizes this problem, but the archive-level source of uneven coverage is not yet narratively documented.
- **What still needs to be done**:
  - Report pages per year and pages per decade in the paper.
  - Distinguish archival absence, missing scans, non-transactional pages, and pages excluded by the pipeline.
  - State whether sparse periods reflect the archive itself or pipeline/data availability.
- **Risk type**: substantive validity, robustness

- **Item**: A4. Representativeness
- **Status**: Not yet addressed / unclear
- **Evidence in repo**:
  - The repo makes it possible to see that the analysis is based on surviving ledger/account pages, but not to evaluate representativeness against the full institutional universe.
  - No explicit archive-manifest or exclusion ledger is present in the default clone.
- **Why it matters**:
  - Economic-history claims often fail not because coding is weak, but because the paper quietly treats surviving records as if they are the full underlying activity.
  - Reviewers will ask whether observed shifts reflect institutional behavior or shifts in what the archive records.
- **What still needs to be done**:
  - Add an archival-limitations subsection.
  - Clarify whether the data represent all transactions, selected ledgers, annual summaries, or surviving parts of a broader bookkeeping system.
- **Risk type**: substantive validity

---

## B. Source-to-Variable Pipeline Reconstruction

- **Item**: B1. End-to-end pipeline clarity
- **Status**: Well addressed
- **Evidence in repo**:
  - `run_all.py` documents PDF-to-image conversion and production extraction in [run_all.py](/tmp/historical_ledger_extraction_review/run_all.py:4).
  - `pipeline/run_pipeline.py` defines the production extraction workflow in [pipeline/run_pipeline.py](/tmp/historical_ledger_extraction_review/pipeline/run_pipeline.py:143).
  - `experiments/enrichment/enrich_supervisor_rows.py` defines semantic enrichment in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:1).
  - `experiments/analysis/analysis_enriched.py` constructs the analysis-ready dataset in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:151).
- **Why it matters**:
  - For paper preparation, it is a major advantage that the pipeline can already be reconstructed as:
    1. archival PDF/pages
    2. page extraction
    3. row structuring
    4. row enrichment
    5. year/era aggregation
- **What still needs to be done**:
  - Turn this implicit code-based reconstruction into an explicit paper flowchart.
  - Clarify which parts are "production" and which are "research-only" scripts.
- **Risk type**: documentation/reproducibility

- **Item**: B2. Stage outputs
- **Status**: Partially addressed
- **Evidence in repo**:
  - Extraction outputs are cached as JSON in production or experiment result directories in [pipeline/run_pipeline.py](/tmp/historical_ledger_extraction_review/pipeline/run_pipeline.py:147) and [experiments/run_experiment.py](/tmp/historical_ledger_extraction_review/experiments/run_experiment.py:303).
  - Enriched outputs are stored page-by-page in `experiments/results/enriched/`.
  - Analysis scripts write many CSV and report outputs to `experiments/reports/`.
  - The repo includes many generated outputs, including `pipeline/cache.zip`.
- **Why it matters**:
  - Inspectable intermediate outputs support traceability and reviewer confidence.
  - However, the default clone does not contain all expected intermediate input directories in unpacked form.
- **What still needs to be done**:
  - Provide a clear inventory of which outputs are committed, which must be regenerated, and which depend on external/raw data not distributed on GitHub.
  - Document how `pipeline/cache.zip`, `experiments/results/v2`, and `experiments/results/enriched` relate to each other.
- **Risk type**: reproducibility/documentation

- **Item**: B3. Observed vs inferred variables
- **Status**: Partially addressed
- **Evidence in repo**:
  - Directly observed extraction fields are row text, amounts, row type, and sometimes side in [src/agents/standalone_extractor.py](/tmp/historical_ledger_extraction_review/src/agents/standalone_extractor.py:66).
  - Model-inferred semantic fields are defined in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:140).
  - Rule-based constructed fields include `section_header` propagation in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:324), year splitting and year weights in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:176), and price deflation in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:217).
- **Why it matters**:
  - A methods section should clearly separate observed transcription, model-coded inference, and rule-based aggregation.
  - That distinction is visible in code but not yet narrated for a paper audience.
- **What still needs to be done**:
  - Add a variable dictionary with four columns:
    - variable name
    - level
    - observed/inferred/rule-based/aggregated
    - role in analysis
- **Risk type**: documentation/reproducibility, substantive validity

- **Item**: B4. Dependency structure
- **Status**: Partially addressed
- **Evidence in repo**:
  - Downstream semantic fields depend on extracted row text and propagated `section_header` in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:324).
  - Yearly and era summaries depend on year parsing, year weighting, deflation, and enrichment fields in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:169).
- **Why it matters**:
  - Some conclusions depend mostly on low-level numeric transcription.
  - Others depend on more inference-heavy semantic labeling and post-processing.
  - The paper should distinguish these dependency depths.
- **What still needs to be done**:
  - Explicitly identify which headline findings rely on:
    - transcription only,
    - semantic coding,
    - or heavy post-processing.
- **Risk type**: substantive validity

---

## C. Ambiguity, Conflict Resolution, and Researcher Degrees of Freedom

- **Item**: C1. Ambiguous rows
- **Status**: Partially addressed
- **Evidence in repo**:
  - The standalone extractor prompt explicitly instructs the model to record ambiguity via confidence and notes in [src/prompts/standalone_extractor.py](/tmp/historical_ledger_extraction_review/src/prompts/standalone_extractor.py:125).
  - Extractor outputs contain `confidence_score` and `notes`, and rows are sanitized rather than dropped in [src/agents/standalone_extractor.py](/tmp/historical_ledger_extraction_review/src/agents/standalone_extractor.py:60).
- **Why it matters**:
  - This is good evidence that ambiguous cases are retained rather than silently discarded.
  - But the paper still needs to say what happens analytically to ambiguous rows.
- **What still needs to be done**:
  - Report whether low-confidence rows were retained in all analyses, filtered in some checks, or used only for robustness.
  - Add a small manual audit of hard-to-read rows.
- **Risk type**: robustness/credibility, substantive validity

- **Item**: C2. Model disagreement
- **Status**: Well addressed
- **Evidence in repo**:
  - The v2 pipeline explicitly uses multiple extractors followed by a supervisor in [src/config.py](/tmp/historical_ledger_extraction_review/src/config.py:71).
  - The supervisor prompt gives a clear disagreement-resolution protocol in [src/prompts/supervisor.py](/tmp/historical_ledger_extraction_review/src/prompts/supervisor.py:22).
  - The supervisor implementation preserves metadata about candidate models in [src/agents/supervisor.py](/tmp/historical_ledger_extraction_review/src/agents/supervisor.py:78).
- **Why it matters**:
  - This is one of the most methodologically reassuring design features in the repo.
  - Disagreement is not hidden; it is part of the extraction architecture.
- **What still needs to be done**:
  - In the paper, describe this as an arbitration design that reduces dependence on any single model.
  - Report how often supervisor disagreement mattered, if that metadata can be tabulated from caches.
- **Risk type**: robustness/credibility

- **Item**: C3. Confidence information
- **Status**: Partially addressed
- **Evidence in repo**:
  - Confidence scores are explicitly generated by extractors and supervisor in [src/prompts/standalone_extractor.py](/tmp/historical_ledger_extraction_review/src/prompts/standalone_extractor.py:126) and [src/prompts/supervisor.py](/tmp/historical_ledger_extraction_review/src/prompts/supervisor.py:79).
  - Confidence is loaded into the analysis dataset in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:199).
  - Confidence-threshold sensitivity is already reported in [experiments/reports/robustness/measurement/measurement_summary.txt](/tmp/historical_ledger_extraction_review/experiments/reports/robustness/measurement/measurement_summary.txt:96).
- **Why it matters**:
  - Confidence can support conservative subsample checks.
  - This is a useful validity tool even if confidence is not used in the main analysis.
- **What still needs to be done**:
  - Clarify in the paper whether confidence affects the main dataset or appears only in robustness analysis.
  - If possible, summarize the share of entries above each confidence threshold.
- **Risk type**: robustness/credibility

- **Item**: C4. Prompt dependence / model dependence
- **Status**: Partially addressed
- **Evidence in repo**:
  - Prompt wording is central to extraction and enrichment in [src/prompts/standalone_extractor.py](/tmp/historical_ledger_extraction_review/src/prompts/standalone_extractor.py:15), [src/prompts/supervisor.py](/tmp/historical_ledger_extraction_review/src/prompts/supervisor.py:22), and [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:140).
  - Inter-model reliability is explicitly tested in [experiments/robustness/reliability_metrics.py](/tmp/historical_ledger_extraction_review/experiments/robustness/reliability_metrics.py:1), with summary output in [experiments/reports/robustness/reliability/reliability_summary.txt](/tmp/historical_ledger_extraction_review/experiments/reports/robustness/reliability/reliability_summary.txt:1).
- **Why it matters**:
  - For a paper whose contribution is substantive rather than methodological, it is enough to show that main variables are not overly sensitive to model choice.
  - The repo partially does this already.
- **What still needs to be done**:
  - Make clear which variables are robust enough for headline claims and which are weaker.
  - Note that prompt wording and model aliases are still part of the measurement process and should be frozen/versioned in the paper appendix.
- **Risk type**: robustness/credibility, substantive validity

- **Item**: C5. Hidden researcher choices
- **Status**: Partially addressed
- **Evidence in repo**:
  - Important choices are embedded in code:
    - scorer weights and matching logic in [src/evaluation/scorer.py](/tmp/historical_ledger_extraction_review/src/evaluation/scorer.py:242)
    - year weighting in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:176)
    - category ontology in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:161)
    - era boundaries and deflator assumptions in [experiments/robustness/measurement_validation.py](/tmp/historical_ledger_extraction_review/experiments/robustness/measurement_validation.py:97)
- **Why it matters**:
  - These choices are exactly the kinds of implicit defaults reviewers call "researcher degrees of freedom."
- **What still needs to be done**:
  - Promote these defaults into a paper appendix called "Key Researcher Choices" or similar.
  - Clearly distinguish pre-specified choices from sensitivity-tested alternatives.
- **Risk type**: substantive validity, reproducibility

---

## D. Systematic Error / Bias Structure

- **Item**: D1. Category-specific vulnerability
- **Status**: Partially addressed
- **Evidence in repo**:
  - The category ontology is explicit and fairly rich in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:161).
  - Inter-model agreement for `category` is decent but not near-perfect in [experiments/reports/robustness/reliability/reliability_summary.txt](/tmp/historical_ledger_extraction_review/experiments/reports/robustness/reliability/reliability_summary.txt:14).
- **Why it matters**:
  - Categories with fuzzy boundaries, such as `administrative`, `financial`, and `other`, are likely to be more vulnerable than gross distinctions such as income vs expenditure.
- **What still needs to be done**:
  - Add a manual error audit focused on borderline category distinctions.
  - Avoid leaning too hard on the finest semantic category splits unless manually validated.
- **Risk type**: substantive validity

- **Item**: D2. Time-specific vulnerability
- **Status**: Partially addressed
- **Evidence in repo**:
  - The enrichment prompt explicitly acknowledges Latin-heavy earlier periods and embedded amounts in early eras in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:51).
  - Language-shift and confidence-threshold robustness are reported in the robustness outputs.
- **Why it matters**:
  - If early pages are harder to parse, apparent long-run shifts may partly reflect time-varying measurement error.
- **What still needs to be done**:
  - Compare extraction/confidence behavior by era.
  - Consider a manual validation sample stratified by early/middle/late periods.
- **Risk type**: substantive validity, robustness

- **Item**: D3. Layout/page-type vulnerability
- **Status**: Partially addressed
- **Evidence in repo**:
  - The extraction prompt explicitly distinguishes standard versus complex dual-column pages in [src/prompts/standalone_extractor.py](/tmp/historical_ledger_extraction_review/src/prompts/standalone_extractor.py:20).
  - The supervisor prompt also treats row-count disagreement and structural variation as meaningful in [src/prompts/supervisor.py](/tmp/historical_ledger_extraction_review/src/prompts/supervisor.py:43).
- **Why it matters**:
  - Balance-sheet pages, dual-column layouts, and pages with unusual ruling lines may have systematically different error profiles.
- **What still needs to be done**:
  - Report whether page layout type is associated with lower extraction quality or lower confidence.
  - If layout metadata are recoverable, include layout-specific robustness checks.
- **Risk type**: substantive validity, robustness

- **Item**: D4. Systematic distortion risk
- **Status**: Partially addressed
- **Evidence in repo**:
  - The repo contains multiple robustness checks suggesting many main patterns survive reasonable alternatives in [experiments/reports/robustness/measurement/measurement_summary.txt](/tmp/historical_ledger_extraction_review/experiments/reports/robustness/measurement/measurement_summary.txt:1).
  - But some quantities, especially arrears-sensitive direction shares in early decades, move materially under alternative treatment.
- **Why it matters**:
  - Remaining error need not be random.
  - If early Latin-heavy arrears pages are harder to code, the resulting distortion could be directional rather than mean-zero.
- **What still needs to be done**:
  - Identify which headline findings are highly stable and which are materially sensitive.
  - Write limitations accordingly rather than treating all variables as equally secure.
- **Risk type**: substantive validity

---

## E. Variable Construction and Harmonization

- **Item**: E1. Key constructed variables
- **Status**: Partially addressed
- **Evidence in repo**:
  - `direction`, `category`, `language`, `payment_period`, `person_name`, and `place_name` are explicitly defined in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:140).
  - Deflated values and year-level weighting are explicit in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:217).
  - More advanced constructs, including modernity/innovation style measures, appear in downstream analysis scripts such as [experiments/analysis/reanalysis_ledger_yearly_v2.py](/tmp/historical_ledger_extraction_review/experiments/analysis/reanalysis_ledger_yearly_v2.py:1).
- **Why it matters**:
  - The repo is strong on explicit coding definitions, but not all constructed variables are equally validated.
- **What still needs to be done**:
  - Divide variables into:
    - core variables used for main claims
    - exploratory derived variables
  - Treat advanced indices and text-derived measures more cautiously in the paper.
- **Risk type**: substantive validity

- **Item**: E2. Construction rules
- **Status**: Well addressed
- **Evidence in repo**:
  - Construction rules are spelled out in long-form prompts and helper functions in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:140).
  - Validation and fallback rules are explicit in [experiments/enrichment/enrich_supervisor_rows.py](/tmp/historical_ledger_extraction_review/experiments/enrichment/enrich_supervisor_rows.py:401).
- **Why it matters**:
  - This is strong material for a methods appendix because it makes latent coding assumptions inspectable.
- **What still needs to be done**:
  - Convert prompt-based definitions into a concise paper variable dictionary.
  - Note where alternative reasonable definitions could exist.
- **Risk type**: documentation/reproducibility

- **Item**: E3. Aggregation logic
- **Status**: Partially addressed
- **Evidence in repo**:
  - Year splitting and year weights are explicit in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:176).
  - Year/era aggregation and share construction are explicit throughout `analysis_enriched.py`.
- **Why it matters**:
  - Aggregation rules are among the most consequential parts of the pipeline for substantive inference.
  - They are visible in code but not yet narratively justified.
- **What still needs to be done**:
  - Explain how multi-year pages are allocated to years.
  - State which results use counts, weighted counts, nominal amounts, or real amounts.
- **Risk type**: substantive validity, reproducibility

- **Item**: E4. Harmonization choices
- **Status**: Partially addressed
- **Evidence in repo**:
  - Deflator anchors are explicit in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:46).
  - Era boundaries are explicit in [experiments/analysis/analysis_enriched.py](/tmp/historical_ledger_extraction_review/experiments/analysis/analysis_enriched.py:74).
  - `measurement_validation.py` reveals that the baseline deflator is more Clark-consistent than genuine PBH in [experiments/robustness/measurement_validation.py](/tmp/historical_ledger_extraction_review/experiments/robustness/measurement_validation.py:78).
- **Why it matters**:
  - Harmonization choices are explicit, which is good.
  - But the deflator labeling issue is methodologically important and should be corrected before publication.
- **What still needs to be done**:
  - Freeze the canonical deflator and label it correctly everywhere.
  - Justify era boundaries historically rather than only analytically.
- **Risk type**: substantive validity, reproducibility

- **Item**: E5. Post-processing choices that matter substantively
- **Status**: Partially addressed
- **Evidence in repo**:
  - The robustness layer already tests era cuts, deflation, year weights, arrears treatment, confidence thresholds, sparse-year exclusion, and change-point thresholds in [experiments/robustness/measurement_validation.py](/tmp/historical_ledger_extraction_review/experiments/robustness/measurement_validation.py:1).
- **Why it matters**:
  - This is one of the repo's strongest assets.
  - It means the most obvious post-processing choices are not hidden.
- **What still needs to be done**:
  - Link each robustness test directly to the specific substantive claim it protects.
  - Make clear which findings remain sensitive despite the robustness battery.
- **Risk type**: robustness/credibility, substantive validity

---

## F. Validation and Robustness Layers

- **Item**: F1. Internal consistency checks
- **Status**: Well addressed
- **Evidence in repo**:
  - Currency constraints are embedded in prompts and supervisor logic in [src/prompts/standalone_extractor.py](/tmp/historical_ledger_extraction_review/src/prompts/standalone_extractor.py:67) and [src/prompts/supervisor.py](/tmp/historical_ledger_extraction_review/src/prompts/supervisor.py:55).
  - Scoring logic checks structure and amount consistency in [src/evaluation/scorer.py](/tmp/historical_ledger_extraction_review/src/evaluation/scorer.py:161).
  - Header-direction validation appears in the robustness suite and report outputs.
- **Why it matters**:
  - Internal consistency checks are exactly what reviewers expect from an archival data-construction pipeline.
- **What still needs to be done**:
  - Summarize these checks in one validation table in the paper.
- **Risk type**: robustness/credibility

- **Item**: F2. Cross-model validation
- **Status**: Well addressed
- **Evidence in repo**:
  - `reliability_metrics.py` computes pairwise and multi-rater agreement across three models in [experiments/robustness/reliability_metrics.py](/tmp/historical_ledger_extraction_review/experiments/robustness/reliability_metrics.py:1).
  - The resulting summary in [experiments/reports/robustness/reliability/reliability_summary.txt](/tmp/historical_ledger_extraction_review/experiments/reports/robustness/reliability/reliability_summary.txt:1) is highly useful for a paper appendix.
- **Why it matters**:
  - For a paper not centered on method innovation, this is probably more persuasive than trying to claim extraction SOTA.
- **What still needs to be done**:
  - Translate these reliability statistics into a paper-ready table.
  - Interpret field-by-field variation rather than reporting one aggregate reassurance.
- **Risk type**: robustness/credibility

- **Item**: F3. External/historical validation
- **Status**: Partially addressed
- **Evidence in repo**:
  - Some historical contextual rules are embedded in prompts and analysis notes.
  - The measurement and robustness layer references historically grounded deflators and institutional conventions.
  - However, I did not find a systematic external validation file comparing outputs to independent historical benchmarks beyond limited manual and model-based checks.
- **Why it matters**:
  - Reviewers in economic history will often want at least one bridge between model-coded variables and external historical plausibility.
- **What still needs to be done**:
  - Add targeted historical validation:
    - legislative/event-date alignment
    - known institutional shifts
    - manually verified benchmark pages or account categories
- **Risk type**: robustness/credibility, substantive validity

- **Item**: F4. Sensitivity analysis
- **Status**: Well addressed
- **Evidence in repo**:
  - The sensitivity battery is extensive in `measurement_validation.py`.
  - The summary in [experiments/reports/robustness/measurement/measurement_summary.txt](/tmp/historical_ledger_extraction_review/experiments/reports/robustness/measurement/measurement_summary.txt:1) covers many meaningful alternatives.
- **Why it matters**:
  - This is probably the strongest reviewer-facing asset of the current repository.
- **What still needs to be done**:
  - Present the sensitivity results selectively around the main claims, rather than as a large undifferentiated appendix dump.
- **Risk type**: robustness/credibility

- **Item**: F5. Coverage/confidence robustness
- **Status**: Well addressed
- **Evidence in repo**:
  - Confidence-threshold and sparse-year robustness are already summarized in [experiments/reports/robustness/measurement/measurement_summary.txt](/tmp/historical_ledger_extraction_review/experiments/reports/robustness/measurement/measurement_summary.txt:96).
- **Why it matters**:
  - These are directly responsive to reviewer concerns about low-quality rows and weakly covered periods.
- **What still needs to be done**:
  - In the paper, connect these checks explicitly to your main historical patterns.
- **Risk type**: robustness/credibility

---

## G. Paper-Readiness of the Data Preparation Section

- **Item**: G1. What can already be described clearly in a paper?
- **Status**: Well addressed
- **Evidence in repo**:
  - The extraction architecture is clear.
  - The enrichment ontology is explicit.
  - The robustness layer is substantial.
- **Why it matters**:
  - The project already has enough material for a serious data-preparation and validation section.
- **What still needs to be done**:
  - Convert code-implicit logic into paper-explicit prose.
- **Risk type**: documentation/reproducibility

- **Item**: G2. What still needs to be explained explicitly?
- **Status**: Partially addressed
- **Evidence in repo**:
  - The code reveals many choices, but the repo does not yet narrate them as a paper would.
  - The archival source definition, representativeness, deflator naming, and final metric version all need explicit explanation.
- **Why it matters**:
  - These are exactly the parts a reviewer would question even if the code is internally coherent.
- **What still needs to be done**:
  - Add narrative explanation for source provenance, coding levels, model dependence, and key post-processing choices.
- **Risk type**: substantive validity, documentation

- **Item**: G3. What would need a table / figure / appendix?
- **Status**: Partially addressed
- **Evidence in repo**:
  - The repo already contains enough material to support:
    - a pipeline flowchart
    - a variable dictionary
    - a validation summary table
    - a robustness summary table
  - The raw pieces are present but not assembled into paper-facing exhibits.
- **Why it matters**:
  - Paper-readiness now depends more on synthesis than on new pipeline construction.
- **What still needs to be done**:
  - Build:
    - archive coverage table
    - pipeline diagram
    - variable dictionary
    - reliability table
    - robustness summary table
    - limitations subsection
- **Risk type**: documentation/reproducibility, robustness

- **Item**: G4. Remaining limitations
- **Status**: Partially addressed
- **Evidence in repo**:
  - The repository already acknowledges some sensitivity and uncertainty through robustness scripts and confidence scores.
  - But archival representativeness, deflator provenance, and full rerunnability remain incompletely addressed.
- **Why it matters**:
  - These limitations are acceptable if stated clearly.
  - They become serious only if left implicit.
- **What still needs to be done**:
  - Separate acceptable limitations from unresolved threats.
  - State clearly what the data can and cannot support.
- **Risk type**: substantive validity, reproducibility

---

## H. Final Synthesis

### H1. Summary diagnosis

From a paper-readiness perspective, the repository is stronger than a typical research codebase on validation and methodological self-awareness, but weaker than it needs to be on archival provenance, full rerunnability from raw source material, and narrative explanation of key data-construction choices. The extraction/enrichment/analysis stack is reconstructable and reasonably inspectable. The main remaining work is not code refactoring. It is turning the existing pipeline into a paper-ready measurement protocol: clearly defining the archive, freezing the final metric/versioning story, identifying which variables are sufficiently reliable for headline claims, and documenting where substantive conclusions depend on post-processing assumptions.

### H2. Top strengths

1. The end-to-end pipeline can already be reconstructed clearly from raw pages to analysis-ready data.
2. The repository preserves many intermediate outputs, including cached extraction JSON and enriched page-level JSON.
3. Multi-model disagreement is handled explicitly through a supervisor/arbitration step rather than being hidden.
4. Confidence scores and row-level notes are built into extraction, which supports conservative robustness checks.
5. The semantic enrichment ontology is explicit and unusually well documented for a research pipeline.
6. The robustness layer is substantial and directly relevant to reviewer concerns.
7. Cross-model reliability is already quantified for key semantic variables.
8. The code already distinguishes, in practice, between extraction, enrichment, and analysis rather than collapsing all decisions into one opaque script.

### H3. Top vulnerabilities

1. The archival source is not yet defined consistently or precisely enough for a paper.
2. The default GitHub clone is not fully rerunnable end-to-end because key raw inputs and some expected intermediate directories are missing.
3. README-level metric descriptions are inconsistent with the actual scorer implementation.
4. The deflator provenance is internally inconsistent and needs to be corrected.
5. Some economically interesting variables, especially `payment_period`, are materially weaker than others.
6. Header propagation and long-page batching create downstream dependence structures that are not yet explicitly discussed.
7. Representativeness of surviving records versus underlying institutional activity is not yet documented.
8. Many key choices remain embedded in code and prompts rather than elevated into a paper-facing measurement narrative.

### H4. Priority actions

#### Must fix / must clarify

1. Fix and freeze the archival source definition.
2. Fix and freeze the final scoring definition used in the paper.
3. Correct and standardize the deflator naming/provenance.
4. Produce a paper-ready archive coverage and exclusion description.
5. State clearly which variables support headline claims and which are exploratory.

#### Should strengthen

1. Add a manual validation sample for core semantic variables.
2. Quantify layout/era-specific confidence and error patterns.
3. Summarize supervisor fallback / failure handling frequencies if recoverable.
4. Turn the robustness suite into a compact reviewer-facing summary table.
5. Build a variable dictionary that distinguishes observed, inferred, rule-based, and aggregated variables.

#### Optional but helpful

1. Add a pipeline flowchart in the paper appendix.
2. Add a "key researcher choices" appendix.
3. Add one external historical validation exercise tied to a known institutional or legislative shift.
4. Add one appendix table comparing main findings under alternative conservative subsamples.

### H5. Draft paper-section guidance

The eventual Data Preparation / Data Construction section could be organized as follows:

1. **Archival Source and Coverage**
   - archive identity
   - institution and account-book scope
   - time span
   - page inventory and exclusions

2. **Source-to-Data Pipeline**
   - page digitization
   - row extraction
   - multi-model arbitration
   - semantic enrichment
   - year/era aggregation

3. **Variable Construction**
   - directly observed fields
   - inferred semantic variables
   - rule-based fields such as header propagation and year weighting
   - deflated amounts and era definitions

4. **Validation**
   - extraction accuracy on annotated pages
   - inter-model agreement on semantic labels
   - internal consistency checks

5. **Robustness of Main Findings**
   - alternative era cuts
   - arrears treatment
   - confidence thresholds
   - sparse-year exclusions
   - year-weight alternatives
   - deflator alternatives

6. **Limitations**
   - surviving-record bias
   - inference-heavy variables
   - model/prompt dependence
   - partial rerunnability if raw archives are not publicly distributed

---

## Suggested paper-use interpretation

For this project, the strongest paper position is likely:

- not "we contribute a superior historical-ledger extraction framework,"
- but rather "we construct a transparent and validated archival dataset, and we show that the main substantive patterns are robust to reasonable alternative coding and aggregation choices."

Under that positioning:
- extraction scores matter as measurement support,
- reliability and robustness matter more,
- and the most important task is to make the data-construction logic auditable and defensible.

