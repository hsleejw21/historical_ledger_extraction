# Four-Level Variable Dictionary (v12 — locked)

> **This is the operational definition the paper commits to.** Decided 2026-06-13 with
> Jungwoo, grounded in Ethan's transformation-depth report (`Oxford_Arxiv_Report0608.pdf`,
> Table 3) and the existing v6 construction (`experiments/analysis/analysis_v6.py:143`).
> Decision rule: **single-variable expenditure-share proxies are the MAIN measures;
> everything richer is a separate ROBUSTNESS layer** (keeps main vs exploratory clean, and
> defensible under the LLM-enrichment scrutiny the paper will attract).

---

## The framework as *dimensions of transformation depth* (not stages)

Following Ethan's reframe and the professor's guidance, the four levels are **analytically
distinct depths**, not a deterministic L1→L2→L3→L4 ladder. The general construct (depth)
travels across eras; the AI-era label and the Oxford proxy are era-specific manifestations.

| Level | General construct (depth) | AI-era manifestation | Oxford proxy |
|-------|---------------------------|----------------------|--------------|
| **L1** | Task / function substitution | Automation | Legacy-function recomposition |
| **L2** | Process adaptation | Personalization / process digitalization | Payment / process regularization |
| **L3** | Operating-model reconfiguration | Operational innovation | Capability / role-system build-up |
| **L4** | Mission / value redefinition | Business-model innovation | Educational mission shift |

---

## MAIN measures (locked — all results rest on these)

All proxies are **shares of total real expenditure within a year**. Because the
price deflator is a single within-year multiplier across all categories, these shares are
**deflation-invariant** (nominal share = real share), which makes them robust to the
price-index choice.

| Level | Definition | Source field(s) | Direction |
|-------|------------|-----------------|-----------|
| **L1** | `1 − traditional_function_share`, where traditional = **{ecclesiastical, maintenance, domestic}** / total expenditure | `category`, amount | expenditure |
| **L2** | **payment-modernity index** — amount-weighted mean of payment-period scores (annual=1.0 … multi-year/irregular→low) | `payment_period`, amount | expenditure |
| **L3** | **salary_stipend** / total expenditure | `category`, amount | expenditure |
| **L4** | **educational** / total expenditure | `category`, amount | expenditure |

**Enablers / context variables (NOT levels):**
- **Resource slack:** financial slack (income − expenditure), real **land-rent income**,
  revenue diversification (**HHI / income_div**). Land-rent is **income-side and sits in
  the slack mechanism, not in L1** (resolves Ethan's open question — the v6 code already
  treats it this way).
- **Institutional shocks:** 1854 (Oxford University Act) and 1877 (Universities Act)
  reform windows, treated as **strategic discontinuities**.

### Per-level measurement caveats (carry into the paper's limitations)

- **L1** — not literal automation; depends on the legacy-category boundary → *settled by
  the robustness test below.*
- **L2** — payment-period reliability is weaker early in the series; may read high pre-1750.
- **L3** — salary/stipend is narrower than the full operating-model-reconfiguration concept.
- **L4** — strongest single proxy, but *spending alone does not prove intentional mission
  strategy* → *addressed by textual triangulation below.*

---

## ROBUSTNESS layer (separate; never replaces a MAIN measure)

| Target | Robustness check | Data (already exists) | Purpose |
|--------|------------------|------------------------|---------|
| **L1 boundary** | Recompute L1 with traditional = +charitable, and +charitable+administrative; report whether era ordering / conclusions move | enriched `category` | **Fix the boundary empirically** (Ethan: "must be fixed before paper-facing estimates") |
| **L4 mission** | Triangulate educational-spend share against textual mission signals: scholarships, prizes, exams, teaching, scientific vocabulary | `analysis_v4/scholarship_prize_trajectory.csv`, `innovation_vocabulary_yearly.csv`, `first_appearances.csv` | Show mission redefinition is real, not an accounting artifact |
| **L2 (future)** | Auxiliary process indicators: Latin→English language share, administrative regularization | `analysis_v4/calendar_language_decades.csv` | Corroborate L2 as background infrastructure |
| **L3 (future)** | Broaden with administrative expenditure + role vocabulary + payee structure | `analysis_v4/person_role_evolution_yearly.csv`, `payee_type_decades.csv` | Widen the capability proxy |
| **Slack (future)** | Reform-conditioned test: does pre-shock slack predict post-shock L3/L4 jump size? | `four_level_proxies.csv` (land_rent_share, income_div) | Strengthen / honestly bury the slack mechanism |

The first two rows (L1 boundary, L4 triangulation) are **built now** in
`variable_validation.py`; the rest are scoped for later and assemble existing v4 outputs.

---

## Status of the L1 boundary and L4 triangulation

> Settled by `variable_validation.py` → outputs in `reports/analysis_v12/`
> (`l1_boundary_sensitivity.csv/.png`, `l4_textual_triangulation.csv/.png`,
> `variable_validation_verdict.txt`). Computation validated against the v6 panel
> (|r| = 0.997 vs v6 traditional share).

- **L1 boundary verdict — LOCKED: `1 − {ecclesiastical, maintenance, domestic}`.**
  Adding **charitable is immaterial** (mean era Δ = 0.012, r = 0.99 vs base) — it may be
  included or not without changing any conclusion. Adding **administrative is material**
  (mean era Δ = 0.153) and administrative is *not* a legacy/traditional function, so it is
  **excluded** — including it would mechanically depress L1. The boundary is therefore
  fixed and defensible (resolves Ethan's open item).

- **L4 triangulation verdict — CORROBORATED at the trend level.** The educational-spend
  share (MAIN L4) co-moves with independent textual mission signals: composite trend
  (10-yr) r = **0.75**; `scholarships_prizes` r = **0.91**, `scientific_professorships`
  r = **0.83**, `lectures_teaching` r = **0.54**. So L4 reflects a **real mission shift,
  not an accounting artifact** — exactly the corroboration Ethan/the professor asked for.
  Sparse early-series indicators (`competitive_examinations`, `prize_intensity`) are too
  thin to be informative and are excluded from the triangulation. Levels-only composite
  r = 0.45 (p ≈ 3e-11) is depressed by those sparse signals and by year-to-year share
  noise; the slow-moving share is best assessed at the trend level.
