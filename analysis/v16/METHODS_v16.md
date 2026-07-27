# v16 Methods: the coupled-system estimation

Companion to `FINDINGS_v16.md`. Documents every modelling choice so the analysis is auditable and
reproducible. Classification of entries into L1–L4 is **unchanged from v15** (`analysis_v15.py`:
`classify_income` / `classify_expenditure`, same cleaning of accounting artefacts and one-off capital).

Reproduce with `cd experiments/reports/analysis_v16 && python analysis_v16.py`. The run is
deterministic (seeded); re-running reproduces every CSV byte-for-byte.

---

## 1. The three systems

"Resources" has three useful meanings, so we model three bivariate systems.

| System | Variables | Question it answers |
|---|---|---|
| **A** | total real income (£), total real expenditure (£) | Does resource *acquisition* drive *deployment*? (the budgeting mechanism) |
| **B** | L4 income share, L4 expenditure share | Does the *transformation* propagate between portfolios? |
| **C** | level-weighted transformation intensity, each side | Same, but using the **full L1–L4 scale**, not just the top level |

System C exists because the professor asked for the transformation portfolios "using the finalized
L1–L4 definitions", and B uses only L4. C is a level-weighted index on a 1–4 scale:

```
intensity_t = Σ_i w(level_i) · amount_i  /  Σ_i amount_i        over classified entries
w(income)      = {L1: 1, L3: 3, L4: 4}
w(expenditure) = {L1: 1, L2: 2, L3: 3, L4A: 4, L4B: 4}
```

"other"/unclassified entries are **excluded** rather than given a weight, so the index cannot be moved
by the unclassified bucket. 1 = purely core/standing activity, 4 = purely higher-order.

## 2. Sample windows and gaps

- **Modern window 1800–1900**: 95 of 101 years observed; 6 interior years (1818, 1874–75, 1887–88,
  1890) linearly interpolated to give the regular annual index a VAR requires. Every headline is
  re-run on **observed years only** and holds (`confounds.csv`).
- **Pre-Industrial-Revolution baseline 1700–1749**: 48 of 50 years observed, ~138 rows/yr. It is
  tempting to write off the whole eighteenth century as too sparse, but that is only true of
  **1750–99** (16 of 50 years). 1700–49 is nearly complete, and it is the natural "before the
  revolution" comparison, so we use it.
- 1750–99 remains unusable, so there is an acknowledged gap between the two windows.

## 3. Stationarity and transformation

All series are **I(1)** (ADF cannot reject a unit root in levels; decisively rejects after one
difference), so the baseline VARs are estimated on **log-growth** (System A) and **first differences**
(B and C). `lag_selection.csv` reports ADF p-values in levels and differences alongside AIC/BIC/HQIC.

## 4. Model 1: baseline VAR

- **Lag order** by BIC (parsimonious; AIC and HQIC reported alongside): A = 2, B = 1, C = 1.
- **Coefficients** with p-values in `var_coefs.csv`.
- **Granger causality**: F-test, both directions, full sample (`granger.csv`).
- **Impulse responses**: orthogonalised (Cholesky), income/revenue ordered first, 10-year horizon,
  cumulative, with 95% bands. **All four responses** (both own- and cross-) are plotted per system
  (`irf_A.png`, `irf_B.png`, `irf_C.png`). Ordering income-first reflects the prior that earning is
  logically upstream of spending; nothing hinges on it, because the full-sample responses are ~zero.

## 5. Model 2: regime shift

### 5.1 Why an error-correction model, and how it is justified

A college cannot outspend its income indefinitely, nor hoard indefinitely, so earning and spending are
tied in the long run and the spending equation needs an error-correction term. We **impose** the
cointegrating vector at (1, −1), the budget gap itself, rather than estimating it. Two consequences:

1. A plain **ADF on the imposed gap** is a valid test of whether the term belongs (standard critical
   values apply precisely *because* the vector is not estimated, so no Engle–Granger correction is needed).
   It gives stat −8.16, p < 0.0001. The term is licensed.
2. The **Johansen** test is reported alongside and **disagrees**: it rejects rank ≤ 0 *and* rank ≤ 1,
   and in a bivariate system rejecting rank ≤ 1 implies **full rank**: both series stationary in
   levels, i.e. *not* cointegrated. **Note the trap:** the second rejection reads at a glance as extra
   support for cointegration, when in a two-variable system it means the opposite. ADF on the levels is
   borderline (p = 0.05–0.07), so "near-stationary levels" is the honest description. We rely on the
   imposed-gap ADF, which is the test that speaks to the specification we actually use, and we disclose
   the disagreement (`cointegration.csv`).

### 5.2 The specification

```
Δexp_t = c + b₁·Δinc_{t−1} + b₂·Δexp_{t−1} + λ·(inc − exp)_{t−1} + ε_t
```

estimated by OLS with **HAC (Newey–West, 2 lags)** errors. **λ is the adjustment speed**: the share of
last year's income–spending gap that spending closes within a year. λ ≈ 1 = spending snaps straight
back to what was earned; λ ≈ 0 = spending is free of last year's earnings. `adjustment_speed.csv`
reports λ by period; `chow.csv` reports the break tests.

### 5.3 Break tests are **fully interacted**

At each candidate break, *every* term is interacted with the regime dummy, not just λ:

```
Δexp_t = … + γ·R + δ₁·(R × Δinc_{t−1}) + δ₂·(R × Δexp_{t−1}) + δ₃·(R × gap_{t−1})
```

We report δ₃ (the change in λ) **and** the joint Chow test that all four break terms are zero. Allowing
only λ to shift while holding the short-run terms fixed would overstate the evidence: under that
restricted test the 1854 break looks sharp (p = 0.005), but under the full interaction it is marginal
(p = 0.0495). The 1820 break is strong under both (p = 0.0008; joint p < 0.0001). We report the full
interaction.

**Candidate breaks are not assumed.** We test 1820 (what the data chooses, Model 3), 1854 (the Oxford
University Act, the hypothesis) and 1870 (the level break), and let the numbers rank them.

### 5.4 Confound tests (`confounds.csv`)

The headline is a *fall* in an estimated coefficient, and **noise in a regressor attenuates its
coefficient toward zero**. Income is recorded far more coarsely after 1854 (80 → 47 line-items/yr,
while expenditure stays flat at ~122 → ~127) and the standard deviation of income log-growth doubles.
This could manufacture a fake collapse. It is tested, not caveated:

- **Alignment.** Recording density is *flat* across the 1820 break (83 → 78 items/yr) while λ collapses
  (0.95 → 0.16). The density drop happens at **1854**, where λ barely moves (0.16 → 0.19). The confound
  and the finding do not line up.
- **Thinning placebo.** Randomly thin the 1800–19 income line-items down to the sparse era's density,
  rebuild the annual income series from the thinned records, re-estimate λ; 200 draws. λ(1800–19) goes
  from 0.95 to **0.76 [0.50, 1.13]**, attenuated, as theory predicts, but **0 of 200 draws** reach the
  loose-era value of 0.19.
- **Deflator.** The ~1820 break sits on the post-Napoleonic price collapse. Re-estimated in **nominal**
  £ with no price index at all: λ = 0.92 / 0.15 / 0.18 (real: 0.95 / 0.16 / 0.19). Not the deflator.
  *(Note the budget gap is deflator-invariant by construction: both sides are divided by the same
  index, but the dependent variable is not, so the test is not vacuous.)*
- **Interpolation.** Observed years only: λ = 1.02 / 0.16 / 0.20.
- **An IV correction was attempted and discarded.** Instrumenting the year t−1 regressors with their
  t−2 counterparts is the textbook fix for this errors-in-variables problem (recording errors two years
  apart are independent). But the **first-stage F is 2.7 (pre) and 0.6 (post)**: hopelessly weak,
  because the budget gap is strongly mean-reverting and its own lag carries almost no signal. We report
  the first-stage F and the fact that we discarded it, rather than reporting a weak-instrument estimate.

## 6. Model 3: data-driven regime detection

**What we search for, and why it matters.** Bai–Perron tests for breaks in **regression coefficients**.
Running a change-point detector on the **mean of a series** is a different and easier question: it cannot
see a change in how two series *relate*, which is exactly what Model 2 claims changed. So Model 3
searches for breaks in the coupling itself, three ways (and keeps the level breaks clearly labelled as a
separate object):

- **Bai–Perron family, least-squares breaks in the ECM regression** (`ruptures`, **linear** cost on the
  design matrix `[y | 1, Δinc_{t−1}, Δexp_{t−1}, gap_{t−1}]`, min segment 15). Dynamic programming with
  1 break returns **1822**. PELT at penalty 12 returns none (the penalty is conservative for a single
  break; reported as-is rather than tuned until it agrees).
- **Trimmed sup-Wald / Quandt–Andrews scan.** For every candidate year in the central 70% of the sample
  (15% trimming), re-estimate the interacted model and record the Wald statistic on the change in λ.
  Compared against **Andrews (1993), 5%, 1 parameter, 15% trim = 8.85**, the critical value that
  accounts for having searched over dates. Result: **argmax 1819, Wald 16.8 (significant)**;
  **Wald@1854 = 8.0, below the critical value**; Wald@1870 = 0.9. Full curve in `supwald_scan.csv` and
  the left panel of `model3_breaks.png`. The peak is *interior*, not a trim-boundary artefact: the
  statistic rises from 8.2 (1817) to 15.0 (1818) to 16.8 (1819), then falls away (14.5, 11.5, 11.9).
- **Markov-switching regression** with a switching λ (`markov_states.csv`). Reported as an **honest
  negative**: it finds a tight state (λ = 0.67) and a loose one (λ = 0.08), but the states last only
  1.5–2.5 years and the tight state covers 15% of pre-1854 years vs 19% of post-1854, it fragments
  into short bursts rather than recovering a persistent historical phase.
  *On switching variance:* we wanted it on, so that genuinely looser coupling could be separated from
  merely noisier data. At n ≈ 100 that likelihood **does not identify**, across seeds it returns
  log-likelihoods from −65 to −68.5, sometimes lands on a degenerate zero-variance regime (fitting a
  handful of points exactly and reporting a nonsense λ), and sometimes fails to construct steady-state
  probabilities. We therefore report the **common-variance** fit, which is exactly reproducible across
  seeds (llf = −73.61), and state the limitation. The density confound is handled by the thinning
  placebo, which does not depend on any of this.
- **Level breaks kept, but relabelled.** Change-point detection on the *level* of the L4 shares returns
  **1870** (and 1870 & 1880 with two breaks forced). This is retained, it is a real finding, but it
  is labelled for what it is: a break in the **level of the transformation mix**, a different object at
  a different date from the break in the coupling.

## 7. What is deliberately **not** claimed

- No structural/causal identification. Granger = predictive precedence; we report precedence,
  adjustment speed, and timing only.
- No full VECM system: we use a **single-equation** error-correction model for the spending equation
  (the quantity of interest is how spending adjusts to the gap), not a two-equation VECM.
- **We do not claim to know what happened around 1820.** It coincides with the post-Napoleonic
  agricultural depression and the resumption of cash payments; we have ruled out the deflator, the
  interpolation, and the recording density. Identifying the cause is the next question, not one this
  analysis settles.

## Outputs

`series.csv`, `density.csv`, `lag_selection.csv`, `var_coefs.csv`, `granger.csv`, `cointegration.csv`,
`adjustment_speed.csv`, `chow.csv`, `confounds.csv`, `regime_granger.csv`, `robustness.csv`,
`comovement.csv`, `breaks.csv`, `supwald_scan.csv`, `markov_states.csv`;
figures `model1_overview.png`, `irf_A.png`, `irf_B.png`, `irf_C.png`, `model2_regime.png`,
`model3_breaks.png`, `timeline.png`.
