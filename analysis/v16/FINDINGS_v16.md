# v16 Findings: income and expenditure as a coupled dynamic system

> **The question.** How did the relationship between resource generation (income) and resource
> deployment (expenditure) evolve through the Industrial Revolution? We treat the two as one coupled
> system and estimate the three requested models.
>
> Every number traces to `analysis_v16.py`, which runs deterministically. Classification into L1–L4 is
> unchanged from v15.
>
> **Language.** Granger causality is *predictive precedence* (does the past of one series help forecast
> the other), not structural cause. We report it as prediction throughout.

---

## The short answer

The two sides **are** dynamically linked, and the link **does** change fundamentally across historical
periods, but it changes at a date nobody assigned it. The coupling between earning and spending breaks
down around **1820**: a generation *before* the Reform Acts, and half a century before the spending
portfolio visibly transforms.

```
 ~1820   the COUPLING between earning and spending breaks     (the mechanism changes)
  1854   the Oxford University Act                            (the institution changes)
  1870   the L4 transformation mix shifts                     (the portfolio visibly changes)
```

The mechanism changes first. The institution reforms second. The visible transformation arrives last.

---

## The data and the three systems

Annual series over **1800–1900** (95 of 101 years observed; 6 interior years interpolated), plus
**1700–1749** (48 of 50 years) as a **pre-Industrial-Revolution baseline**. All money is
inflation-adjusted.

| System | The two series | The question it answers |
|---|---|---|
| **A** | total real income vs total real expenditure | Does resource *acquisition* drive *deployment*? (the budgeting question) |
| **B** | L4 income share vs L4 expenditure share | Does the *transformation* spread between portfolios? |
| **C** | transformation-intensity index on each side | The same, using the **full L1–L4 scale**, not just the top level |

System C weights each entry by its level (L1 = 1 … L4 = 4) and averages, giving one number between 1
and 4 per year per side. It exists so the L1–L4 definitions are used in full.

---

## Model 1: Baseline dynamic interaction

**What we did.** A bivariate VAR on each system: this year's income depends on past income *and* past
expenditure, and vice versa. Lag order by BIC (A = 2, B = 1, C = 1). Granger tests in both directions;
impulse responses traced over 10 years.

| System | Direction | F | p | Helps predict? |
|---|---|---|---|---|
| A | income → expenditure | 1.51 | 0.224 | no |
| A | expenditure → income | 0.64 | 0.530 | no |
| B | L4 income → L4 spend | 0.18 | 0.669 | no |
| B | L4 spend → L4 income | 0.44 | 0.510 | no |
| C | income intensity → spend intensity | 0.08 | 0.784 | no |
| C | spend intensity → income intensity | 0.11 | 0.735 | no |

**What it means.** Over the whole century, **neither side predicts the other, in any system**, and the
impulse responses are small and fade within a few years. But a century-long average of zero is exactly
what you get when the relationship is strong in some periods and absent in others and the two cancel
out. The flat result is itself the first evidence that the relationship is **not stable over time**.
That is what Model 2 tests.

(Coefficients: `var_coefs.csv`. All four impulse responses per system: `irf_A/B/C.png`.)

---

## Model 2: Historical regime shift

**What we did.** A college cannot outspend its income indefinitely, nor hoard indefinitely. So the
quantity of interest is the **adjustment speed**: when income and spending drift apart, what share of
that gap does spending close within a year?

- **Near 1**: spending snaps straight back to what was earned. The classic endowment discipline:
  *spend what the land yields*.
- **Near 0**: spending has come loose from last year's earnings.

Formally this is an error-correction model: Δexp\_t regressed on past Δincome, past Δexpenditure, and
last year's income–expenditure gap, with HAC standard errors. The coefficient on the gap is the
adjustment speed, λ. We estimate λ period by period, and test for a break at *each candidate date*
rather than assuming the reform is the right split.

*(Specification note: we impose the cointegrating vector at (1, −1), the budget gap itself, rather than
estimating it. An ADF on that imposed gap decisively rejects a unit root (−8.16, p < 0.0001), which
licenses the error-correction term; because the vector is imposed and not estimated, standard ADF
critical values apply. The Johansen test disagrees: it rejects rank ≤ 0 **and** rank ≤ 1, and in a
bivariate system rejecting rank ≤ 1 implies full rank, i.e. levels stationary and *not* cointegrated.
ADF on the levels is borderline (p = 0.05–0.07), so "near-stationary levels" is the honest description.
We rely on the imposed-gap ADF, the test that speaks to the specification actually used, and disclose
the disagreement in `cointegration.csv`.)*

### The core result

| Period | Adjustment speed λ | p | Reading |
|---|---|---|---|
| 1700–1749 (pre-Industrial-Revolution) | 0.463 | 0.0002 | tight |
| 1800–1819 | **0.948** | <0.0001 | tightest |
| **1820–1853** | **0.159** | **0.44** | **already loose: still before the reform** |
| 1854–1900 (post-reform) | 0.185 | 0.017 | loose |

Before 1820, spending closed almost the whole of any income gap within a year. From 1820 it closes
about a sixth, and is no longer statistically distinguishable from zero. **By the time the 1854 reform
arrives, the coupling has already gone.**

### Break tests (fully interacted: *every* term may shift, not just λ)

| Break | What the date is | Δλ | p | joint Chow p |
|---|---|---|---|---|
| **1820** | chosen by the data (Model 3) | **−0.764** | **0.0008** | **<0.0001** |
| 1854 | the Oxford University Act | −0.374 | 0.0495 | 0.047 |
| 1870 | break in the *level* of the transformation mix | −0.253 | 0.55 | 0.84 |

The 1820 break is an order of magnitude stronger. The apparent effect at 1854 is marginal and appears
only because 1854 lies *downstream* of the real break: split anywhere after 1820 and the loose era
lands on the right-hand side.

### The direction of influence flips

| Era | Which side leads | p |
|---|---|---|
| 1800–1819 (tight) | **expenditure → income** | 0.004 |
| 1820–1853 (loose) | **income → expenditure** | 0.009 |
| 1854–1900 | neither | 0.72 / 0.81 |

In the tight era, spending came first and income followed. The college committed to its obligations and
the estate was worked to meet them. After 1820 this reverses: income leads and spending responds. After
1854 the predictive link disappears in both directions.

### Four checks that this is not a data artefact

The most serious threat is dull but important: **income is recorded far more coarsely later on** (≈80
income line-items/yr before 1854, 47 after; expenditure line-items stay flat at ≈122 → ≈127). Noise in a
regressor attenuates its coefficient toward zero, so a "collapse" in λ could in principle be nothing but
sloppier bookkeeping. Four tests (`confounds.csv`):

| Worry | Test | Result |
|---|---|---|
| Coarser records fake the collapse | Does the density change *line up* with the break? | **No.** Density is flat across the 1820 break (83 → 78 items/yr) where λ collapses 0.95 → 0.16. The density drop happens at **1854**, where λ barely moves (0.16 → 0.19). |
| Coarser records fake the collapse | **Placebo:** randomly discard early income records until the early period is as sparse as the late one, re-estimate; 200 draws | λ(1800–19) falls only from 0.95 to **0.76 [0.50, 1.13]**. **0 of 200 draws** reach the loose-era value of 0.19. |
| 1820 = the post-Napoleonic deflation, an inflation-adjustment artefact? | Re-estimate on **nominal £**, no price index at all | λ = 0.92 / 0.15 / 0.18 (vs real 0.95 / 0.16 / 0.19) |
| The 6 interpolated years create the break | Re-estimate on observed years only | λ = 1.02 / 0.16 / 0.20 |

The finding survives all four.

*(An IV correction using lag-2 instruments was attempted and discarded: first-stage F = 2.7 and 0.6,
far below the usual threshold of 10. The budget gap is strongly mean-reverting, so its own lag carries
almost no signal. We report this rather than a weak-instrument estimate.)*

---

## Model 3: Data-driven regime detection

**What we did.** So far we chose the dates. The stronger test is to tell the model nothing about history.
Three approaches, all searching for a change **in the relationship between the two series**, not in the
level of either one, which is a different and easier question:

- **Bai–Perron structural break** on the error-correction regression (least-squares breaks in the
  regression *coefficients*).
- **Exhaustive break search (sup-Wald / Quandt–Andrews).** Try every candidate year, re-estimate, record
  the strength of evidence, then compare against a threshold (Andrews 5% = 8.85) that already accounts
  for having searched over many dates.
- **Markov-switching model.** Assume two hidden states with different coupling strengths and let the
  model infer which years belong to which.

| What we search | Method | Result |
|---|---|---|
| **the coupling** | Bai–Perron structural break | **1822** |
| **the coupling** | exhaustive search over every year | **argmax 1819, statistic 16.8: significant.** **1854 scores 8.0: below the 8.85 threshold.** 1870 scores 0.9 |
| the coupling | Markov-switching (hidden states) | *no lasting periods: see below* |
| the *level* of the L4 mix | change-point detection | 1870 (and 1870, 1880 with two breaks forced) |

**Given no historical information, the data lands on 1819–1822 and declines to select 1854.** Two
independent methods agree on the early date. Once we properly account for having searched across
candidate years, the evidence for a break at the reform year **does not reach significance at all**.

The last row is a *different question* with a different answer: the **level** of higher-order spending
clearly shifts around **1870**, that is when the transformation becomes visible in the accounts. But
the **mechanism** linking earning to spending had already changed fifty years earlier.

### One model that did not work, reported as such

The **Markov-switching model failed to recover any historical periods.** It does identify a tight state
(λ = 0.67) and a loose one (λ = 0.08), but assigns them in bursts of 1.5–2.5 years that flicker across
the whole century, the tight state covers 15% of pre-1854 years and 19% of post-1854 years, i.e. it
found nothing historical. With ~100 annual observations, a model inferring hidden states, transition
probabilities and coefficients simultaneously has too little to work with.

*(We wanted a switching-variance version, so that genuinely looser coupling could be distinguished from
merely noisier data. At this sample size that likelihood does not identify: across seeds it returns
log-likelihoods from −65 to −68.5, sometimes lands on a degenerate zero-variance regime, and sometimes
fails outright. We report the stable common-variance fit and state the limit.)*

This does not weaken the finding, the two methods suited to this sample size both land on ~1820, but
it is an honest limit on the "let the data decide" approach, and we report it rather than dropping the
model that disagreed.

---

## Interpretation, the three questions

**1. Does new revenue enable later transformation investment?**
**No reliable lead.** In both the L4 mix (System B) and the full L1–L4 intensity index (System C), new
revenue and higher-order spending move *together*, with neither reliably ahead of the other in any era.
*(This refines v15: the apparent ~6-year lead came from comparing two trending lines. Once the shared
trend is removed by differencing, the head start is not statistically real.)*

**2. Does higher-order investment generate a later change in the revenue model?**
**No reliable lead in that direction either.** The relationship between the two portfolios is one of
simultaneous movement, not lead-and-follow.

**3. Does the interaction become fundamentally different across phases?**
**Yes, this is the central result, but the phases are not the ones institutional history would
suggest.** The coupling is tight through the eighteenth century (λ = 0.46), tightest in 1800–19
(λ = 0.95), and breaks around 1820 (λ = 0.16, no longer distinguishable from zero). The *direction* of
influence flips at the same moment, and after 1854 the link dissolves entirely. **Oxford's
resource-allocation mechanism did not change *because* of the Reform Acts. It had already changed, a
generation before them.**

### What this means for the AI framing

Oxford did not merely change *what* it earned and spent on. It changed **the rule connecting the two**, and it changed that rule long before the institutional reform, and half a century before the shift
became visible in its portfolio.

- **The old rule was: spend what the core business yields.** Deployment was pulled straight back to
  whatever the endowment produced, closing nearly the whole gap within a year. An organisation that
  funds new capability only out of what the existing business throws off is running exactly this rule.
- **The rule loosened first, and quietly.** Nothing in the levels announced it; it is visible only in
  the *coupling*.
- **The visible transformation came last**: fifty years after the mechanism changed. An organisation
  that waits for the budget to visibly move before believing a transformation is underway is reading the
  last signal, not the first.

The earliest observable signal of a resource reallocation is not the size of the transformation budget.
It is the loosening of the link between what an organisation earns and what it is willing to spend.

---

## Limits, stated plainly

- Annual data, ~100 points in the main window; a bivariate system is all it can support. We claim
  prediction and timing, not proven causation.
- The tight 1800–19 era rests on only 18 usable observations. λ = 0.95 there is precisely estimated
  (p < 0.0001) and survives every robustness re-run, but it is a short window.
- The break-search peak (1819) is an **interior** peak, not a boundary artefact, the statistic rises
  from 8.2 (1817) to 15.0 (1818) to **16.8 (1819)** and then falls away (14.5, 11.5, 11.9), and the
  Bai–Perron break lands independently at 1822. We date the break as "~1820", not to the year.
- **What happened around 1820, we do not claim to know.** It coincides with the post-Napoleonic
  agricultural depression and the return to the gold standard. We have ruled out the inflation
  adjustment, the interpolated years, and the recording density. Identifying the cause is the natural
  next question, not something this analysis settles.
- 1750–99 remains unusable (16 of 50 years have records), leaving a gap between the
  pre-Industrial-Revolution baseline and the main window.
- The "other/unclassified" bucket noted in v15 is unchanged; it does not enter these dynamics.
