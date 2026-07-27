# v17 Findings — the decision rule that changed at ~1820

> **The question (Prof. Hu, after v16).** v16 established *when* Oxford's resource-allocation mechanism
> changed — the coupling between earning and spending breaks around **1820**, a generation before the
> 1854 reform and half a century before the spending portfolio visibly transforms (1870) — and stopped
> there: *"What happened around 1820, we do not claim to know."* v17 answers the next question: **what
> organizational decision rule changed?**
>
> Every number traces to `analysis_v17.py`, which runs deterministically and reuses the v15 L1–L4
> classification and cleaning unchanged. We do **not** re-open the break date; v16 settled it robustly.

---

## The short answer

Oxford's budgeting rule changed from **"match spending to this year's receipts"** to **"hold spending
on a planned, smoothed path and let income fluctuate around it."**

- **Before 1820** spending was slaved to income: it closed ~95% of any income–spending gap within a
  year (v16 λ = 0.95), and it was actually *more volatile than income itself* — spending moved first
  and the estate was worked to meet it. This is hand-to-mouth endowment discipline: **spend what the
  land yields, and adjust hard every year.**
- **After 1820** spending comes loose from *total* income (λ = 0.12, indistinguishable from zero) but
  **keeps tracking the stable traditional endowment** (land + church, λ = 0.40, p = 0.04). It stopped
  chasing the volatile *new* streams that were now growing. Because the traditional core is smooth while
  total income turns lumpy, spending ends up *smoother than income*. This is **forward-budgeted
  allocation: spend against the stable core, and let the new money pass through.**

This reconciles v16 and v17 into one mechanism, and rules out two alternatives:

1. It is **not** that spending simply *re-anchored* to the new revenue stream. Spending stopped tracking
   total income but did **not** start tracking the new streams — it fell back on the stable traditional
   core (v16's alternative-income table found the same: land/church λ = 0.39, still significant, after
   1820).
2. It is **not** rigidification of fixed commitments. The standing-charge share of spending *fell*
   (0.54 → 0.38); discretionary latitude *grew*.

The one-line version: **the earliest signal of Oxford's transformation was not a bigger budget for new
things — it was the quiet loosening of the rule tying what it spent to what it earned.**

```
 ~1820   the RULE changes: matching -> smoothing        (this analysis, v17)
  1854   the Oxford University Act                        (institution changes)
  1870   the L3/L4 spending portfolio visibly shifts      (portfolio changes)
```

---

## The data

Annual, inflation-adjusted (Phelps Brown–Hopkins), 1800–1900 (95/101 years observed, 6 interior years
interpolated), plus 1700–1749 as a pre-Industrial-Revolution baseline — identical construction to v16.
Income is additionally decomposed by **economic source**:

- **Traditional estate income** = `land_rent` + `ecclesiastical` (the endowment's core produce).
- **New / liquid income** = all other genuine external income (fees, securities/dividends, banked
  funds, benefactions, administrative receipts).

*(We decompose by source rather than by the v15 L-tag because the L-tag routes ~38% of income into an
"other financial" residual — bankers' receipts, interest, benefactions — that is neither cleanly
traditional nor cleanly new. Source is the defensible cut for the anchor question.)*

Series in `series_v17.csv`; overview figure `mechanism_overview.png`.

---

## Test 1 — Did spending track the traditional estate yield, and did that anchor break at 1820?

The pre-1820 rule, if it is "spend what the estate earns," predicts that spending error-corrects to
**traditional** income specifically. We re-estimate the v16 adjustment speed λ with two anchors — total
income, and traditional estate income alone. `t1_anchor_shift.csv`:

| Era | λ to **total** income | p | λ to **traditional** income | p | new-income share |
|---|---|---|---|---|---|
| 1700–1749 (baseline) | 0.53 | <.001 | 0.44 | <.001 | 0.30 |
| **1800–1819 (tight)** | **0.95** | <.001 | **0.90** | .004 | 0.51 |
| 1820–1853 (loose) | 0.12 | .63 | **0.40** | **.039** | 0.62 |
| 1854–1900 (loose) | 0.18 | .020 | 0.13 | .025 | 0.83 |

**Reading.** Before 1820 spending tracked *both* total income (λ = 0.95) and the traditional estate yield
(λ = 0.90) tightly — the "spend what the land yields" rule is real and datable. At 1820 the tie to
**total** income collapses (λ = 0.12, not distinguishable from zero) **but the tie to the traditional
land/church endowment survives** (λ = 0.40, still significant at p = 0.04). Spending did not switch to the
new streams; it fell back on the *stable* core and let the volatile new money grow untracked. This
reproduces v16's alternative-income finding exactly, and it is the reason spending becomes smoother than
income (Test 2): it is anchored to the smooth core, not the lumpy total. *(STRONG.)*

The new-income share climbs across exactly this window (0.51 → 0.83), so new revenue is clearly arriving
— it just isn't being used as the budget anchor. What that does to spending's volatility is Test 2.

*(Correction: an earlier draft reported the traditional-anchor λ also collapsing (to 0.14) and read this
as "spending stopped anchoring to income of any kind." That was a specification error — the traditional
anchor was estimated while controlling for total-income growth. With the internally consistent control it
is 0.40 and significant, matching v16. The mechanism is anchoring to the stable core, not abandonment of
anchoring.)*

---

## Test 2 — From matching to smoothing (the headline)

If spending stops chasing income, what is it doing instead? Holding a smoother path. Three independent
signatures say so, and they agree.

### (a) Relative volatility flips — `t2_relative_volatility.csv`

The cleanest single number is the ratio of spending volatility to income volatility (sd of annual
log-changes), with a bootstrap 95% CI because the tight era rests on only ~19 points:

| Era | sd Δlog income | sd Δlog spending | **spending / income** | 95% CI |
|---|---|---|---|---|
| 1700–1749 | 0.93 | 0.85 | 0.91 | [0.62, 1.33] |
| **1800–1819 (tight)** | 0.41 | 0.64 | **1.56** — spending swings *more* than income | [0.93, 2.29] |
| **1820–1853 (loose)** | 0.68 | 0.49 | **0.72** — spending now *smoother* | [0.46, 1.11] |
| 1854–1900 (loose) | 1.19 | 0.64 | 0.53 (raw) → **0.92** clean | [0.36, 0.96] |

Before 1820 spending is *more* volatile than income (spending leads and commits, income follows). After
1820 the relationship inverts: income becomes more volatile (the new, lumpier revenue streams) while
spending stays comparatively flat, so the ratio falls through 1.0.

**Two honest corrections from the v17 self-review** (`t2_ratio_robustness.csv`):

1. **The clean, artefact-free flip is 1.56 → 0.72** (1800–1819 vs. 1820–1853). Both eras contain **zero**
   interpolated years and **zero** flagged years — this is the load-bearing comparison.
2. **The raw post-1854 value of 0.53 was overstated.** Two years (1859, 1862) are incomplete-income
   extractions — 1859 records income of ~£32 against £8,353 of spending from just 15 income line-items,
   which is not a real economic event but a missing income side. Excluding them the post-1854 ratio is
   **0.92**, not 0.53. Further smoothing after 1854 is real in *direction* (<1) but marginal in
   *magnitude*; we do not lean on the 0.53.

The per-era bootstrap CIs (1800–1819 [0.93, 2.29] and 1820–1853 [0.46, 1.11]) overlap, but **overlapping
CIs are not a difference test** — a common fallacy. Bootstrapping the *difference* directly
(`t2_flip_significance.csv`) gives 1.56 − 0.72 = 0.84, **95% CI [0.12, 1.65], 99% of resamples positive
→ significant**. So the flip is statistically real, not merely suggestive. (The 15-year rolling-ratio
break test, `t2_smoothing_break.csv`, p = 0.015, is reported as **illustrative only** — overlapping
windows autocorrelate the residuals and understate the p; the difference bootstrap is the clean test.)
Together with the direction flip and v16's confound-tested λ collapse, three signatures agree. *(STRONG.)*

### (b) The direction of precedence flips with it — `t2_direction.csv`

Pre-whitened (differenced) Granger tests, reproducing v16's regime-Granger:

| Era | which side leads | p |
|---|---|---|
| 1800–1819 (tight) | **expenditure → income** | 0.007 |
| 1820–1853 (loose) | **income → expenditure** | 0.011 |
| 1854–1900 (loose) | neither | 0.72 / 0.81 |

In the tight era spending came first and the estate was worked to meet it (commitment-led). After 1820
income comes first and spending responds — the hallmark of a plan that takes expected receipts as an
input rather than a hard yearly constraint. This is the **cleanest leg**: it uses only the 1800–1819 and
1820–1853 windows (neither contains a flagged year) and is robust to lag order (exp→inc p = 0.007/0.010
at lags 1/2; inc→exp p = 0.011/0.075). *(STRONG — independent method, same story, matches v16.)*

**Together (a)+(b)+v16's λ collapse (0.95→0.16, confound-tested) are the mechanism:** spending went from
leading-and-volatile (matching by hard annual adjustment) to following-and-smoothed (planning over the
income stream). No single leg is decisive at ~100 annual points; the three agreeing is what makes it
STRONG. "Predictive precedence, not structural cause" language applies throughout.

---

## Test 3 — The buffer that must absorb the smoothing (honest data limit)

If spending is smoothed while income fluctuates, something must absorb the difference — a reserve drawn
down in lean years and topped up in fat ones. We reconstructed the internal reserve/timing items that
v15/v16 quarantine as accounting artefacts (opening/closing balances, carry-forwards, and income
arrears) and looked for that buffer directly. `t3_gap_magnitude.csv`, `t3_buffer_absorption.csv`.

**We cannot cleanly observe the reserve stock, and we report that rather than dress it up.** Explicit
balance / carried-forward rows are sparse (≈294 across two centuries, a handful per year) and, after
1854, arrears rows nearly disappear from the extracted record (6 of 47 years). The within-year
income–spending gap relative to flow does not show a clean 1820 jump (0.32 → 0.30 → 0.44), and arrears
co-move only weakly and ambiguously with the gap. The buffer is a **necessary accounting counterpart**
of the smoothing we observe in the flows (Test 2), but Oxford's ledger does not record the reserve
densely enough to measure it as a stock. *(SUGGESTIVE / data-limited — stated as a limit, not claimed
as a finding. Identifying the smoothing vehicle is the natural target for archival follow-up.)*

---

## Test 4 — Did spending rigidify toward standing commitments? (No — the opposite)

A competing mechanism: perhaps spending decoupled simply because it filled up with fixed obligations
(salaries, upkeep) that must be paid regardless of receipts. The data reject this. `t4_composition.csv`:

| Era | committed / standing share of spending |
|---|---|
| 1700–1749 | 0.72 |
| 1800–1819 | 0.54 |
| 1820–1853 | 0.47 |
| 1854–1900 | 0.38 |

The standing-charge share **falls** monotonically. Spending moved *toward* discretionary categories, not
away from them — consistent with growing managerial latitude to plan allocation rather than with rigid
commitments crowding out adjustment. *(USEFUL NULL — rules out the fixed-cost story and mildly
reinforces the planning interpretation.)*

---

## The link — the loosening is the precondition for the later L3/L4 transformation

Why does this matter beyond bookkeeping? Because a hand-to-mouth rule — every pound out matched by a
pound in, this year — structurally *cannot* fund a multi-year higher-order (L3/L4) investment. Loosening
the rule is what makes the later transformation affordable. And the sequence in the data is exactly that
of an enabling precondition, not a coincidence. `link_higher_order.csv`:

| Era | higher-order (L3+L4) spend share |
|---|---|
| 1800–1819 (tight) | 0.163 |
| 1820–1853 (loose) | 0.160 |
| 1854–1900 (loose) | 0.247 |

The higher-order share is **flat across 1820** (0.163 → 0.160) and rises only after 1854 — the visible
portfolio shift that v16's change-point analysis independently dates to 1870. **The rule changed first
(1820); the portfolio moved decades later.** The 1820 loosening gives the anchor narrative's previously
SUGGESTIVE claim — "slack/discretion enables higher-order transformation" — a concrete, dated mechanism:
the enabling change is not more money, it is the *release of spending from the annual income constraint*.
*(STRONG as a descriptive sequence; the enabling channel is interpretive, in keeping with our no-causal-
proof stance.)*

---

## What this says to the AI-transformation reader

1. **The first sign that an organization is beginning to transform is not a bigger innovation budget.**
   It is a change in the *rule* linking what it earns to what it is willing to spend. At Oxford that
   loosening ran 30–50 years ahead of any visible change in the spending mix. An organization that waits
   for the transformation budget to move before believing a transformation is underway is reading the
   last signal, not the first.
2. **Fund higher-order (L3/L4) bets by decoupling them from current operating receipts, not by waiting
   for a good year.** The hand-to-mouth rule — "invest in new capability only out of what the existing
   business throws off this quarter" — is precisely the rule Oxford *abandoned* before it could
   transform. Smoothing/forward-budgeting is the enabling move.
3. **New revenue changes budgeting by making income lumpier, which forces the question of how to plan
   over it — not by becoming the new thing you chase.** Oxford's new streams did not become the budget
   anchor; the organizational response was to stop anchoring to current income at all.

---

## Limits, stated plainly

- Annual data, ~100 points; a single-equation error-correction / bivariate frame is what it supports. We
  claim predictive precedence and timing, not proven causation.
- The tight 1800–1819 era rests on ~18–19 usable observations. λ = 0.95 and the volatility ratio 1.56
  are precisely estimated there and survived v16's four confound tests, but it is a short window.
- The reserve/buffer stock (Test 3) is not directly measurable in the extracted ledger; the smoothing is
  inferred from the flows. This is the clearest target for archival follow-up.
- 1750–1799 remains too sparse to use (16/50 years), leaving a gap between the baseline and the modern
  window.
- The rolling smoothing-ratio curve (figure panel B) crosses 1.0 a few years after 1820 because of the
  15-year window's lag; the era-level statistics (1.56 → 0.72 at the 1820 split) are the precise
  statement, the rolling curve the illustration.

---

## Outputs

| File | Contents |
|---|---|
| `series_v17.csv` | full annual panel (income, expenditure, traditional/new income, buffer items, committed/discretionary, higher-order share) |
| `t1_anchor_shift.csv` | λ to total vs. traditional income, by era |
| `t2_relative_volatility.csv`, `t2_rolling_smoothing.csv`, `t2_smoothing_break.csv`, `t2_direction.csv` | the smoothing headline |
| `t3_gap_magnitude.csv`, `t3_buffer_absorption.csv` | buffer reconstruction (data-limited) |
| `t4_composition.csv` | committed vs. discretionary share |
| `link_higher_order.csv` | higher-order spend share by era |
| `strength_assessment_v17.csv` | house-style strength tags |
| `mechanism_overview.png` | four-panel overview |
