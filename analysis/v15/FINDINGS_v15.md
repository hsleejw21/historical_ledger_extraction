# v15 — Findings: the transformation as a resource-allocation problem

> Oxford lets us watch both sides of the allocation: how it **generated** resources (income) and how
> it **deployed** them (expenditure). Construct rules are fixed in `CODING_PROTOCOL_v15.md`; every
> number traces to `analysis_v15.py`. Income is cleaned first (opening balances, arrears,
> carry-forwards, aggregate totals and one-off asset sales removed; 75% of income retained).

---

## 1. Revenue portfolio — how Oxford generated resources

Income share by level, by era (%):

| Era | L1 traditional (land, church) | L4 new business (fees, investment) | L3 admin | other |
|---|---|---|---|---|
| 1700–1779 | 66.7 | 7.3 | 1.6 | 24.4 |
| 1780–1819 | 48.8 | 2.9 | 3.2 | 45.1 |
| 1820–1859 | 33.6 | 3.4 | 2.7 | 60.3 |
| **1860–1900** | **18.6** | **42.7** | 7.2 | 31.5 |

The revenue model changed only in the last era, and it changed sharply. Traditional land-and-church
income falls steadily from **66% to 18%**. New-business income (fees + investment) sits near **3–7%**
for a century and a half, then jumps to **42.5%** after 1860. L2 income is essentially nil —
personalisation is a way of spending, not earning.

**Timing of the new revenue.** Both new sources exist as occasional early entries (a securities line
in 1709, a fee line in 1700), but they become a *sustained* model only late: the emergence
change-point is **1870 for fees** and **1889 for investment income**. New revenue is a phenomenon of
the reform decades, not the eighteenth century.

**On the "other" bucket.** With current enrichment categories, ~a third of income (more mid-century)
is miscellaneous receipts that cannot be attributed to a level. Reading only the *classifiable*
income (`revenue_classified.csv`), the shift is even clearer: new-business income rises from ~**10%
to 62%** of classifiable income, traditional falls from ~88% to 29%. The pivot does not depend on how
the residual is treated.

**Answer:** Oxford shifted from living on its endowment to earning from a new activity — selling
education for fees — within a single generation.

---

## 2. Investment portfolio — how Oxford deployed resources

Expenditure share by level, by era (%):

| Era | L1 upkeep+personnel | L2 awards | L3 admin | L4 mission+business | other |
|---|---|---|---|---|---|
| 1700–1779 | 67.1 | 2.0 | 14.7 | 3.7 | 12.4 |
| 1780–1819 | 30.6 | 24.3 | 14.1 | 2.0 | 29.0 |
| 1820–1859 | 33.1 | 13.9 | 14.6 | 0.9 | 37.5 |
| **1860–1900** | 37.8 | 9.2 | 15.4 | **10.0** | 27.6 |

Deployment shifts too, but **much more modestly than revenue**. Higher-order spending (L3 + L4) is
**18% → 16% → 15% → 25%** — a real late rise, driven by **L4 rising from ~1% to 10%** (change-point
1860). But the college still spends most of its money the old way: upkeep + personnel (L1) stays the
largest block (~38%), and administration (L3) is stable at ~15% throughout — maintained, not re-grown.

**Answer:** Oxford shifted spending toward higher-order transformation, but modestly — most of the
budget still ran the existing operation.

> **Correction from the first draft.** An earlier version counted the *whole* `financial` spending
> category as L4 investment, inflating L4 deployment to ~30% by including accounting artefacts
> (aggregate totals, carried-forward balances, losses). Restricting L4A to genuine investment
> purchases and cleaning artefacts on both sides brings it to ~10%. The revenue figures are unaffected.

---

## 3. Resource reallocation — where did each additional pound go?

Comparing spending per year, early (1780–1853) vs late (1854–1900):

| Level | Share of the marginal pound | Relative share change (pp) |
|---|---|---|
| L1 upkeep+personnel | 38.0% | +3.8 |
| **L4 mission+business** | 25.7% | **+16.8** |
| other | 21.9% | −8.4 |
| L3 admin | 7.8% | −4.5 |
| L2 awards | 6.6% | −7.8 |

The largest share of new spending simply scaled the old operation: **38 pence of each additional
pound went to L1** (upkeep + personnel). But the next-largest share, **26 pence, went to L4** — and
from an almost-zero base, so L4 gained by far the most in *relative* terms (**+16.8 points**). L2, L3
and "other" all lost relative ground.

**Answer:** most of the marginal pound scaled the existing operation, but the fastest-growing
claimant was higher-order transformation.

---

## 4. Connecting the two portfolios (association only, no causal claims)

Early (1780–1853) vs late (1854–1900):

| Measure | Early | Late |
|---|---|---|
| New-business income share | 2.5% | 37.9% |
| Higher-order spend share (L3+L4) | 16.1% | 23.6% |
| Traditional land income share | 41.2% | 17.5% |
| Income diversity (effective # of sources) | 1.99 | 2.08 |

Observations, all descriptive:

1. **The transformation was lop-sided.** Revenue moved a great deal (2.5% → 38% new business) while
   deployment moved much less (16% → 24% higher-order). Oxford changed *how it earned* far more than
   *how it spent* — though both moved in the same late window (the 1860s–70s).
2. **Business model appears just before mission.** Business-model revenue (L4A income) leads mission
   spending (L4B expenditure) by about **6 years** in the cross-correlation — the same order v14
   found, now visible on both sides of the ledger.
2b. **New revenue did not build operational innovation.** L3 (administration) was already in place
   and steady at ~14% of spending long before the new revenue arrived, so new revenue did not
   finance the operating model — it had been built earlier. New revenue coincides with L4, not L3.
3. **Traditional income did not fund the expansion.** Land income *fell* (41% → 17%) exactly as
   higher-order spending rose; the new spending coincides with the new revenue, not with the old.
4. **This was substitution, not diversification.** The effective number of income sources stayed
   near **2** throughout: Oxford swapped one dominant source (land) for another (fees), rather than
   spreading its income across many. The revenue base was *renewed*, not *widened*.

**A limit we state plainly.** We do not pin down which side moved first year-by-year: both change in
the same short window and the series are close (indeed the L4-spending change-point, 1860, is a touch
earlier than the L4-income one, 1870, so a naive reading could even say spending moved first). The one
ordering we rely on is **new-business revenue rising just ahead of mission spending (L4A→L4B)**. The
headline is the **size** gap, not the timing: revenue changed far more than spending.

---

## 4b. Formal dating of the pivot (`dating.csv`)

Change-point tests on the aggregate L4 shares confirm the late pivot and date it:

- **L4 income share** breaks at **1870** (Welch *t* = 14.0 — a very sharp break).
- **L4 expenditure share** breaks at **1860** (*t* = 4.9).

(The interrupted-time-series level shifts are unstable here because the pre-1854 baseline is near
zero, so the *change-point* is the reliable statistic; both put the shift firmly in the 1860s–70s.)

---

## 4c. Robustness — do the classification choices matter? (`sensitivity.csv`)

We re-computed the three headline numbers under alternative choices:

Late window = 1860–1900, average yearly share (same measure as the portfolio tables, so these line up
with §1–§2).

| Classification choice | L4 income % | L4 spend % |
|---|---|---|
| Base | 42.7 | 10.0 |
| Salary→L3 instead of L1 | 42.7 | 10.0 |
| Income not cleaned | 38.6 | 10.0 |
| *L4A = all financial (old bug)* | 42.7 | *30.3* |
| **Fees→L2 instead of L4** | **4.2** | 10.0 |

Two lessons. The **spending measure was fragile to one definition**: counting the whole financial
category as L4 (the broad, discarded definition) trebled L4 spending (30% vs 10%) — the bug we
corrected; only genuine investment purchases count. Once fixed, the spending result barely moves under
the salary and cleaning choices. The **revenue result has one choice it depends on**: treating fees as
a new business model (L4) rather than personalisation (L2); if L2, the new-revenue share collapses to
4%. We defend L4 but flag it. Either way, the underlying fact holds — old revenue source replaced by a
new one. (The reallocation conclusion in §3 is likewise stable to these choices.)

**Coverage caveat.** The eighteenth-century bands rest on sparse data (only 16 of 50 years in
1750–1799 have records); the dense, reliable window is **1800–1900**, where the pivot sits.

---

## 5. AI interpretation — resource allocation under a technological revolution

Read as a management lesson:

- The organisation that stayed relevant through the Industrial Revolution changed **how it earned far
  more than how it spent**. Revenue was rebuilt (new fee/investment income); spending shifted toward
  higher-order activity too, but only gradually, while most of the budget kept running the existing
  operation.
- The **fastest-growing claim on new money was higher-order transformation (L4)**, even though
  scaling the old operation (L1) still took the largest single share. Efficiency and personalisation
  were kept up, but they were not where the extra emphasis went.
- It **renewed its income base** rather than widening it — one new source grew to rival the old one,
  instead of many small ones.

For an organisation putting scarce resources into AI today, the pattern to notice is that the new
spending went hand in hand with a new way of earning — not funded indefinitely from the old business.
This is a single case, not proof; but it shows what the choices looked like for an organisation that
came through the last great technological upheaval still relevant.

### For next discussion
- The mid-period "other" bucket (income ≈ miscellaneous receipts; expenditure ≈ the `other` category)
  is large in 1820–1859; tightening enrichment categories there would sharpen the portfolios.
- Whether to date the linkage more formally once we accept the L3-baseline caveat above.
