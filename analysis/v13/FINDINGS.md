# v13 — Findings: essay-aligned levels + finer-unit / precursor evidence

> Rationale-first summary. Every number below traces to a script in this folder; the *reason* for
> each test is stated, not assumed. Two questions drove the week: (1) do the levels survive being
> re-operationalised to the professor's essay definitions, and (2) at a finer unit, did the
> 1854/1877 changes have precursors before the Acts?

---

## 1. Re-operationalising L1–L4 to the essay (Part A)

We rebuilt each proxy from the essay's operating-logic definition (see `MAPPING.md` for the full
per-level justification) and computed it beside the old proxy (`proxies_v13.csv`,
`proxy_comparison.png`). Essay-aligned era means:

| Level (essay) | proxy | pre | trans | early | late |
|---|---|---|---|---|---|
| **L1 Automation / cost** | maintenance+domestic upkeep share | 0.26 | 0.09 | 0.15 | 0.18 |
| **L2 Personalisation / revenue** | individual award/fee spend share | 0.04 | 0.23 | 0.14 | 0.15 |
| **L3 Operational innovation / process** | admin + accounting-structure index | 0.24 | 0.25 | 0.26 | **0.43** |
| **L4 Business innovation / new markets** | securities + education-fee income share | 0.05 | 0.02 | 0.03 | **0.32** |

**What this shows, and why it matters.** The core asymmetry survives the stricter definitions: the
two *higher* levels (L3 operational redesign, L4 new business) are where the late transformation
concentrates, L3 reaching 0.43 and L4 0.32 in the late-industrial era from near-zero. But now the
proxies actually mean what the essay says:

- **L4 is now a genuine "new business" measure**, not educational spending. It is income from
  things the college did **not** run in 1700: a securities-investment portfolio (consols, canal,
  railway dividends) and **education sold as fee revenue** (composition/examination fees). This
  passes the essay's "did it exist before?" test, which the old educational-spend proxy failed.
- **L3 is now process redesign** (administrative coordination + a more elaborate, reconciled
  accounting structure), not salary cost.
- **L2 is now individual differentiation** (scholarships, exhibitions, prizes, individual fees),
  not payment regularity — fixing the single clearest old mismatch.

**Caveats (stated up front):** the L4 income signals are keyword-classified from descriptions, so
the very first appearance years (e.g. a stray pre-1750 "per cent" match) are noisy; we rely on
*material* years and change-points, not first-mention. The `other` category is large and
uncategorised in the transition/early window, so mid-period shares are soft.

---

## 2. Finer-unit decomposition of the 1854/1877 breaks (Part B-1)

Rationale: an aggregate index can break at a reform even if no single account did. We ran ITS
(1854 & 1877 level shifts, normalised by the local pre-reform mean) on each **per-category** share
series (`category_its.csv`). The aggregate higher-order break decomposes cleanly:

- **salary_stipend**: −0.99\*\*\* at 1854, **+2.08\*** at 1877 (the disruption-then-acceleration that
  drives the old L3 signal).
- **educational spend**: −0.58\* at 1854, +1.25\* at 1877.
- **financial income**: **+0.67\*** at 1854 (investment income rises right at the first reform).
- **ecclesiastical spend**: −0.53\*\* at 1877 (legacy function fades); **charitable** +1.66\*\*\* at 1877.

So the reform "effect" is not diffuse: it is concentrated in a handful of higher-order accounts
(salaries, education, investment income), exactly the components our L3/L4 proxies are built from.

---

## 3. Were there precursors before the Acts? (Part B-2) — **yes**

Rationale: if the shift already began before 1854, the Act ratified a drift rather than causing a
jump. We re-scanned every series for a change-point **restricted to the pre-reform window
1820–1853** (`changepoints.csv`, `precursor_timeline.csv/png`). The result is consistent and
striking: **every higher-order component already has a change-point in the 1820s–1830s**, a
generation before the 1854 Act.

| component | pre-Act change-point | strength (t) |
|---|---|---|
| educational spend | **1827** | **4.63 (strong)** |
| administrative spend | 1832 | 2.15 |
| L4 securities income | 1824 | 2.10 |
| L4 education-fee income | 1826 | 1.58 |
| ecclesiastical spend | 1829 | 1.38 |
| land-rent income (full-series) | 1806 | 16.6 |

At the same time, the **largest** breaks in the full series are *late* (ecclesiastical 1857,
administrative 1881, educational 1889, domestic 1896). The honest reading is therefore a
**two-phase pattern**: a gradual reallocation that begins in the 1820s–1830s, which the Reform
Acts then **crystallise and amplify** into the large late-century breaks. Educational spending is
the clearest case — a strong, early break at 1827 — and investment income and a broadening revenue
base (land-rent income already breaking by 1806) were underway well before either Act.

**Why this matters for the paper.** It refines our headline. The Reform Acts are still the moment
the change becomes large and durable, but they did not start it; describing them as pure
"strategic discontinuities" overstates the suddenness. "A long pre-reform drift that the Acts
ratified and scaled up" is both more accurate and more interesting, and it is consistent with the
placebo caveat we already flagged (pre-1854 cutoffs also register).

**Caveat:** several pre-Act change-points are statistically modest (t ≈ 1.4–2.2); only educational
spend is strong. We report them as *suggestive precursors*, with educational reallocation as the
one firm pre-1854 onset.

---

## 4. Account-level and sub-annual (secondary)

- **Per-account** (`account_breaks.csv`): the largest persistent accounts carry break years
  consistent with the category story (e.g. a "various financial receipts" income account 1822–1856
  breaks at 1845; ecclesiastical stipends break mid-century). Account text is normalised crudely
  (first 40 chars), so this is illustrative, not definitive.
- **Audit-term language** (`term_timing.csv`): feast-day terms decline after 1854 (Michaelmas
  0.065 → 0.017 of rows; Lady Day 0.023 → 0.010), an indicative sign that the *format* of the
  accounts modernised around the reforms. Coverage is a minority of rows, so this is a hint, not a
  sub-annual series.

---

## 5. Bottom line

1. **The framework survives the essay's stricter definitions** — higher-order L3/L4 still carry the
   late transformation — and the proxies are now faithful (L4 = genuinely new revenue business,
   L3 = process/coordination redesign, L2 = individual differentiation), which addresses the core
   reason the earlier mapping felt arbitrary.
2. **The reforms had precursors.** At the finer unit, the higher-order reallocation begins in the
   1820s–1830s (educational spend breaks at 1827, investment income and revenue diversification
   from the 1820s), and the 1854/1877 Acts amplify it into the large durable breaks. The change was
   building before the Acts, not created by them.

### Next steps to discuss
- Fold the essay-aligned proxies into the main panel and re-run the v11/v12 leapfrogging/durability
  tests on them (do the STRONG results hold with L4 = new-revenue and L3 = structure?).
- Reframe the paper's "strategic discontinuity" as "pre-reform drift ratified and scaled by the
  Acts," using the 1827 educational break as the lead example.
