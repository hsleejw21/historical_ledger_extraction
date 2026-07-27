# V18 — Discriminating Among Competing Mechanisms

**Task (Prof. Hu, 2026-07-21):** stop confirming the budgeting-rule story. Treat it as one candidate
among several, ask what each candidate *would* imply, test those implications, and report which
survive. Do not settle on a narrative yet.

**Script:** `analysis_v18.py` · **Outputs:** `t0_*` … `t6_*` CSVs, `mechanism_discrimination.png`,
`absorption_horserace.png`

---

## The one-paragraph answer

Of the six candidate mechanisms, **one is supported, one is a genuine partial contributor, and four
are contradicted by the archive.** The decision-rule story survives because it makes a prediction the
others do not — that the *anchor* of spending switches from current income to multi-year income — and
that switch is exactly what the data show (λ on current income 0.76 → 0.26 n.s.; λ on 5-year
permanent income 0.06 n.s. → 0.67, p = 0.021). Revenue diversification turns out to be a real but
partial contributor: simulating the unchanged pre-1820 rule against the actual post-1820 income
reproduces **only ~29%** of the observed smoothing, leaving ~71% that requires the rule itself to have
changed. Governance, planning horizon, financial management and accounting practice all leave datable
traces in the ledger, and **none of them breaks at 1820** — they break in 1846–1883, i.e. they follow
the 1854 reform rather than precede the 1820 change.

**Two honest caveats that matter more than the verdicts.** (1) One of the tests I built — the
absorption horse race, T5 — **failed**: a meaningless linear time trend absorbs 93% of the effect, so
that design has no power to attribute anything, and no rival can be ruled in *or* out by it. (2) M1
is currently supported partly by elimination, and its independent evidence (the anchor switch) rests
on 15 usable pre-1820 observations. Discriminating further needs cross-sectional variation this
single institution cannot supply.

---

## T0 — Which results are facts and which are choices?

Before testing anything, one housekeeping result, because it changed how everything below is
computed. Two years (1859, 1862) are known-bad income extractions (v17: 1859 records £32 of income
against £8,353 of spending). Three defensible treatments — keep them (what v16/v17 did), drop them,
or interpolate them:

| treatment | λ 1800-19 | λ 1820-53 | λ 1854-1900 | 1820 interaction | scan argmax | smoothing 1800-19 → 1820-53 |
|---|---|---|---|---|---|---|
| raw (v16/v17) | 0.948*** | 0.115 n.s. | 0.184* | −0.764 (p<0.001) | 1828 | 1.56 → 0.72 |
| **masked (v18 default)** | 0.948*** | 0.115 n.s. | 0.244 n.s. | −0.719 (p=0.005) | 1819 | 1.56 → 0.72 |
| interpolated | 0.948*** | 0.115 n.s. | 0.300 n.s. | −0.670 (p=0.007) | 1819 | 1.56 → 0.72 |

**Fact:** the 1820 collapse (0.95 → 0.12) and the smoothing flip (1.56 → 0.72) are identical under all
three. **Choice:** everything about the *post-1854* era. The default here is **masking** — dropping
those years rather than inventing an income path for them.

**A near-miss worth recording.** With a *restricted* specification (only the error-correction term
allowed to differ across regimes) and the bad years repaired, the break scan moves to the **1860s**
and the 1820 step fits no better than a smooth trend. That looked, briefly, like the 1820 story
falling apart. It is the restriction talking: the eras differ in their short-run dynamics too, and a
model forbidden from saying so launders that difference through the gap coefficient. Fully
interacted, the scan returns 1819 and the two-break model is decisively rejected (ΔBIC 12.1, second
interaction p = 0.61). Both versions are in `t5b_step_vs_trend.csv`.

---

## T1 — Current vs permanent income *(the professor's test 1)*

Two specifications, era by era. The **levels** version — what does spending error-correct *back to*? —
is the informative one.

| era | λ on **current** income | λ on **5-yr permanent** income |
|---|---|---|
| 1800-1819 | **0.759 (p = 0.020)** | 0.062 (n.s.) |
| 1820-1853 | 0.260 (n.s.) | **0.674 (p = 0.021)** |
| 1854-1900 | 0.139 (n.s.) | 0.082 (n.s.) |

**This is the cleanest result in the file, and the professor's predicted pattern exactly.** Before
1820 spending corrects to this year's receipts and ignores the multi-year average; after 1820 the two
swap places. Stable at k = 3, 5, 7.

The third row is new and was not predicted by anyone: after 1854 spending anchors to **neither**
income concept. Either a third regime begins at the reform, or the post-1854 income series is too
compromised to detect an anchor (T0 shows this era is exactly where the data are weakest). Not
resolved here.

The **growth** version (spending growth on current and permanent income growth) is much weaker and
close to uninformative pre-1820 — the growth of a 5-year mean has about a fifth of the standard
deviation of income growth, so the permanent coefficient is barely identified. Reported in
`t1_growth_horserace.csv`, not leaned on.

**Strength: STRONG for the switch, with the caveat that the pre-1820 era supplies only 15 usable
observations** (a 5-year trailing mean cannot start before 1805, and 1750-99 is an archive gap).

---

## T2 — Income-shock response *(the professor's test 2)*

### (a) Transitory vs permanent — **does not deliver a clean verdict**

In the exact-decomposition form the pre-1820 coefficients are too imprecise to distinguish anything
(n = 15). In the level-deviation form the transitory response falls 1.86 → 0.62 → 0.97, but the
pooled test of that change gives p = 0.17. Reported as **inconclusive**; the two parametrisations
disagree and I am not going to pick the one that flatters the story.

### (b) Positive vs negative shocks — **the informative half**

| era | response to **positive** shocks | response to **negative** shocks |
|---|---|---|
| 1800-1819 | **1.76 (p = 0.004)** | −0.35 (n.s.) |
| 1820-1853 | 0.76 (p = 0.013) | 0.05 (n.s.) |
| 1854-1900 | 0.81 (p < 0.001) | 0.22 (n.s.) |

What changed at 1820 is the treatment of **windfalls**, not of shortfalls. Before 1820 a good year was
spent, roughly one-and-three-quarters-for-one; after 1820 less than half as much. Negative shocks were
never passed through to spending, in *any* era.

This discriminates against M5 (a reserve appeared). A newly-available reserve should show up first
on the **downside** — that is what reserves are for — and the downside is precisely where nothing
changes. What changed is that the college stopped automatically spending money it happened to
receive, which is a rule, not a buffer.

**Strength: SUGGESTIVE.** The direction is clear and consistent across eras, but the formal test of
the change in the positive-shock response is p = 0.13.

---

## T3 — Excess smoothness *(the professor's test 3)*, and the test that decides M1 vs M2

| era | sd(income) | sd(spending) | smoothing ratio | excess smoothness vs PIH benchmark |
|---|---|---|---|---|
| 1800-1819 | 0.407 | 0.636 | 1.56 | 2.29 |
| 1820-1853 | 0.679 | 0.487 | 0.72 | 1.29 |
| 1854-1900 | 0.703 | 0.649 | 0.92 | 1.41 |

Spending is smoother than income after 1820 — but note **why the ratio moves**: spending volatility
falls modestly (0.64 → 0.49) while income volatility rises sharply (0.41 → 0.68). That is exactly what
rival M2 predicts, and it is the reason M2 has to be taken seriously rather than waved away.

### T3b — The composition placebo (the decisive test)

Estimate the pre-1820 rule once. Run that **fixed, unchanged** rule forward from 1820, driven by the
**actual** 1820-53 income. If lumpier income alone explains the smoothing, the placebo should
reproduce it.

- **Calibration first** (without which the placebo would be unfalsifiable): fed its own 1800-19
  income, the simulator returns 1.87 [1.14, 3.21] against an observed 1.56. It reproduces the era it
  was fitted to. ✓
- **Placebo:** fed the actual 1820-53 income, the unchanged rule gives **1.32 [0.86, 2.22]**, against
  an observed **0.72**. 99.7% of draws lie above what Oxford actually did.
- **Decomposition of the fall 1.56 → 0.72:** income composition accounts for **28.9%**; the remaining
  **71.1%** requires the rule itself to have changed.

So M2 is neither dismissed nor sufficient. It is a **real contributor of roughly three-tenths**, which
is a more useful and more defensible statement than either "diversification explains it" or
"diversification is irrelevant."

**Strength: STRONG** (calibrated, bootstrapped over both parameter and innovation uncertainty).

---

## T4 — Break-date horse race: whose trace moves at 1820?

If a mechanism is to *explain* an 1820 break, its own observable trace has to move at or before 1820.
Sup-Wald mean-shift scan, 15% trimming, Andrews 5% critical value 8.85.

| mechanism | proxy | break | significant? |
|---|---|---|---|
| — (reference) | **income–spending coupling** | **1819** | yes |
| M3 governance | distinct persons per **page** | 1818 | yes — *but see below* |
| M6 accounting | rows per year | 1823 | **no** |
| M2 diversification | non-endowment income share | 1846 | yes |
| M6 accounting | rows per page | 1848 | no |
| M3 governance | institutional payee share | 1849 | yes |
| M5 financial | investment purchases | 1850 | no |
| M5 financial | securities income | 1854 | yes |
| M6 accounting | arrears share | 1857 | yes |
| M3 governance | signature rows per page | 1868 | yes |
| M4 horizon | commitment length | 1872 | no |
| M2 diversification | effective no. of income sources | 1881 | yes |
| M6 accounting | Latin share / sections / subtotals / pages | 1882 | yes |
| M3 governance | persons per **row** | 1882 | yes |
| M4 horizon | multi-year share | 1883 | yes |

**The one apparent hit is not a hit.** Distinct people per *page* breaks at 1818 — but rows per page
falls at the same moment (33.3 → 26.8), and a page with fewer lines names fewer people. Normalised per
row, the same governance marker breaks in **1882**. This is why the density-normalised proxy is in the
table: the raw version would have handed M3 a false positive.

**Nothing else comes near 1820.** Every rival's trace moves between 1846 and 1883 — clustered on the
1854 reform and the 1870s-80s. The change in how Oxford spent has *no observable companion* in how
Oxford was governed, how long it committed, what it invested in, or how it kept its books.

**Strength: STRONG as elimination.** Its weakness is symmetrical and should be said out loud: **M1
has no independent trace either.** The coupling *is* M1's trace. That is why T1's anchor switch and
T3b's placebo carry the positive case, and T4 only clears the field.

---

## T5 — Absorption horse race: **a failed test, reported as failed**

The design: put each rival proxy into the error-correction model interacted with the gap, and see
whether the post-1820 collapse (θ) survives. Then I added a **linear time trend as a placebo rival**.

> The meaningless trend absorbs **93%** of the effect: θ goes from −0.417 (p = 0.021) to −0.029
> (p = 0.95) — more than any real rival manages.

With a single 99-year series, any smoothly-moving covariate can soak up a step change. The test
therefore has essentially no power, and the fact that `latin_share` reduces θ by 68% is **not**
evidence for an accounting explanation — the meaningless trend does better and means nothing. Every number in
`t5_absorption.csv` should be read with the placebo row alongside it.

I am reporting this rather than deleting it because the failure is informative: **this archive cannot
discriminate mechanisms by controlling for them.** It can only discriminate by (i) distinctive
predictions, (ii) break-date coincidence, and (iii) simulation. What T5 was reaching for needs
cross-sectional variation — several institutions with different governance changing at different
dates — which is a data-collection question, not an econometric one.

### T5b — What the shape of the change can still tell us

A drift is what most rivals imply (diversification, lengthening horizons, the Latin-to-English shift
are all gradual). An abrupt change is what a *rule* implies — rules are adopted, not drifted into.
Fully interacted, on BIC:

| model | break | ΔBIC | θ |
|---|---|---|---|
| **STEP (best scanned)** | **1819** | **0.0** | −0.805 (p = 0.002) |
| STEP @1820 | 1820 | 0.8 | −0.719 (p = 0.005) |
| TREND | — | 2.7 | −0.781 (p = 0.096) |
| TWO STEPS | 1820 + 1872 | 12.1 | second: +0.249 (p = 0.61) |

The data prefer **one abrupt change, at 1819-1820** — not a drift, and not two changes.

---

## T6 — Verdicts

| mechanism | verdict | strength |
|---|---|---|
| **M1 decision rule** (current → permanent anchor) | **SUPPORTED** — the only candidate whose distinctive prediction is confirmed | STRONG |
| **M2 diversification** | **PARTIAL** — a real contributor (~29% of the effect), insufficient alone | STRONG |
| **M3 governance** | NOT SUPPORTED at 1820 — markers move with the 1854 reform; the one 1818 hit is a page-layout artefact | MEDIUM (proxies indirect) |
| **M4 planning horizon** | CONTRADICTED — horizons lengthen ~50 years too late, and multi-year spend is only 2-4% of the total | STRONG |
| **M5 financial management** | NOT SUPPORTED as cause — what changed is the response to windfalls, not to shortfalls; financial assets arrive with the reform | MEDIUM (reserve *stock* unobserved) |
| **M6 accounting practice** | NOT SUPPORTED — the ledger's form changes 1854-1882, never at 1820 | STRONG |

---

## What I am *not* claiming

- **Not** that M1 is proved. It survives; the others do not. Survival plus one confirmed distinctive
  prediction is a strong position, not a demonstration.
- **Not** that the professor's tests 1–3 all came back positive. Test 1 did, decisively. Test 3 did,
  once the composition rival was properly simulated rather than assumed away. **Test 2 did not**: the
  transitory/permanent split is inconclusive in this sample, and the informative result there (the
  windfall asymmetry) is only suggestive at p = 0.13.
- **Not** that governance is irrelevant to the story. The archive records payees and signatures, not
  statutes, committees or minutes. What I can say is that the *recorded* governance markers do not
  move at 1820. The decisive records are outside this dataset.
- **Not** that the post-1854 era behaves like 1820-53. On the contrary, spending anchors to neither
  income concept after 1854 (T1) and the smoothing ratio returns to ~1.0 (T3). Whether that is a third
  regime or a data limit is unresolved and worth its own week.

## What would decide it

1. **Cross-institutional variation.** Other colleges' bursarial accounts, ideally ones whose
   governance changed at different dates. This is the only thing that gives T5 the power it lacks.
2. **The reserve stock.** If a bursar's balance book exists for 1810-1840, M5 becomes testable
   directly instead of by inference from flows.
3. **The 1854-1900 anomaly.** The anchor disappears entirely after the reform. That is either the
   most interesting finding in the project or an artefact of the sparsest part of the archive, and
   the two are currently indistinguishable.
