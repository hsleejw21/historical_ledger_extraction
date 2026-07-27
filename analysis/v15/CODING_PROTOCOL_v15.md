# v15 — Coding Protocol: the two portfolios (income and expenditure)

> **Purpose.** V14 classified mostly *expenditure*. The resource-allocation task needs both sides:
> how the college **generated** resources (income) and how it **deployed** them (expenditure). This
> document (a) extends the L1–L4 protocol to the **income side** with the same Q1–Q4 discipline,
> (b) states the **non-revenue exclusion rules** that clean the income before it is classified, and
> (c) gives the **full-coverage expenditure mapping** so every pound lands in exactly one level and
> the shares sum to one. Construct definitions are unchanged from `../analysis_v14/CODING_PROTOCOL.md`;
> here we only decide *where each income and expenditure line belongs*.

---

## A. Cleaning both sides first (the raw ledger is not all real money)

A large share of recorded entries is **internal accounting**, not money moving to or from the outside
world — and this is true on **both** sides of the ledger (receipt-side "total receipts", "arrears";
payment-side "total payments", "closing balance", "loss of guineas"). Left in, it swamps the
portfolios. The same artefact filter (`RX_ART`, plus `RX_CAPITAL` for one-off asset sales) is applied
to income **and** expenditure before classification (~75% of income, ~87% of expenditure survive).
*This symmetry matters: an earlier draft cleaned only income, which left the expenditure L4 inflated.*

| Removed family | Examples in the ledger | Why it is not revenue |
|---|---|---|
| **Aggregates / totals** | "total receipts recorded (aggregate)…" | A sum of other rows — double-counts if kept. |
| **Carry-forward / opening balances** | "opening receipts computed from the sums", "amount brought in this year" | Last year's closing balance, not this year's earning. |
| **Arrears / debts of the preceding year** | "arrearages received", "debts due from the preceding year" | Timing of an old claim, not new income. |
| **Unclear / transcription** | "unclear — possible transcription error", "no description" | Not attributable to any source. |
| **Capital (one-off asset sales)** | "proceeds from the sale of stock/investments" | Disposing of an asset is a capital event, not a recurring revenue stream. |

*Falsification (Q4) for the cleaning itself:* an item wrongly excluded would be one that is in fact a
**recurring external receipt** mislabelled as a balance/arrear. We keep the rules keyword-explicit so
this is auditable, and we report the retained fraction.

---

## B. Income → L1–L4 (the revenue portfolio)

The revenue side does **not** populate all four levels equally: personalisation and operational
innovation are ways of *spending*, not *earning*. The honest shape of Oxford's revenue is
**Traditional (L1) vs New business (L4)**, and we say so rather than forcing a four-way split.

### L1 — Traditional revenue *(the unchanged earning activity)*
- **Q1 (AI def).** The same activity, done as before; value driver = cost/scale, not a new market.
- **Q2 (proxy).** `land_rent` + `ecclesiastical` income — the endowment the college has always lived
  on (estate rents, tithes). Present in 1700 and still present in 1900.
- **Q3 (supports).** It is the *core, recurring* receipt; its logic (own land, collect rent) never
  changes.
- **Q4 (falsifies).** If a "rent" were in fact a **new** instrument (a rent-charge sold as a security)
  it would be L4; if estate management became a **fee business** for others it would be L4.

### L4 — New-business revenue *(a revenue model that did not exist before)*
- **Q1 (AI def).** A genuinely new way of capturing value; test: *"if it existed before, it is not L4."*
- **Q2 (proxy).** (i) **securities/annuity investment** — `financial` income matching `RX_SEC`
  (consols, canal, railway, exchequer annuities): the college becomes an *investor*. (ii)
  **education-as-fee** — `educational` income or `RX_FEEINC` (composition, tuition, caution, battels,
  matriculation, degree/entrance, **fee fund**): teaching sold as a *paid service*.
- **Q3 (supports).** Absent-early / present-late, and a new value-capture logic (investor; paid
  service), confirmed by the first-appearance test (`first_appearance.csv`).
- **Q4 (falsifies).** If the "new" income is **relabelled old rent/interest**, or an internal
  transfer with no external market, it is not L4.

### L2 / L3 income — reported, but essentially nil
- **L2 (individual-differentiated revenue):** no clean signal — the college awards to individuals
  (an expenditure), it does not sell to them by individual. Reported as ≈0 and stated plainly.
- **L3 (operational revenue):** `administrative` income (management fees/fines) — small; included for
  completeness (2–8% late), not a revenue model.
- Everything else genuine but unclassifiable (miscellaneous receipts, benefactions) → **other**.

---

## C. Expenditure → L1–L4 (full coverage, shares sum to 1)

Same construct definitions as v14, extended so **every** category is assigned. The one new decision
is personnel: by v14's own rule (*"more clerks doing the same thing = L1 headcount, not L3"*) the
standing wage/stipend bill is the **cost of running the existing establishment** and belongs to L1;
L3 is reserved for the *administrative* category (the reorganisation of the operating model itself).

| Level | Ledger categories / rules | Rationale (Q3) | Would re-classify it (Q4) |
|---|---|---|---|
| **L1 upkeep + personnel** | `maintenance`, `domestic`, `ecclesiastical`, `salary_stipend` | Recurring cost of the unchanged operation, incl. the standing wage bill | A wage tied to a *redesigned* workflow → L3; a role that is a new paid service → L4 |
| **L2 individual awards** | any line matching `RX_AWARD` (scholar/exhibition/prize/…) | Provision differing by named individual | If it reorganised teaching → L3; if a fee market → L4 |
| **L3 administration** | `administrative` | The operating/coordination model itself | If only more headcount doing the same → L1 |
| **L4A business-model outflow** | `financial` expenditure matching `RX_INVEST` (purchase of stock/consols/investments) **only** | Deploying money into the new investment business | Other `financial` rows (internal fund transfers, losses, totals) are **not** investment → "other" |
| **L4B mission** | `educational` **not** matching `RX_AWARD` | Reallocation toward the teaching/scientific purpose | If really a revenue play → L4A; if scaling an old function → L1 |
| **other** | `other`, `charitable`, `land_rent` (exp.) | Unspecified / not attributable with current granularity | — (kept explicit, ≈20%, not hidden) |

For the two portfolios we report four levels (**L4 = L4A + L4B**); the L4A/L4B split is retained only
for the income↔expenditure linkage, where "does the business model precede the mission?" is the
question.

---

## D. How this protocol is used
- Every series in `analysis_v15.py` is built from exactly one rule above.
- The revenue portfolio uses the **cleaned** income (Section A); the investment portfolio uses all
  expenditure (Section C). Per-year shares sum to 1 by construction (verified in `main()`).
- The **"other"** bucket is reported, never dropped, so the reader sees how much is unclassified
  (income ≈ misc receipts; expenditure ≈ the `other` category).
