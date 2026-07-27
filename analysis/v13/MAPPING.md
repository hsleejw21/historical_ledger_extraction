# v13 — Re-operationalising L1–L4 from the professor's essay

> **Why this document exists.** The earlier proxies were attached to the Four-Level framework
> *after the fact* and mapped loosely. The professor's essay (`ai-transformation-in-business.txt`)
> defines the levels by **operating logic and value driver**, not by spending category. This memo
> re-derives each Oxford proxy *from* the essay's definition, states the value driver, names the
> data fields, and gives the reason the proxy is a faithful translation, together with what the
> old proxy got wrong. Nothing here is chosen "just because"; each choice has a stated rationale.

## What the essay actually says (the definitions we must match)

| Level | Essay definition | Defining test | Value driver | Process changed? |
|-------|------------------|---------------|--------------|------------------|
| **L1 Automation** | A machine does the *same predefined task* a human did; substitution **without redesign** | "substitution without redesign" | **Cost** (capped by the labour replaced) | No |
| **L2 Personalisation** | *Differentiated outputs to individuals* using individual-level data | "differentiation without process redesign" | **Revenue** (better demand alignment) | No |
| **L3 Operational Innovation** | The *workflow itself is redesigned*; decision rights move; human role is repositioned from execution to supervision/coordination | "has the workflow architecture changed, and the human role been repositioned?" | **Process performance** | **Yes** |
| **L4 Business Innovation** | A *new business / revenue / industry structure that did not exist before* | "if the business existed before, it is not L4" | **New markets** (unbounded) | The business itself changes |

Three of the essay's framing rules constrain us: the framework is **diagnostic, not a ladder**
(levels coexist), it is **not technology-based** (the level is the *role* intelligence plays, not
the tool), and **risk/value rise together** as depth increases. Our proxies must therefore measure
the *role a function plays in the operating model*, not merely how much was spent on a category.

---

## The v13 mapping (with rationale)

Empirical shares below are amount-weighted, by era (pre-industrial / transition / early- /
late-industrial), from the enriched ledger. They motivate each choice.

### L1 — Automation → *cost of running the unchanged operation*
- **Proxy:** expenditure share on **recurring operational upkeep** that is delivered the same way
  year after year: `maintenance + domestic` (building fabric, servants, provisioning), read as a
  **cost** burden, together with how **standardised/regular** those payments are
  (`payment_period` regularity on this bloc).
- **Fields:** `category ∈ {maintenance, domestic}`, `payment_period`, real amount.
- **Rationale (why faithful):** the essay's L1 is "same task, cheaper, process unchanged." That is
  exactly recurring housekeeping: the college always maintains buildings and feeds members; L1
  asks only whether that *fixed* function is run leanly and regularly. So L1 should track the
  **cost of the unchanged operation**, and its standardisation, not strategic change.
- **What the old proxy missed:** old L1 = `1 − (eccl+maint+domestic)` share, i.e. "how much the
  college has moved *away from* legacy." That measures strategic exit, which is an L3/L4 idea, not
  L1's "run the same task cheaper." We separate the two: ecclesiastical decline is reallocation
  (belongs to the higher-level story); maintenance/domestic *cost-efficiency* is L1.
- **Signal in data:** maintenance 9.8→6.9→11.2→7.7%; domestic 9.8→2.1→3.0→6.6% — a bounded,
  non-strategic operational bloc, consistent with an L1 "cost" reading.

### L2 — Personalisation → *individual-level differentiation that earns/【allocates by individual】*
- **Proxy:** share of transactions that **differentiate provision at the individual level**:
  scholarships, exhibitions, and prizes awarded to *named individuals*, and individualised
  educational fees — operationalised as the **scholarship/prize/exhibition + per-individual fee**
  share, supported by `person_name` density on award/fee lines.
- **Fields:** `category = educational` filtered to individual awards/fees, `person_name`,
  description terms (scholar, exhibitioner, prize, composition fee).
- **Rationale:** the essay's L2 is "different output *to each individual*, within an unchanged
  process, for revenue/engagement." A college's analogue of "personalised output" is
  differentiated treatment of individual members — who receives a scholarship, an exhibition, a
  prize, a tailored fee. The process (educating) is unchanged; *what each individual receives*
  varies. That is personalisation in the essay's exact sense.
- **What the old proxy missed:** old L2 = payment-modernity index (how *regular* payments are).
  Regularity of payment is a process-hygiene measure; it has **nothing to do with individual
  differentiation**. This was the single clearest mismatch, and the main reason to rebuild.
- **Signal in data:** `person_name` appears on 41–51% of entries (individual-level granularity is
  pervasive and measurable); educational provision to individuals grows sharply late.

### L3 — Operational Innovation → *redesign of the operating/coordination machinery*
- **Proxy:** **administrative coordination + accounting-structure formalisation**: administrative
  expenditure share **plus** the formalisation of the ledger's own operating model — entries per
  page, the entry-to-total ratio, `section_header` diversity (number of distinct managed
  accounts), and the adoption of structured double-entry — plus salaries of **coordinating** roles
  (bursar, auditor, administrator).
- **Fields:** `category = administrative`, `section_header` (account structure), row-type mix
  (entry/total ratio), accounting-structure metrics already computed in `analysis_v2/`.
- **Rationale:** the essay's L3 is *workflow redesign + role repositioning*: the process itself
  changes and humans move from doing to coordinating/supervising. In a ledger we cannot see
  meetings, but we can see the **operating model becoming more structured** — more distinct
  managed accounts, more subtotalling and reconciliation, a permanent administration. A more
  elaborate, reconciled, professionally-administered account *is* the historical trace of workflow
  redesign and role repositioning.
- **What the old proxy missed:** old L3 = salary/stipend share, i.e. the *cost of labour*. Paying
  more salaries is not redesigning a workflow (you can pay many people inside an unchanged
  process). L3 must measure **coordination and structure**, not headcount cost.
- **Signal in data:** administrative spend is a steady 11–15%; the accounting structure
  (sections, subtotals, double-entry) measurably elaborates over the 19th century.

### L4 — Business Innovation → *a revenue business that did not exist before*
- **Proxy:** share, first-appearance, and growth of **genuinely new revenue logics** that fail the
  "existed before?" test: (a) **institutional securities investment** income — dividends from
  consols (from 1745), canal (1809), and railway (1845) holdings, i.e. Oxford entering the
  capital market as an investor; and (b) **education sold as fee revenue at scale** — composition
  and examination fees (educational *income*), a revenue model rather than a charitable function.
- **Fields:** `direction = income`, `category ∈ {financial, educational}` sub-classified by
  description (consols, dividend, stock, 3 per cent, canal, railway; composition/examination fee),
  with **first-appearance dates** to enforce the newness test.
- **Rationale:** the essay's L4 is strict — *the business itself is new*; "if it existed before, it
  is not L4." Teaching existed at Oxford for centuries, so educational *spending* is **not** L4
  (this is the old proxy's error). What is new is the **business model**: owning a securities
  portfolio (a financial business the college did not run in 1700) and **charging individuals fees
  as a major revenue stream** (education monetised). Both are datable new revenue categories, so we
  test them with first-appearance, exactly as the essay's "did it exist before?" demands.
- **What the old proxy missed:** old L4 = educational *expenditure* share. Oxford always spent on
  education; a rising share is reallocation/mission-emphasis, not a *new business*. L4 must be the
  appearance of revenue that **was not previously possible**.
- **Signal in data:** **educational income** rises 2.2→0.8→0.4→**29.8%** (a near-absent revenue
  line becomes a third of income late — the education *business*); **financial income** is large
  throughout (29–53%), so we must isolate the *securities-investment* part by description to avoid
  counting old rents/interest as "new."

---

## Honest caveats (stated up front, per the rationale principle)
- **"other" is noisy** (up to ~40% of early-era expenditure is uncategorised); category shares in
  the transition/early window must be read with that in mind.
- **L4 requires sub-classification by text** (securities vs ordinary financial income); we report
  the rule used and its coverage, not a black-box number.
- These are **expenditure/income shares and account structure**, not decision rights or headcounts;
  we measure the *trace* of each operating logic in the books, which is the most the ledger allows.

## Old vs new — at a glance
| Level | Old proxy | v13 proxy | Core reason for change |
|-------|-----------|-----------|------------------------|
| L1 | 1 − traditional share | maintenance+domestic cost & regularity | old measured strategic exit, not "run same task cheaper" |
| L2 | payment-modernity index | individual awards/fees (personalisation) | regularity ≠ individual differentiation |
| L3 | salary/stipend share | admin + accounting-structure formalisation | labour cost ≠ workflow redesign |
| L4 | educational share | new revenue businesses (securities, fees) + newness test | spending on an old function ≠ a new business |
