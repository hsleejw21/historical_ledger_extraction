# analysis_v12 — Workstream Roadmap

> **This folder is the hub for the next phase.** It holds (a) the plan for where the
> discussion goes, (b) the analyses that serve that plan, and (c) the integrating
> deliverable. The frozen central message lives in
> [`../../ANCHOR_NARRATIVE.md`](../../ANCHOR_NARRATIVE.md) — v12 does not relitigate it,
> it executes against it.
>
> **Foundation (done):** the four-level variables are locked and validated in
> [`VARIABLE_DICTIONARY.md`](VARIABLE_DICTIONARY.md) (+ `variable_validation.py`). Theory
> scaffolding is in Ethan's report `Oxford_Arxiv_Report0608.pdf` (transformation-depth
> lens, literature mapping, Table 3 operationalization). Build everything else on these.

---

## 1. Big picture (the one thing this phase is about)

**Stop thinking of this as a paper about Oxford. It is a study of how organizations
transform during a technological revolution — using a *completed* revolution (Oxford,
1700–1900) as a laboratory to generate testable lessons for the *ongoing* one (AI).**

The governing question (the professor's words):

> *What kinds of transformation actually matter when technology fundamentally changes
> the environment?*

Our answer, operationalized through the **Four-Level AI Transformation Framework**
(L1 Automation · L2 Personalization · L3 Operational Innovation · L4 Business-Model
Innovation): **L1/L2 suffice in calm periods; durable adaptation under disruption
requires L3/L4, and organizations can get trapped optimizing the lower levels.**

Oxford = historical laboratory. AI = contemporary motivation. The bridge is the
**Oxford→AI interpretation table** (§4, deliverable D1).

---

## 2. What is ALREADY done (do not re-run — consolidate)

Most of the professor's empirical asks were answered in **analysis_v11**. v12 inherits
these; it does not redo them.

| Professor's ask (Jungwoo) | Status | Where it lives |
|---|---|---|
| **1. L1–L4 importance before/after shocks; calm vs disruption** | ✅ done | `reports/analysis_v11/conditioning_by_regime.csv` — L4 concentration 1.38 / response 2.59 vs 1.74; L2 below-neutral 0.94 |
| **2. Evidence L1/L2 insufficient (persistence, long-run contribution, sustained vs temporary)** | ✅ done | `reports/analysis_v11/persistence_decomposition.csv` — only L4 "sustained/amplifying"; L3 transitory; L1/L2 no durable gain |
| **3. Lower-level trap (efficiency trap)** | ✅ done | `reports/analysis_v11/lockin_counterfactual.csv` — L1/L2-only org net-regresses; higher-order gap +0.18–0.24 by 1890s |
| **4. Strengthen resource-slack story** | ⚠️ **WEAK — the real gap** | `reports/analysis_v11/slack_leadlag.csv` — every CI includes zero; only significant link is *contemporaneous* land-rent↔L4 where **L4 leads**, not slack |
| **5. Oxford→AI interpretation table** | 🟡 v1 exists, needs elevation | `reports/analysis_v11/oxford_ai_table.csv` — 8 rows, good but buried in an HTML report |

**Conclusion:** asks 1–3 are STRONG and finished. Ask 4 is the weak link *and* the
professor's stated conviction. Ask 5 needs to be promoted from a buried CSV to the
project's flagship artifact. **That defines v12's two jobs.**

---

## 3. v12 focus (two jobs)

> **Re-scoped 2026-06-13:** after locking the variables (D0), the round's priority became
> the **L4 textual triangulation** (done — corroborated) and moving toward the **paper
> skeleton + Oxford→AI table** (D1/D2). The slack work below is real but deferred to D4;
> it is the eventual strengthening pass, not this round's focus.

### Job A — Strengthen the resource-slack story *(the genuine new empirical work)*

The professor: *"I continue to believe this may be an important mechanism… What enabled
Oxford to move toward L3 and L4?"* Our own v11 assessment tags it **SUGGESTIVE / weakest
link**. This is where new analysis has the highest marginal value.

The v11 slack test was thin: two proxies (financial slack = income−expenditure;
land-rent income), simple cross-correlation, wide CIs. v12 should make a real attempt to
either *establish* or *honestly bury* the mechanism:

- **Broaden the slack construct.** Revisit *all* existing slack-related evidence the
  professor named: land-rent income (`analysis_v4`), financial capacity / surplus,
  revenue diversification (HHI, `analysis_v4/revenue_diversification_yearly.csv`, falls
  <0.35 in 1783), and capability investment (L3). Build a small panel of 3–4 slack
  proxies rather than 2.
- **Better identification.** Move beyond raw cross-correlation: first-differenced
  lead-lag with block bootstrap; condition on the reform windows (does slack *before* a
  shock predict the *size* of the L3/L4 response *after*?). The reform-conditioned
  version is the sharpest test and isn't in v11.
- **Be willing to report a null.** The professor explicitly is *not* asking for causal
  proof — "whether the data support this interpretation." A clean "slack co-moves but
  does not lead" is a publishable, honest result. Do not inflate.

**Output:** `analysis_v12/slack_mechanism.py` → `reports/analysis_v12/slack_*.csv` + one
figure + a strength verdict (expected: SUGGESTIVE→MEDIUM at best; possibly a documented
null).

### Job B — Promote the Oxford→AI interpretation table to the flagship deliverable (D1)

The v11 table (8 rows) is good but lives inside an HTML report. The professor wants this
to be *the* contribution: *"For each major finding, document Oxford evidence,
corresponding AI interpretation, and the lesson for contemporary organizations."*

v12 produces a standalone, citable **`OXFORD_TO_AI.md`** in this folder that:
- carries every finding with its **strength tag** (STRONG/MEDIUM/SUGGESTIVE/USEFUL NULL),
- maps each to a concrete **contemporary-organization lesson**,
- links each row to its source artifact (the appendix already drafted in ANCHOR_NARRATIVE).

This is mostly consolidation + the updated slack row from Job A. Low risk, high visibility.

---

## 4. Deliverables

| ID | Deliverable | Type | Status |
|----|-------------|------|--------|
| **D0** | `VARIABLE_DICTIONARY.md` + `variable_validation.py` — locked 4-level variables; L1 boundary fixed; L4 triangulation corroborated | foundation | ✅ **done** |
| **D1** | `OXFORD_TO_AI.md` — flagship interpretation table (every finding → AI lesson, strength-tagged) | consolidation | next (P0) |
| **D2** | `PAPER_SKELETON.md` — manuscript outline (RQ, motivation, data, strategy, findings, mechanism, AI implications, boundary conditions) | consolidation | next (P0) |
| **D3** | `REGIME_EVIDENCE.md` — clean pre/post-reform + calm/disruption summary | consolidation (v11) | P1 |
| **D4** | `slack_mechanism.py` — reform-conditioned slack test (strengthen *or* bury) | new analysis | P1 (not prioritized this round) |
| **D5** | `theory_mapping.md` / `theory_to_data.md` — stubs integrating Ethan's tables | scholarship (Ethan) | P2 |

**Done this round:** D0 (variable decision + validation). **Next:** D1/D2 — turn the locked
variables + existing v8–v11 evidence into the flagship Oxford→AI table and the paper
skeleton. Slack (D4) stays a later strengthening pass, per the round's scoping.

---

## 5. What we are NOT doing

- **Not re-running v11.** Asks 1–3 are answered and STRONG; re-deriving them is the
  "too much analysis" trap.
- **Not adding new transformation findings for their own sake.** Everything in v12 must
  attach to an AI lesson or it doesn't belong (the *"의미가 없음"* test).
- **Not overselling slack.** If Job A returns a null, we report the null.

---

## 6. Folder conventions

- Scripts in `experiments/analysis_v12/`: `ROOT = Path(__file__).resolve().parents[2]`
  (same depth as `experiments/analysis/`). To reuse the v9/v11 engine, add
  `experiments/analysis/` to `sys.path` or import via relative path.
- Write computed outputs to `experiments/reports/analysis_v12/` with
  `OUT.mkdir(parents=True, exist_ok=True)`.
- Narrative/markdown deliverables (D1, D3–D5) live *in this folder* — it is the hub.

---

## 7. Links

- Frozen central message: [`../../ANCHOR_NARRATIVE.md`](../../ANCHOR_NARRATIVE.md)
- Prior empirical engine: `experiments/analysis/analysis_v11.py` (+ v9, v10)
- v11 outputs: `experiments/reports/analysis_v11/`
- Research inventory (preserved findings): v10 inventory / `reports/analysis_v10/`
