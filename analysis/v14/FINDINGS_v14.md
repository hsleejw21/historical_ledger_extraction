# v14 — Findings: L4 boundary, three phases, and sensitivity

> Answers to the professor's joint task. Construct definitions are fixed in `CODING_PROTOCOL.md`
> (each with a falsification clause). Rationale is stated for every step; numbers trace to
> `analysis_v14.py`.

---

## 1. The L4 boundary: business model (L4A) and mission (L4B) are not the same thing

We measured the two halves separately and asked the professor's three questions.

- **Do they move together?** Not tightly. Their year-to-year movements correlate only **0.26**
  at the same time. So lumping them into one "L4" hides more than it shows.
- **Does one appear first?** Yes. The business-model series (L4A: investment income and fees) has
  its main turning point in **1876**; the mission series (L4B: educational spending) turns in
  **1889**. The new revenue comes first.
- **Does one lead the other?** The best match between them is with **L4A in front by about 8
  years**, and the direction is robust: run three ways, L4A leads on the levels (correlation 0.76),
  on raw year-to-year changes (0.38), and on smoothed changes (0.36) — **all three peak at the same
  +8-year lag** (see `l4_leadlag_summary.csv`). The change-points tell the same story, 1876 for L4A
  against 1889 for L4B, a 13-year gap. The reading is that **the new business model leads, and the
  mission expansion follows.** The figure makes this plain: L4A climbs sharply from the late 1850s,
  and L4B only climbs after the late 1870s.

**Why this matters.** It supports keeping the two apart, exactly as you asked. And the direction is
interesting in its own right: Oxford first found new ways to *earn* (it became an investor and
began charging fees), and only afterward expanded what it *was for* (teaching and scholarship as a
purpose). The new value-capture preceded the mission shift, rather than the other way round.

*Caveat:* this rests on two short series, so we report the *direction* (L4A first) as well
supported across three transforms and the change-points, but the exact lag (roughly 8 to 13 years)
as approximate.

---

## 2. The pre-reform story, told in three phases

The professor's three questions are about **acceleration**, so we measure each dimension by its
trend (OLS slope) within three windows — before 1854, between the Acts (1854–77), and after 1877 —
not by a one-off level jump. Slopes are change in share per year (×100); **bold** = |t| ≥ 2.

| Dimension | Pre-1854 trend | 1854–77 trend | Post-1877 trend | Accelerates in |
|---|---|---|---|---|
| L1 cost | −0.16 (t −1.4) | +0.54 (t 1.3) | −0.04 (t −0.1) | — none |
| L2 personalisation | **+0.81 (t 3.0)** | +0.17 (t 2.2) | **+0.93 (t 2.7)** | Phase 1 & 3 |
| L3 operational | **+0.55 (t 2.0)** | +0.06 (t 0.1) | −0.37 (t −1.6) | Phase 1, then fades |
| L4A business model | −0.02 (t −0.3) | **+0.98 (t 2.4)** | −0.14 (t −0.3) | **Phase 2 (1854–77)** |
| L4B mission | +0.04 (t 1.4) | **+0.26 (t 4.1)** | **+0.99 (t 2.8)** | **Phase 3 (post-1877)** |

This answers the three questions directly:

- **Begin before 1854:** personalisation (L2) and the operating model (L3) — both already have a
  clear upward trend in the 1820s–1830s, a generation before Parliament acted.
- **Only accelerate after 1854:** the business model (L4A) — flat before 1854, then climbing steeply
  between the two Acts. It is the dimension the first reform most clearly switched on.
- **Accelerate again after 1877:** the mission (L4B) above all, with a second surge in
  personalisation (L2). The operating model (L3) instead peaks mid-century and then declines.

**Why this matters (and a correction to the earlier read).** Measuring acceleration rather than a
level step separates L4A from L4B in time: **the business model accelerates first (1854–77) and the
mission accelerates afterward (post-1877).** This is the same order the lead–lag test found in
Part 1, so the two analyses now reinforce each other. The earlier level-step table had lumped both
into "post-1877," which obscured exactly the sequence Part 1 is built on. The reforms confirmed and
scaled changes already under way rather than starting them.

**Reading the lower shapes against their definitions.**

- **L3 (admin share)** peaks at the reforms then recedes. This is a *share* effect, not a reversal:
  absolute real admin spend more than doubles into 1854–77 (≈£545 → £1,150/yr — the operating model
  is rebuilt), and the later decline reflects total spending growing faster as L4 scales. A one-time
  operational redesign looks exactly like this, which is why we read L3 as ratification, not growth.
- **L2 (individual awards)** is U-shaped: merit-award share ≈18% (pre-1854) → <2% (1854–77) → ≈14%
  (post-1877). The dip-and-recovery tracks the old endowed-award regime being replaced by the modern
  competitive scholarship/exhibition system: individual provision contracts during the reform, then
  re-expands on a reformed basis. The late rise is re-expansion, not new personalisation.

---

## 3. Sensitivity: do the conclusions survive the definitions?

For each level we computed three definitions (old, new, and an alternative) and re-checked three
conclusions: does the level carry the **late** change, does it **respond to the reforms**, and does
it have a **pre-1854 precursor**.

**The Level 4 conclusions are robust to the definition.**

| L4 definition | late minus early | reform response (1877) | precursor |
|---|---|---|---|
| old: educational spend | +0.12 | 1.25 | 1827 |
| new: new-revenue (L4A) | +0.29 | 1.34 | 1826 |
| alt: L4A + L4B | +0.40 | 1.33 | 1827 |

Under every definition, the top level grows late, responds strongly to 1877, and already has a
turning point in the late 1820s. **Our central claim does not depend on how we choose to measure
L4.** That is the result that makes the finding convincing.

The honest counterpoint is **L3**, which is the definition-sensitive level. The old salary measure
shows a strong reform response (1877 effect 2.08), but the new administrative measure does not
(-0.34), and whether L3 "carries the late change" flips sign between definitions. We flag L3 as the
construct that still needs care, rather than hiding it. The lower levels (L1, L2) show no consistent
late concentration under any definition, which is what we expect of them.

---

## 4. Bottom line

1. **L4 should stay split.** Business model and mission do not move together; the new business
   model (investment income, fees) leads, and the mission expansion follows by roughly a decade.
2. **The transformation has three phases**, not one shock: drift from the 1820s, ratified at 1854
   (operational), scaled after 1877 (business model and mission).
3. **The Level 4 conclusions survive every definition we tried**, which is exactly the robustness
   that strengthens the contribution. **Level 3 is the soft spot** and is flagged honestly.
4. Following the protocol, **resource slack is treated as an enabling mechanism, not a level.**

### Known construct issue to tighten
- **L2 currently overlaps L4B.** The personalisation proxy is coded as *educational-category spend
  OR award text*, but the educational-category component is exactly the L4B mission proxy, so the two
  series correlate 0.46 at the level. The protocol intends L2 to be *individually-named awards only*.
  Restricting L2 to award/person lines drops the L2↔L4B correlation to 0.23 and matches the
  protocol's own "keep them on opposite sides" discipline. This does **not** change the L4
  conclusions (Part 3), so we will adopt the sharper L2 definition in the next pass.

### For next discussion
- Confirm the L4A leads L4B reading with a longer lead-lag test (Granger-type) once we are
  comfortable with the two series; the current direction is robust across three transforms (all +8y)
  but the series are short.
- Decide whether L4B (mission) stays a named dimension in the paper or is folded into the
  discussion, since it has no direct AI counterpart.
