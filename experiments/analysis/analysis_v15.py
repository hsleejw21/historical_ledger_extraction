#!/usr/bin/env python
"""
analysis_v15.py — the Transformation Resource-Allocation analysis (Prof. Hu's V15 task).

Oxford is a rare case where we can watch BOTH sides of the resource-allocation problem: how the
college GENERATED resources (income) and how it DEPLOYED them (expenditure). We build two
portfolios on the revised L1-L4 definitions, study reallocation, and connect the two.

  Part 1  Revenue portfolio     : income share by L1-L4 per year (traditional vs new business).
  Part 2  Investment portfolio  : expenditure share by L1-L4 per year.
  Part 3  Reallocation          : "where did each additional pound go?" (marginal capture + shift).
  Part 4  Income<->expenditure  : did new revenue accompany higher-order spend? (association only).

Construct definitions follow CODING_PROTOCOL_v15.md. BOTH sides are cleaned symmetrically: aggregate
totals, opening/closing balances, carry-forwards, arrears and losses are accounting artefacts (not
real income or real deployment); one-off asset sales are capital. All excluded (see RX_ART/RX_CAPITAL).
On expenditure, only genuine investment purchases count as L4A (RX_INVEST); other financial-category
rows are internal transfers and fall in "other".

Outputs (this folder): revenue_portfolio.csv/png, investment_portfolio.csv/png, reallocation.csv/png,
                       income_expenditure_link.csv/png, leadlag.csv, first_appearance.csv
Run: cd experiments/reports/analysis_v15 && python analysis_v15.py
"""

from pathlib import Path
import sys, re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V13 = Path(__file__).resolve().parents[1] / "analysis_v13"
sys.path.insert(0, str(V13))
import build_proxies_v13 as bp          # load_entries, price_index, RX_SEC/RX_FEE/RX_AWARD
import finer_unit as fu                 # best_changepoint (emergence dating)

OUT = Path(__file__).resolve().parent
CUT1, CUT2 = 1854, 1877

# --- palette: L1->L4 are ORDINAL, so a sequential navy ramp (light=low, dark=high) + grey "other".
RAMP = {"L1":"#cdd9e5", "L2":"#93b0cc", "L3":"#4f77a3", "L4":"#1f3b5c"}
OTHER = "#d7dbdf"
NAVY, GREY, ORANGE, TEAL = "#1f3b5c", "#9aa4ad", "#b5530f", "#2b7a72"

# --- cleaning rules (documented in the protocol) ---------------------------
# ARTEFACTS: internal-accounting rows that are not real external income OR real deployment. These
# occur on BOTH sides of the ledger (receipt-side "total receipts", "arrears"; payment-side "total
# payments", "closing balance", "loss of guineas") and must be removed from income AND expenditure.
RX_ART = re.compile(
    r"total (?:receipt|payment|amount|sum|of all)|summary of|sum of dis|sum.{0,3}omnium"
    r"|opening (?:receipt|balance)|closing balance|balance (?:from|brought)"
    r"|brought (?:in|from|forward)|carried (?:forward|over)|deficit carried"
    r"|amount introduced|introduced/|computed from the sums|computed remaining"
    r"|arrear|debts?\b.{0,18}preceding|preceding year|loss (?:of|from|incurred)"
    r"|account rendered|unclear|no description|transcription", re.I)
# capital (one-off asset disposal), not a recurring stream.
RX_CAPITAL = re.compile(r"proceeds? from the sale|sale of (?:stock|investment|the)", re.I)
# genuine investment deployment (the only 'financial' expenditure that is business-model activity).
RX_INVEST = re.compile(r"purchase|consol|invest|\bstock\b|\bbond|annuit|\bshares?\b|debenture|exchequer", re.I)
# education-as-fee revenue signal (adds 'fee fund' to the v13 fee vocabulary).
RX_FEEINC = re.compile(r"fee fund|composition|tuition|caution|battels|matriculat|degree fee|entrance", re.I)


def _clean(frame):
    """Drop accounting artefacts and one-off capital items from either side of the ledger."""
    return frame[~(frame.text.str.contains(RX_ART) | frame.text.str.contains(RX_CAPITAL))].copy()


# --- classification layer ---------------------------------------------------

def classify_income(df, clean=True, fees_level="L4"):
    """Cleaned external income, one row per entry, tagged L1/L3/L4 (+ other; L2 income ~ nil).
    `clean` toggles the non-revenue/capital exclusion; `fees_level` lets the sensitivity test move
    fee income to another level."""
    inc = df[df.direction == "income"].copy()
    if clean:
        inc = _clean(inc)
    def lvl(r):
        c, t = r.category, r.text
        if c in ("land_rent", "ecclesiastical"):                         return "L1"        # core endowment
        if c == "educational" or RX_FEEINC.search(t):                    return fees_level  # education-as-fee
        if c == "financial" and bp.RX_SEC.search(t):                     return "L4"        # securities/annuity
        if c == "administrative":                                        return "L3"        # mgmt income (small)
        return "other"
    inc["lvl"] = inc.apply(lvl, axis=1)
    return inc


def classify_expenditure(df, salary_to="L1", clean=True, l4a_mode="invest"):
    """All expenditure, one row per entry, tagged L1/L2/L3/L4A/L4B (+ other). Cleaned symmetrically
    with income. `salary_to`, `clean`, `l4a_mode` are for the sensitivity test. L4A = *genuine
    investment deployment* only (l4a_mode='invest'); the discarded broad definition ('all_financial')
    counted every financial-category row, which double-counted totals and internal transfers."""
    exp = df[df.direction == "expenditure"].copy()
    if clean:
        exp = _clean(exp)
    def lvl(r):
        c, t = r.category, r.text
        if c == "educational":                                           # split award vs mission
            return "L2" if bp.RX_AWARD.search(t) else "L4B"
        if bp.RX_AWARD.search(t):                                        return "L2"        # individual awards
        if c == "salary_stipend":                                        return salary_to   # standing personnel
        if c in ("maintenance", "domestic", "ecclesiastical"):          return "L1"        # standing upkeep
        if c == "administrative":                                        return "L3"        # operating-model / coord
        if c == "financial":
            if l4a_mode == "all_financial":                             return "L4A"       # (old, broad)
            return "L4A" if RX_INVEST.search(t) else "other"                                # only real investment
        return "other"                                                                      # 'other', charitable, land_rent
    exp["lvl"] = exp.apply(lvl, axis=1)
    return exp


def yearly_shares(frame, levels):
    """Year x level share matrix (each row sums to 1 over the given levels + other)."""
    tot = frame.groupby("year").amount_real.sum()
    m = (frame.groupby(["year", "lvl"]).amount_real.sum().unstack(fill_value=0)
         .reindex(columns=levels, fill_value=0))
    return m.div(tot, axis=0).reindex(tot.index)


def smooth(s, w=10):
    return pd.Series(s).rolling(w, center=True, min_periods=4).mean()


# --- Part 1: revenue portfolio ---------------------------------------------

def part1_revenue(inc):
    lv = ["L1", "L2", "L3", "L4", "other"]
    sh = yearly_shares(inc.assign(lvl=inc.lvl.where(inc.lvl.isin(lv), "other")), lv).fillna(0)
    sh["L2"] = 0.0                                    # no clean individual-revenue signal (stated)
    sh = sh.reset_index().rename(columns={"index": "year"})
    sh.to_csv(OUT / "revenue_portfolio.csv", index=False)

    # classified-only view: L1/L2/L3/L4 renormalised excluding the unattributable "other" bucket,
    # so the "proportion of income from each level" is readable without the enrichment's residual.
    cls = inc[inc.lvl.isin(["L1", "L3", "L4"])]
    clt = cls.groupby("year").amount_real.sum()
    clm = (cls.groupby(["year", "lvl"]).amount_real.sum().unstack(fill_value=0)
           .reindex(columns=["L1", "L3", "L4"], fill_value=0).div(clt, axis=0))
    clm.insert(0, "L2", 0.0)
    clm.reset_index().to_csv(OUT / "revenue_classified.csv", index=False)

    # timing of new revenue: first appearance, first-material, and the EMERGENCE change-point
    # (a sustained shift, which ignores brief early blips that mislead 'first material').
    tot = inc.groupby("year").amount_real.sum()
    def timing(mask, thr=0.05):
        sub = inc[mask & (inc.amount_real > 0)]
        if sub.empty: return None, None, None
        ann = (sub.groupby("year").amount_real.sum() / tot).reindex(tot.index).fillna(0)
        fm = ann[ann > ann.max() * thr]
        cp = fu.best_changepoint(ann.index.values, ann.values, 1750, 1895)
        return int(sub.year.min()), (int(fm.index.min()) if len(fm) else None), cp[0]
    rows = []
    for name, mask in [("securities/annuity investment", (inc.category == "financial") & inc.text.str.contains(bp.RX_SEC)),
                       ("education-as-fee", (inc.category == "educational") | inc.text.str.contains(RX_FEEINC))]:
        fy, fm, em = timing(mask)
        rows.append(dict(signal=name, first_year=fy, first_material_year=fm, emergence_year=em))
    pd.DataFrame(rows).to_csv(OUT / "first_appearance.csv", index=False)

    _stack_plot(sh, ["L1", "L3", "L4"], "Revenue portfolio: how Oxford generated resources",
                OUT / "revenue_portfolio.png",
                labels={"L1": "L1 traditional (land, church)", "L3": "L3 admin income",
                        "L4": "L4 new business (fees, investment)"})
    return sh


# --- Part 2: investment portfolio ------------------------------------------

def part2_investment(exp):
    lv5 = ["L1", "L2", "L3", "L4A", "L4B", "other"]
    sh5 = yearly_shares(exp, lv5).fillna(0)
    sh = pd.DataFrame({"L1": sh5.L1, "L2": sh5.L2, "L3": sh5.L3,
                       "L4": sh5.L4A + sh5.L4B, "other": sh5.other})
    for c in ["L4A", "L4B"]:
        sh[c] = sh5[c]
    sh = sh.reset_index().rename(columns={"index": "year"})
    sh.to_csv(OUT / "investment_portfolio.csv", index=False)
    _stack_plot(sh, ["L1", "L2", "L3", "L4"], "Investment portfolio: how Oxford deployed resources",
                OUT / "investment_portfolio.png",
                labels={"L1": "L1 upkeep + personnel", "L2": "L2 individual awards",
                        "L3": "L3 administration", "L4": "L4 mission + business model"})
    return sh


def _stack_plot(sh, levels, title, path, labels):
    """Stacked-area share plot with a 2px surface gap between bands; direct labels at the right."""
    yr = sh.year.values
    order = levels + ["other"]
    cols = {**RAMP, "other": OTHER}
    S = {c: smooth(sh[c]).values for c in order}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    base = np.zeros(len(yr))
    for c in order:
        top = base + np.nan_to_num(S[c])
        ax.fill_between(yr, base, top, color=cols[c], linewidth=0, zorder=2)
        ax.plot(yr, top, color="white", lw=1.6, zorder=3)         # 2px surface gap
        base = top
    # direct labels at the last valid year
    base = np.zeros(len(yr))
    for c in order:
        v = np.nan_to_num(S[c]); mid = base + v/2; base = base + v
        j = np.where(~np.isnan(S[c]))[0]
        if len(j):
            k = j[-1]
            lab = labels.get(c, "other" if c == "other" else c)
            if v[k] > 0.03:
                ax.text(yr[k]+1.5, mid[k], lab, va="center", ha="left", fontsize=8,
                        color="#33404d" if c in ("L1", "other") else cols[c])
    for c in (CUT1, CUT2):
        ax.axvline(c, color="#5a6b7b", ls=":", lw=1, zorder=4)
    ax.text(CUT1, 1.02, "1854", color="#5a6b7b", fontsize=8, ha="center")
    ax.text(CUT2, 1.02, "1877", color="#5a6b7b", fontsize=8, ha="center")
    ax.set_xlim(1700, 1930); ax.set_ylim(0, 1.05); ax.set_xticks(range(1700, 1901, 50))
    ax.set_xlabel("Year"); ax.set_ylabel("share of total (10-yr smoothed)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --- Part 3: reallocation ("where did each additional pound go?") -----------

EARLY, LATE = (1780, 1853), (1854, 1900)

def part3_reallocation(exp):
    lv = ["L1", "L2", "L3", "L4A", "L4B", "other"]
    def period_real(lo, hi):
        s = exp[(exp.year >= lo) & (exp.year <= hi)]
        yrs = hi - lo + 1
        return s.groupby("lvl").amount_real.sum().reindex(lv, fill_value=0) / yrs   # £/yr
    e = period_real(*EARLY); l = period_real(*LATE)
    # collapse L4A+L4B -> L4 for the reallocation view
    def to4(x): return pd.Series({"L1": x.L1, "L2": x.L2, "L3": x.L3, "L4": x.L4A + x.L4B, "other": x.other})
    e4, l4 = to4(e), to4(l)
    delta = l4 - e4
    total_growth = delta.sum()
    marg = delta / total_growth                       # share of the marginal pound
    sh_e, sh_l = e4/e4.sum(), l4/l4.sum()
    out = pd.DataFrame({"early_real_per_yr": e4.round(1), "late_real_per_yr": l4.round(1),
                        "delta_per_yr": delta.round(1), "marginal_capture": marg.round(3),
                        "share_early": sh_e.round(3), "share_late": sh_l.round(3),
                        "pp_change": ((sh_l - sh_e)*100).round(1)})
    out.to_csv(OUT / "reallocation.csv")

    order = ["L1", "L2", "L3", "L4", "other"]
    cols = {**RAMP, "other": OTHER}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    # left: share of the marginal pound (each extra £ of late-vs-early spending)
    ax[0].bar(order, [marg[c]*100 for c in order], color=[cols[c] for c in order],
              edgecolor="white", linewidth=1.5)
    ax[0].axhline(0, color="black", lw=.8)
    ax[0].set_title("Where each additional pound went\n(share of total spending growth, %)", fontsize=10)
    ax[0].set_ylabel("% of the marginal pound")
    # right: gained / lost relative investment (percentage-point share change)
    pp = [(sh_l[c]-sh_e[c])*100 for c in order]
    ax[1].bar(order, pp, color=[cols[c] for c in order], edgecolor="white", linewidth=1.5)
    ax[1].axhline(0, color="black", lw=.8)
    ax[1].set_title("Gained vs lost relative investment\n(share change, early→late, pp)", fontsize=10)
    ax[1].set_ylabel("percentage points")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "reallocation.png", dpi=150); plt.close(fig)
    return out


# --- Part 4: connect income and expenditure --------------------------------

def _leadlag(a, b, lags=range(-12, 13)):
    cor = []
    for k in lags:
        x, y = a.shift(k), b
        m = x.notna() & y.notna()
        cor.append(stats.pearsonr(x[m], y[m])[0] if m.sum() > 5 else np.nan)
    i = int(np.nanargmax(np.abs(cor)))
    return list(lags)[i], round(cor[i], 2), round(cor[list(lags).index(0)], 2)


def part4_link(inc, exp):
    inc_lv = ["L1", "L2", "L3", "L4", "other"]
    exp_lv = ["L1", "L2", "L3", "L4A", "L4B", "other"]
    ish = yearly_shares(inc.assign(lvl=inc.lvl.where(inc.lvl.isin(inc_lv), "other")), inc_lv).fillna(0)
    esh = yearly_shares(exp, exp_lv).fillna(0)
    yrs = sorted(set(ish.index) & set(esh.index))
    d = pd.DataFrame(index=yrs)
    d["new_revenue"]   = ish.L4.reindex(yrs)                       # new-business income share
    d["higher_spend"]  = (esh.L3 + esh.L4A + esh.L4B).reindex(yrs) # L3+L4 expenditure share
    d["L4_spend"]      = (esh.L4A + esh.L4B).reindex(yrs)          # L4-only deployment (clean co-move)
    d["L4A_income"]    = ish.L4.reindex(yrs)                       # business-model revenue
    d["L4B_spend"]     = esh.L4B.reindex(yrs)                      # mission expenditure
    d["land_income"]   = ish.L1.reindex(yrs)                       # traditional income
    # income diversification: effective number of income sources (1/HHI) across the reported levels
    parts = ish[inc_lv].reindex(yrs).clip(lower=0)
    hhi = (parts.div(parts.sum(axis=1), axis=0)**2).sum(axis=1)
    d["income_diversity"] = 1.0/hhi
    d = d.reset_index().rename(columns={"index": "year"})
    d.to_csv(OUT / "income_expenditure_link.csv", index=False)

    # lead-lag on levels and on smoothed first-differences (does revenue lead spend?)
    ll = []
    for lab, a, b in [("new_revenue -> higher_spend (levels)", d.new_revenue, d.higher_spend),
                      ("new_revenue -> higher_spend (smoothed d)", smooth(d.new_revenue).diff(), smooth(d.higher_spend).diff()),
                      ("L4A_income -> L4B_spend (levels)", d.L4A_income, d.L4B_spend),
                      ("L4A_income -> L4B_spend (smoothed d)", smooth(d.L4A_income).diff(), smooth(d.L4B_spend).diff())]:
        lag, best, contemp = _leadlag(a, b)
        ll.append(dict(pair=lab, best_lag=lag, best_corr=best, contemp_corr=contemp,
                       verdict=("income leads" if lag > 0 else "spend leads" if lag < 0 else "contemporaneous")))
    pd.DataFrame(ll).to_csv(OUT / "leadlag.csv", index=False)

    # figure: two panels, single axis each (no dual-axis).
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    yr = d.year.values
    ax[0].plot(yr, smooth(d.new_revenue), color=ORANGE, lw=2.2, label="new-business income (revenue)")
    ax[0].plot(yr, smooth(d.L4_spend), color=NAVY, lw=2.2, ls=(0, (5, 2)),
               label="higher-order deployment (L4 spend)")
    ax[0].set_title("Do the two portfolios move together?", fontsize=10.5)
    ax[0].set_ylabel("share of total (smoothed)"); ax[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    def n01(x):
        x = smooth(x); return (x-x.min())/((x.max()-x.min()) or 1)
    ax[1].plot(yr, n01(d.L4A_income), color=ORANGE, lw=2.2, label="business-model revenue (L4A income)")
    ax[1].plot(yr, n01(d.L4B_spend), color=TEAL, lw=2.2, ls=(0, (5, 2)), label="mission spend (L4B expenditure)")
    ax[1].set_title("Business model first, mission after (normalised)", fontsize=10.5)
    ax[1].set_ylabel("normalised"); ax[1].legend(fontsize=8.5, frameon=False, loc="upper left")
    for a in ax:
        for c in (CUT1, CUT2): a.axvline(c, color="#5a6b7b", ls=":", lw=1)
        a.set_xlim(1700, 1900); a.set_xlabel("Year"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "income_expenditure_link.png", dpi=150); plt.close(fig)
    return d, pd.DataFrame(ll)


# --- Part 5: formal dating of the pivot (change-point + ITS) ----------------

def _level_share(frame, lvls):
    tot = frame.groupby("year").amount_real.sum()
    return (frame[frame.lvl.isin(lvls)].groupby("year").amount_real.sum() / tot).reindex(tot.index).fillna(0)

def part5_dating(inc, exp):
    """Date the shift to L4 formally, on both sides: change-point year + interrupted-time-series
    level shifts at the 1854 and 1877 reforms (segmented OLS, HAC errors) via the v13 tools."""
    rows = []
    for name, s in [("L4 income share", _level_share(inc, ["L4"])),
                    ("L4 expenditure share", _level_share(exp, ["L4A", "L4B"]))]:
        cp = fu.best_changepoint(s.index.values, s.values, 1800, 1895)
        it = fu.its(pd.DataFrame({"year": s.index, "y": s.values}))
        rows.append(dict(series=name, changepoint=cp[0], changepoint_t=round(cp[1], 1),
                         its_1854_norm=round(it["n1854"], 2), p_1854=round(it["p1854"], 3),
                         its_1877_norm=round(it["n1877"], 2), p_1877=round(it["p1877"], 3)))
    out = pd.DataFrame(rows); out.to_csv(OUT / "dating.csv", index=False)
    return out


# --- Part 6: sensitivity (do the conclusions survive the classification choices?) ---

def _late_share(frame, lvls, lo=1860, hi=1900):
    """Average yearly share of `lvls` over [lo,hi] — the SAME metric the portfolio tables use, so the
    sensitivity numbers line up with Sections 1 and 2 (no competing metrics)."""
    tot = frame.groupby("year").amount_real.sum()
    s = (frame[frame.lvl.isin(lvls)].groupby("year").amount_real.sum() / tot).reindex(tot.index).fillna(0)
    return round(s[(s.index >= lo) & (s.index <= hi)].mean() * 100, 1)

def part6_sensitivity(df):
    base_inc, base_exp = classify_income(df), classify_expenditure(df)
    # vary only the DEFINITIONAL choices; late window fixed at 1860-1900 to match the portfolios.
    variants = [
        ("Base", base_inc, base_exp),
        ("Salary to L3 (not L1)", base_inc, classify_expenditure(df, salary_to="L3")),
        ("Income not cleaned", classify_income(df, clean=False), base_exp),
        ("L4A = all financial (old, inflated)", base_inc, classify_expenditure(df, clean=False, l4a_mode="all_financial")),
        ("Fees counted as L2 (not L4)", classify_income(df, fees_level="L2"), base_exp),
    ]
    rows = []
    for name, inc, exp in variants:
        rows.append(dict(variant=name, late_L4_income_pct=_late_share(inc, ["L4"]),
                         late_L4_exp_pct=_late_share(exp, ["L4A", "L4B"])))
    out = pd.DataFrame(rows); out.to_csv(OUT / "sensitivity.csv", index=False)
    return out


# --- Part 7: synthesis figure (both portfolios, early vs late, at a glance) --

def part7_synthesis(inc, exp):
    def era_share(frame, lvls, lo, hi):
        f = frame[(frame.year >= lo) & (frame.year <= hi)]
        return [f[f.lvl.isin(v if isinstance(v, list) else [v])].amount_real.sum()/f.amount_real.sum()*100 for v in lvls]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(4); w = 0.38
    # income: L1, L3, L4, other
    inc_lv = [("L1", "L1"), ("L3", "L3"), ("L4", "L4"), ("other", "other")]
    ie = era_share(inc, [l[0] for l in inc_lv], 1780, 1853); il = era_share(inc, [l[0] for l in inc_lv], 1854, 1900)
    cols_i = [RAMP["L1"], RAMP["L3"], RAMP["L4"], OTHER]
    ax[0].bar(x-w/2, ie, w, color=cols_i, edgecolor="white", label="early (1780–1853)")
    ax[0].bar(x+w/2, il, w, color=cols_i, edgecolor="white", hatch="////", label="late (1854–1900)")
    ax[0].set_xticks(x); ax[0].set_xticklabels([l[1] for l in inc_lv]); ax[0].set_title("Revenue portfolio (income)", fontsize=10.5)
    ax[0].set_ylabel("share of total (%)")
    # expenditure: L1, L2, L3, L4(=L4A+L4B), other
    exp_lv = [("L1","L1"), ("L2","L2"), ("L3","L3"), (["L4A","L4B"],"L4"), ("other","other")]
    x2 = np.arange(5)
    ee = era_share(exp, [l[0] for l in exp_lv], 1780, 1853); el = era_share(exp, [l[0] for l in exp_lv], 1854, 1900)
    cols_e = [RAMP["L1"], RAMP["L2"], RAMP["L3"], RAMP["L4"], OTHER]
    ax[1].bar(x2-w/2, ee, w, color=cols_e, edgecolor="white")
    ax[1].bar(x2+w/2, el, w, color=cols_e, edgecolor="white", hatch="////")
    ax[1].set_xticks(x2); ax[1].set_xticklabels([l[1] for l in exp_lv]); ax[1].set_title("Investment portfolio (expenditure)", fontsize=10.5)
    ax[1].set_ylabel("share of total (%)")
    # a single legend explaining solid=early, hatch=late
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#c9ced3", edgecolor="white", label="early (1780–1853)"),
           Patch(facecolor="#c9ced3", edgecolor="white", hatch="////", label="late (1854–1900)")]
    ax[0].legend(handles=leg, fontsize=8.5, frameon=False, loc="upper right")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("How Oxford earned vs how it spent — early (solid) vs late (hatched)", fontsize=11.5, y=1.03)
    fig.tight_layout(); fig.savefig(OUT / "synthesis.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --- main -------------------------------------------------------------------

def main():
    df = bp.load_entries()
    inc = classify_income(df)
    exp = classify_expenditure(df)

    rev = part1_revenue(inc)
    invv = part2_investment(exp)
    realloc = part3_reallocation(exp)
    link, ll = part4_link(inc, exp)
    dating = part5_dating(inc, exp)
    sens = part6_sensitivity(df)
    part7_synthesis(inc, exp)

    def era_share(sh, cols, lo, hi):
        m = (sh.year >= lo) & (sh.year <= hi)
        return {c: round(sh.loc[m, c].mean()*100, 1) for c in cols}

    print("=== SANITY ===")
    rsum = rev[["L1", "L2", "L3", "L4", "other"]].sum(axis=1)
    isum = invv[["L1", "L2", "L3", "L4", "other"]].sum(axis=1)
    print(f"revenue shares/yr sum: {rsum.min():.3f}-{rsum.max():.3f} | invest shares/yr sum: {isum.min():.3f}-{isum.max():.3f}")
    print("\n=== Part 1 revenue portfolio (share % by era, of TOTAL income) ===")
    for lo, hi in [(1700, 1779), (1780, 1819), (1820, 1859), (1860, 1900)]:
        print(f"  {lo}-{hi}: {era_share(rev, ['L1','L3','L4','other'], lo, hi)}")
    revc = pd.read_csv(OUT / "revenue_classified.csv")
    print("  classified-only (excl. 'other', renormalised):")
    for lo, hi in [(1700, 1779), (1780, 1819), (1820, 1859), (1860, 1900)]:
        print(f"    {lo}-{hi}: {era_share(revc, ['L1','L3','L4'], lo, hi)}")
    print("\n=== Part 2 investment portfolio (share % by era) ===")
    for lo, hi in [(1700, 1779), (1780, 1819), (1820, 1859), (1860, 1900)]:
        print(f"  {lo}-{hi}: {era_share(invv, ['L1','L2','L3','L4','other'], lo, hi)}")
    print("\n=== Part 3 reallocation ==="); print(realloc.to_string())
    print("\n=== Part 4 lead-lag ==="); print(ll.to_string(index=False))
    print("\n=== Part 5 formal dating (change-point + ITS) ==="); print(dating.to_string(index=False))
    print("\n=== Part 6 sensitivity ==="); print(sens.to_string(index=False))
    print("\n=== first appearance ==="); print(pd.read_csv(OUT/'first_appearance.csv').to_string(index=False))
    # data coverage caveat: how many years/rows underpin each era
    print("\n=== data coverage (years covered / rows) ===")
    for lo, hi in [(1700, 1749), (1750, 1799), (1800, 1849), (1850, 1900)]:
        s = df[(df.year >= lo) & (df.year <= hi)]
        print(f"  {lo}-{hi}: {s.year.nunique()}/{hi-lo+1} yrs, {len(s)} rows")
    print("\nwrote revenue_portfolio.*, revenue_classified.csv, investment_portfolio.*, reallocation.*,")
    print("      income_expenditure_link.*, leadlag.csv, first_appearance.csv, dating.csv, sensitivity.csv, synthesis.png")


if __name__ == "__main__":
    main()
