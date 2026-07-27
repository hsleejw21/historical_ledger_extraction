#!/usr/bin/env python
"""
analysis_v14.py — Jungwoo's tasks from the professor's feedback.

  Part 1  L4 boundary: split L4A (business-model / new revenue) from L4B (mission), and test
          whether they move together, which appears first, and which leads.
  Part 2  Pre-reform story: the professor asked about *acceleration*, so for each dimension we fit
          the trend (slope) within three windows -- pre-1854, 1854-77, post-1877 -- and read which
          phase it accelerates in. Answers: which begin before 1854, which only accelerate after
          1854, which accelerate again after 1877.
  Part 3  Sensitivity: compute old / new / alternative definitions of each level and check whether
          the main conclusions (higher levels carry the late change; they respond to the reforms;
          they have a pre-1854 precursor) survive the definition choice.

Construct definitions follow CODING_PROTOCOL.md. Reuses the v13 loader and tools.

Outputs (this folder): l4_split.csv, l4_leadlag.csv, l4_split.png,
                       phases.csv, phases.png, sensitivity.csv, sensitivity.png
Run: cd experiments/reports/analysis_v14 && python analysis_v14.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V13 = Path(__file__).resolve().parents[1] / "analysis_v13"
sys.path.insert(0, str(V13))
import build_proxies_v13 as bp          # loader, price index, keyword rules (RX_SEC/RX_FEE/RX_AWARD)
import finer_unit as fu                 # best_changepoint, its

OUT = Path(__file__).resolve().parent
CUT1, CUT2 = 1854, 1877
NAVY, GREY, ORANGE = "#1f3b5c", "#9aa4ad", "#b5530f"


# --- series builders (each one proxy from the protocol) ---------------------

def build_series(df):
    """Return a year-indexed DataFrame with every level series we need."""
    exp = df[df.direction == "expenditure"]; inc = df[df.direction == "income"]
    texp = exp.groupby("year").amount_real.sum()
    tinc = inc.groupby("year").amount_real.sum()
    yrs = texp.index

    def esh(mask):  # expenditure share
        return (exp[mask].groupby("year").amount_real.sum() / texp).reindex(yrs).fillna(0)
    def ish(mask):  # income share
        return (inc[mask].groupby("year").amount_real.sum() / tinc).reindex(yrs).fillna(0)

    s = pd.DataFrame(index=yrs)
    # protocol proxies (new definitions)
    s["L1"]  = esh(exp.category.isin({"maintenance","domestic"}))
    s["L2"]  = esh((exp.category=="educational") | exp.text.str.contains(bp.RX_AWARD))
    s["L3"]  = esh(exp.category=="administrative")
    s["L4A"] = ish(((inc.category=="financial") & inc.text.str.contains(bp.RX_SEC))
                   | (inc.category=="educational") | inc.text.str.contains(bp.RX_FEE))
    s["L4B"] = esh(exp.category=="educational")
    # extra series used only by the sensitivity variants
    s["trad"]      = esh(exp.category.isin({"ecclesiastical","maintenance","domestic"}))
    s["eccl"]      = esh(exp.category=="ecclesiastical")
    s["salary"]    = esh(exp.category=="salary_stipend")
    s["edu_exp"]   = s["L4B"]
    s["sec_inc"]   = ish((inc.category=="financial") & inc.text.str.contains(bp.RX_SEC))
    s["person_den"]= (exp.assign(p=exp.has_person.astype(float)).groupby("year").p.mean()).reindex(yrs).fillna(0)
    # payment-modernity (old L2)
    pm = exp[exp.payment_period.isin(bp.PAYMENT_SCORES)].copy()
    pm["sc"] = pm.payment_period.map(bp.PAYMENT_SCORES)
    s["pay_mod"] = ((pm.sc*pm.amount_real).groupby(pm.year).sum()
                    / pm.groupby("year").amount_real.sum()).reindex(yrs).fillna(np.nan)
    return s.reset_index().rename(columns={"index":"year"})


def smooth(s, w=10):
    return pd.Series(s).rolling(w, center=True, min_periods=4).mean()


# --- Part 1: L4A vs L4B -----------------------------------------------------

def part1_l4(s):
    d = s[["year","L4A","L4B"]].copy()
    d.to_csv(OUT/"l4_split.csv", index=False)

    # first material year (>10% of own max), change-point, and lead-lag on first differences
    def first_material(col):
        v = smooth(s[col]); thr = v.max()*0.10
        m = s.year[v > thr]
        return int(m.min()) if len(m) else None
    a_break = fu.best_changepoint(s.year.values, s.L4A.values, 1800, 1900)
    b_break = fu.best_changepoint(s.year.values, s.L4B.values, 1800, 1900)

    # Lead-lag is run on THREE transforms so the direction does not rest on the smoothing choice:
    #   levels (trended, so high baseline corr), raw first differences (no smoothing), and
    #   smoothed first differences. Reporting all three is the robustness the report cites.
    lags = list(range(-10,11))          # +-10 yr; wider ranges invite edge artifacts
    def leadlag(a, b):
        cor = []
        for k in lags:
            x, y = a.shift(k), b
            m = x.notna() & y.notna()
            cor.append(stats.pearsonr(x[m], y[m])[0] if m.sum()>5 else np.nan)
        best = int(np.nanargmax(np.abs(cor)))
        return cor, lags[best], cor[best], cor[lags.index(0)]
    transforms = {
        "levels":          (s.L4A,                  s.L4B),
        "raw_diffs":       (s.L4A.diff(),           s.L4B.diff()),
        "smoothed_diffs":  (smooth(s.L4A).diff(),   smooth(s.L4B).diff()),
    }
    ll_rows, corrs_for_csv = [], {}
    for name, (a, b) in transforms.items():
        cor, blag, bcorr, contemp0 = leadlag(a, b)
        corrs_for_csv[name] = cor
        ll_rows.append(dict(transform=name, best_lag=blag,
                            best_corr=round(bcorr,2), contemp_corr=round(contemp0,2),
                            verdict=("L4A leads L4B" if blag>0 else "L4B leads L4A" if blag<0
                                     else "contemporaneous")))
    # positive lag = L4A shifted forward matches L4B => L4A leads
    corrs = corrs_for_csv["smoothed_diffs"]
    best_lag = ll_rows[2]["best_lag"]; contemp = corrs[lags.index(0)]
    pd.DataFrame({"lag":lags, **corrs_for_csv}).to_csv(OUT/"l4_leadlag.csv", index=False)
    pd.DataFrame(ll_rows).to_csv(OUT/"l4_leadlag_summary.csv", index=False)

    leads = "L4A leads L4B" if best_lag>0 else "L4B leads L4A" if best_lag<0 else "contemporaneous"
    summary = dict(L4A_first_material=first_material("L4A"), L4B_first_material=first_material("L4B"),
                   L4A_break=a_break[0], L4B_break=b_break[0],
                   contemp_corr=round(contemp,2), best_lag=best_lag, best_corr=round(corrs[best_lag+10],2),
                   verdict=leads,
                   leadlag_by_transform={r["transform"]: (r["best_lag"], r["best_corr"]) for r in ll_rows})

    fig, ax = plt.subplots(figsize=(8.5,3.8))
    def n01(x):
        x = smooth(x); return (x-x.min())/((x.max()-x.min()) or 1)
    ax.plot(s.year, n01(s.L4A), color=NAVY, lw=2.0, label="L4A  business model (new revenue)")
    ax.plot(s.year, n01(s.L4B), color=ORANGE, lw=2.0, ls=(0,(5,2)), label="L4B  mission (educational purpose)")
    for c in (CUT1,CUT2): ax.axvline(c, color=GREY, ls=":", lw=1)
    ax.text(CUT1, 1.02, "1854", color="#5a6b7b", fontsize=8, ha="center")
    ax.text(CUT2, 1.02, "1877", color="#5a6b7b", fontsize=8, ha="center")
    ax.set_xlim(1700,1900); ax.set_ylim(-0.05,1.12); ax.set_xlabel("Year")
    ax.set_ylabel("normalised (10-yr)"); ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title("L4A (business model) vs L4B (mission): do they move together?", fontsize=10.5)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT/"l4_split.png", dpi=150); plt.close(fig)
    return summary


# --- Part 2: phases ---------------------------------------------------------

DIMS = [("L1","L1 cost"),("L2","L2 personalisation"),("L3","L3 operational"),
        ("L4A","L4A business model"),("L4B","L4B mission")]
# The professor asked about *acceleration* (does the trend get steeper?), not a one-off level jump.
# We therefore fit the slope of each series within each phase window and read the phase from where
# the upward acceleration is strongest. Windows: Phase 1 pre-reform, Phase 2 between the Acts,
# Phase 3 after the second Act.
WINDOWS = [("pre1854", 1820, 1853), ("mid_1854_77", 1854, 1877), ("post1877", 1878, 1900)]


def _slope(s, col, lo, hi):
    """OLS slope (share per year) and t-stat of `col` within [lo,hi]."""
    d = pd.DataFrame({"year": s.year, "y": s[col]}).dropna()
    d = d[(d.year >= lo) & (d.year <= hi)]
    if len(d) < 4:
        return np.nan, np.nan
    r = sm.OLS(d.y, sm.add_constant(d[["year"]])).fit()
    return float(r.params["year"]), float(r.tvalues["year"])


def part2_phases(s):
    recs = []
    for col, label in DIMS:
        onset = fu.best_changepoint(s.year.values, s[col].values, 1820, 1853)  # first-move year
        sl = {name: _slope(s, col, lo, hi) for name, lo, hi in WINDOWS}
        # the phase in which the dimension accelerates = window with the strongest *significant*
        # positive slope (t>=2). 'none' => no clear acceleration in any window (improvement, not change).
        sig = {k: v[0] for k, v in sl.items() if pd.notna(v[1]) and v[1] >= 2.0 and v[0] > 0}
        accel = max(sig, key=sig.get) if sig else "none"
        recs.append(dict(dimension=label, onset_year=onset[0], onset_t=round(onset[1], 1),
                         pre1854_slope=round(sl["pre1854"][0]*100, 3),    pre1854_t=round(sl["pre1854"][1], 1),
                         mid_slope=round(sl["mid_1854_77"][0]*100, 3),    mid_t=round(sl["mid_1854_77"][1], 1),
                         post1877_slope=round(sl["post1877"][0]*100, 3),  post1877_t=round(sl["post1877"][1], 1),
                         accelerates_in=accel))
    out = pd.DataFrame(recs); out.to_csv(OUT/"phases.csv", index=False)

    # figure: each dimension with the three phase bands shaded, so the reader sees *when* it climbs.
    bands = {"pre1854": (1800, 1854, "#eef2f6"), "mid_1854_77": (1854, 1877, "#fbeede"),
             "post1877": (1877, 1900, "#e9f0ea")}
    fig, axes = plt.subplots(1, len(DIMS), figsize=(13, 3.1), sharey=False)
    for ax, (col, label) in zip(axes, DIMS):
        v = smooth(s[col]); yv = (v-v.min())/((v.max()-v.min()) or 1)
        for lo, hi, cpal in bands.values():
            ax.axvspan(lo, hi, color=cpal, zorder=0)
        ax.plot(s.year, yv, color=NAVY, lw=1.8, zorder=2)
        onset = fu.best_changepoint(s.year.values, s[col].values, 1820, 1853)
        if onset[0] and onset[1] >= 2.0:
            ax.scatter(onset[0], yv[s.year == onset[0]].values[0], s=40, color=ORANGE, zorder=3)
        for c in (CUT1, CUT2): ax.axvline(c, color=GREY, ls=":", lw=0.9, zorder=1)
        ax.set_title(label, fontsize=9); ax.set_xlim(1800, 1900); ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([]); ax.tick_params(labelsize=8); ax.spines[["top","right","left"]].set_visible(False)
    axes[0].set_ylabel("normalised", fontsize=8)
    fig.suptitle("When each dimension accelerates  —  bands: Phase 1 (pre-1854) · Phase 2 (1854–77) · "
                 "Phase 3 (post-1877); orange dot = pre-1854 onset", fontsize=9.5, y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"phases.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# --- Part 3: sensitivity ----------------------------------------------------

def part3_sensitivity(s):
    # each level: (variant label, series column, higher-or-lower expectation)
    variants = {
      "L1": [("old: 1-traditional", 1-s.trad), ("new: maint+domestic", s.L1), ("alt: ecclesiastical", s.eccl)],
      "L2": [("old: payment-modernity", s.pay_mod), ("new: individual awards", s.L2), ("alt: person density", s.person_den)],
      "L3": [("old: salary share", s.salary), ("new: admin share", s.L3), ("alt: salary+admin", s.salary+s.L3)],
      "L4": [("old: educational spend", s.L4B), ("new: new-revenue (L4A)", s.L4A), ("alt: L4A+L4B", s.L4A+s.L4B)],
    }
    rows = []
    for lvl, vs in variants.items():
        for name, series in vs:
            sd = pd.DataFrame({"year":s.year, "y":series.values})
            v = smooth(series)
            late = v[(s.year>=1860)&(s.year<=1900)].mean()
            early = v[(s.year>=1820)&(s.year<=1859)].mean()
            it = fu.its(sd)
            pre = fu.best_changepoint(s.year.values, series.values, 1820, 1853)
            rows.append(dict(level=lvl, definition=name,
                             late_minus_early=round(late-early,3),
                             its_1877_norm=round(it["n1877"],2) if pd.notna(it["n1877"]) else np.nan,
                             precursor_year=pre[0], precursor_t=round(pre[1],1)))
    out = pd.DataFrame(rows); out.to_csv(OUT/"sensitivity.csv", index=False)

    # figure: late-minus-early per level across the three definitions
    fig, ax = plt.subplots(figsize=(9,3.8))
    levels = ["L1","L2","L3","L4"]; x = np.arange(len(levels)); w=0.26
    cols = {0:GREY,1:NAVY,2:ORANGE}; labs = {0:"old",1:"new",2:"alt"}
    for i in range(3):
        vals = [out[(out.level==L)].iloc[i].late_minus_early for L in levels]
        ax.bar(x+(i-1)*w, vals, w, color=cols[i], label=labs[i], alpha=.9)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(levels); ax.set_ylabel("late minus early (share)")
    ax.legend(frameon=False, fontsize=9, title="definition")
    ax.set_title("Sensitivity: does 'higher levels carry the late change' survive the definition?", fontsize=10.5)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT/"sensitivity.png", dpi=150); plt.close(fig)
    return out


def main():
    df = bp.load_entries()
    s = build_series(df)
    l4 = part1_l4(s)
    ph = part2_phases(s)
    se = part3_sensitivity(s)

    print("=== Part 1: L4A vs L4B ===")
    for k,v in l4.items(): print(f"  {k}: {v}")
    print("\n=== Part 2: phases ==="); print(ph.to_string(index=False))
    print("\n=== Part 3: sensitivity ==="); print(se.to_string(index=False))
    print("\nwrote l4_split.csv/png, l4_leadlag.csv, phases.csv/png, sensitivity.csv/png")


if __name__ == "__main__":
    main()
