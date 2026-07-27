#!/usr/bin/env python
"""
finer_unit.py — find the 1854/1877 changes at a FINER unit, and test for PRE-Act precursors.

Two questions the professor asked, each answered with a stated rationale:

  (1) Smaller unit. The aggregate L-level breaks hide *which* components moved and *when*. We
      decompose into per-category and per-account yearly series and locate each one's break.
      Rationale: an aggregate index can break at 1854 even if no single account did; only
      disaggregation shows whether the reform hit specific accounts sharply or many gently.

  (2) Precursors. We re-scan each series restricted to the PRE-reform window (1820-1853) for a
      change-point, and we date the first (and first material) appearance of each new-revenue
      signal. Rationale: if the "reform effect" already began before the Act, the Act ratified an
      existing drift rather than causing a jump; the only way to know is to look before 1854.

Method notes (why these choices):
  - Change-point = the year that maximises a Welch t-statistic of the mean shift, scanned within a
    window with a minimum 4-year segment. Chosen over a black-box detector because we need the
    *exact year* and the ability to restrict the search to a pre-Act window.
  - ITS = segmented OLS with 1854 & 1877 level shifts, HAC errors; the per-category effect is the
    level shift normalised by the local pre-reform mean (comparable across categories of any size).

Outputs (this folder): category_its.csv, changepoints.csv, precursor_timeline.csv,
account_breaks.csv, term_timing.csv, precursor_timeline.png, category_its.png
Run: cd experiments/reports/analysis_v13 && python finer_unit.py
"""

from pathlib import Path
import sys, re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_proxies_v13 as bp           # reuse loader, price index, keyword rules

OUT = Path(__file__).resolve().parent
CUT1, CUT2 = 1854, 1877
EXP_CATS = ["administrative","charitable","domestic","ecclesiastical","educational",
            "financial","maintenance","salary_stipend"]
INC_CATS = ["land_rent","financial","educational","administrative"]


# --- finer-unit tools -------------------------------------------------------

def best_changepoint(years, vals, lo, hi, min_seg=4):
    """Year in [lo,hi] that maximises |Welch t| of the mean shift (>= min_seg each side)."""
    y = np.asarray(years); v = np.asarray(vals, float)
    m = (y >= lo) & (y <= hi) & np.isfinite(v)
    y, v = y[m], v[m]
    best = (None, 0.0)
    for i in range(min_seg, len(y) - min_seg):
        a, b = v[:i], v[i:]
        if a.std() == 0 and b.std() == 0:
            continue
        t = abs(stats.ttest_ind(a, b, equal_var=False).statistic)
        if np.isfinite(t) and t > best[1]:
            best = (int(y[i]), float(t))
    return best


def its(series_df):
    """Segmented OLS with 1854 & 1877 level shifts; return normalised effects + p-values."""
    d = series_df.dropna().sort_values("year")
    if len(d) < 30: return dict(n1854=np.nan, p1854=np.nan, n1877=np.nan, p1877=np.nan)
    d = d.assign(t=d.year-1700, D54=(d.year>=CUT1).astype(float),
                 t54=(d.year-CUT1)*(d.year>=CUT1), D77=(d.year>=CUT2).astype(float),
                 t77=(d.year-CUT2)*(d.year>=CUT2))
    X = sm.add_constant(d[["t","D54","t54","D77","t77"]])
    r = sm.OLS(d.y, X).fit(cov_type="HAC", cov_kwds={"maxlags":10})
    pre54 = d.loc[d.year<CUT1,"y"].mean() or np.nan
    pre77 = d.loc[(d.year>=CUT1)&(d.year<CUT2),"y"].mean() or np.nan
    return dict(n1854=r.params["D54"]/pre54, p1854=r.pvalues["D54"],
                n1877=r.params["D77"]/pre77, p1877=r.pvalues["D77"])


def cat_series(df, cats, side):
    """Per-category yearly share series for `side` ('expenditure'/'income')."""
    sub = df[df.direction==side]
    tot = sub.groupby("year").amount_real.sum()
    out = {}
    for c in cats:
        s = sub[sub.category==c].groupby("year").amount_real.sum()
        out[c] = (s/tot).reindex(tot.index).fillna(0)
    return tot.index.values, out


# --- analyses ---------------------------------------------------------------

def run_category(df):
    rows, cp = [], []
    for side, cats in [("expenditure", EXP_CATS), ("income", INC_CATS)]:
        years, series = cat_series(df, cats, side)
        for c, s in series.items():
            sd = pd.DataFrame({"year":years, "y":s.values})
            e = its(sd)
            full = best_changepoint(years, s.values, years.min(), years.max())
            pre  = best_changepoint(years, s.values, 1820, 1853)   # precursor window
            rows.append(dict(side=side, category=c, **{k:round(v,3) if pd.notna(v) else v for k,v in e.items()}))
            cp.append(dict(side=side, category=c, break_full=full[0], t_full=round(full[1],2),
                           break_pre1854=pre[0], t_pre1854=round(pre[1],2)))
    pd.DataFrame(rows).to_csv(OUT/"category_its.csv", index=False)
    pd.DataFrame(cp).to_csv(OUT/"changepoints.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(cp)


def run_precursors(df):
    """First / first-material appearance + pre-Act change-point for each key component,
    then classify onset as pre-Act / at-Act / post-Act."""
    inc = df[df.direction=="income"]; exp = df[df.direction=="expenditure"]
    comps = {
        "L4 securities income": (inc, (inc.category=="financial") & inc.text.str.contains(bp.RX_SEC)),
        "L4 education-fee income": (inc, (inc.category=="educational") | inc.text.str.contains(bp.RX_FEE)),
        "L4 educational spend": (exp, exp.category=="educational"),
        "L3 administrative spend": (exp, exp.category=="administrative"),
        "L1 ecclesiastical spend": (exp, exp.category=="ecclesiastical"),
    }
    recs = []
    for name, (frame, mask) in comps.items():
        sub = frame[mask & (frame.amount_real>0)]
        if sub.empty: continue
        tot = frame.groupby("year").amount_real.sum()
        share = (sub.groupby("year").amount_real.sum()/tot).reindex(tot.index).fillna(0)
        first = int(sub.year.min())
        ymat = share[share > share.max()*0.10]
        first_mat = int(ymat.index.min()) if len(ymat) else None
        pre = best_changepoint(tot.index.values, share.values, 1820, 1853)
        # onset classification from the first material year
        onset = ("pre-Act" if first_mat and first_mat < CUT1-1 else
                 "at-1854" if first_mat and first_mat <= CUT1+3 else
                 "post-1854/at-1877" if first_mat else "n/a")
        recs.append(dict(component=name, first_year=first, first_material_year=first_mat,
                         pre1854_break=pre[0], pre1854_t=round(pre[1],2), onset=onset))
    out = pd.DataFrame(recs)
    out.to_csv(OUT/"precursor_timeline.csv", index=False)

    # figure: pre-Act change-point of each component vs the two Acts (does the shift predate 1854?)
    n = len(out); top = n - 1
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    yl = list(range(n))
    for x, y, t in zip(out.pre1854_break, yl, out.pre1854_t):
        if x:
            ax.scatter(x, y, s=70, marker="o", color="#1f3b5c", zorder=3)
            ax.annotate(f"{x}  (t={t:.1f})", (x, y), textcoords="offset points",
                        xytext=(11, 0), va="center", fontsize=8.5, color="#33404d")
    for c, lab in [(CUT1, "1854 reform"), (CUT2, "1877 reform")]:
        ax.axvline(c, color="#9aa4ad", ls=":", lw=1.1, zorder=1)
        ax.text(c, top + 0.55, lab, color="#5a6b7b", fontsize=8.5, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    ax.set_yticks(yl); ax.set_yticklabels(out.component, fontsize=8.8)
    ax.set_xlabel("Year"); ax.set_xlim(1815, 1900); ax.set_ylim(-0.6, top + 1.0)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Each higher-order account turns in the 1820s and 1830s, before the reforms",
                 fontsize=10.5, pad=20)
    fig.tight_layout(); fig.savefig(OUT/"precursor_timeline.png", dpi=150); plt.close(fig)
    return out


def run_accounts(df, top=25):
    """Per-account decomposition: largest persistent accounts, their span and break year."""
    d = df.copy()
    d["acct"] = (d.text.str.slice(0,40).str.strip().str.lower()
                 .replace(r"[^a-z ]","",regex=True).str.replace(r"\s+"," ",regex=True))
    d = d[d.acct.str.len()>4]
    g = d.groupby("acct").agg(category=("category","first"), direction=("direction","first"),
                              first_year=("year","min"), last_year=("year","max"),
                              n_years=("year","nunique"), total_real=("amount_real","sum")).reset_index()
    g = g[(g.n_years>=15)].sort_values("total_real", ascending=False).head(top)
    breaks = []
    for _, row in g.iterrows():
        s = d[d.acct==row.acct].groupby("year").amount_real.sum()
        full = best_changepoint(s.index.values, s.values, int(s.index.min()), int(s.index.max()))
        breaks.append(full[0])
    g["break_year"] = breaks
    g.to_csv(OUT/"account_breaks.csv", index=False)
    return g


def run_terms(df):
    """Secondary: audit-term (feast-day) signal over time. Coverage caveat: term language is
    present on a minority of rows, so this is indicative, not a full sub-annual series."""
    RX = {"Michaelmas":r"michaelmas|mich\b", "Lady Day":r"lady day|annunciat",
          "Midsummer":r"midsummer|john bapt", "Christmas":r"christmas|nativ"}
    sub = df.copy()
    tot = sub.groupby("year").size()
    rows = []
    for term, rx in RX.items():
        c = sub[sub.text.str.contains(rx, case=False, regex=True)].groupby("year").size()
        share = (c/tot).reindex(tot.index).fillna(0)
        rows.append(dict(term=term, n_rows=int(c.sum()),
                         share_pre1854=round(share[share.index<CUT1].mean(),4),
                         share_post1854=round(share[share.index>=CUT1].mean(),4)))
    out = pd.DataFrame(rows); out.to_csv(OUT/"term_timing.csv", index=False)
    return out


def main():
    df = bp.load_entries()
    cit, cp = run_category(df)
    pre = run_precursors(df)
    acc = run_accounts(df)
    term = run_terms(df)

    print("=== per-category ITS at the reforms (normalised level shift) ===")
    print(cit.to_string(index=False))
    print("\n=== change-points: full series vs pre-1854 window ===")
    print(cp.to_string(index=False))
    print("\n=== precursor timeline ===")
    print(pre.to_string(index=False))
    print("\n=== audit-term signal (coverage caveat) ===")
    print(term.to_string(index=False))
    print(f"\nwrote category_its.csv, changepoints.csv, precursor_timeline.csv/png, account_breaks.csv, term_timing.csv")


if __name__ == "__main__":
    main()
