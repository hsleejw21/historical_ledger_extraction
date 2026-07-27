#!/usr/bin/env python
"""
analysis_v18.py -- DISCRIMINATING AMONG COMPETING MECHANISMS (Prof. Hu's V18 task).

Where we are. V16 dated the change: the coupling between what Oxford earned and what it spent
breaks around 1820 (adjustment speed lambda 0.95 -> 0.16). V17 proposed a mechanism: the budgeting
rule changed from "match spending to this year's receipts" to "hold spending on a smoothed path."

The professor's instruction for V18 is to STOP CONFIRMING that story and start trying to KILL it.
The budgeting rule is now treated as ONE CANDIDATE AMONG SEVERAL. For every candidate we ask the
same three questions:

    (i)   what would we OBSERVE if this mechanism were true?
    (ii)  can we test that implication in this archive?
    (iii) does the evidence support it, contradict it, or fail to speak?

The six candidates on the table (the professor's list):
    M1  DECISION RULE      spending moved from a current-receipts rule to a planned/permanent rule
    M2  DIVERSIFICATION    income simply became a lumpier portfolio; the rule never changed
    M3  GOVERNANCE         who authorises spending changed
    M4  PLANNING HORIZON   commitments lengthened, so spending mechanically stopped tracking the year
    M5  FINANCIAL MGMT     real reserves/securities appeared and absorbed the swings
    M6  ACCOUNTING         nothing organisational changed; the ledger changed how it records

The tests, in the order the professor asked for them:

    T1  CURRENT vs PERMANENT INCOME   is spending predicted by this year's income before the break
                                      and by multi-year income after it?                        (M1)
    T2  INCOME-SHOCK RESPONSE         does spending stop responding to TRANSITORY income while
                                      still responding to PERMANENT income; are positive and
                                      negative shocks treated differently?                      (M1 vs M5)
    T3  EXCESS SMOOTHNESS             is spending smoother than income *by more than the change in
                                      income's own composition can explain*?                    (M1 vs M2)
        T3b COMPOSITION PLACEBO       simulate the UNCHANGED pre-1820 rule driven by the ACTUAL
                                      post-1820 income. If the placebo reproduces the observed
                                      smoothing, M2 wins and M1 is unnecessary.
    T4  BREAK-DATE HORSE RACE         each candidate leaves a datable trace. Whose trace breaks at
                                      1820, and whose breaks at 1854/1870 (too late to be a cause)?
    T5  ABSORPTION HORSE RACE         put each candidate's proxy INTO the error-correction model as
                                      an interaction. Does any of them absorb the post-1820 collapse
                                      in the adjustment speed?
    T6  VERDICT TABLE                 implication / testable? / evidence / verdict, one row each.

Classification, cleaning, deflation, the window and the interpolation of interior missing years all
reuse v15/v16/v17 unchanged, so the numbers speak the same language as the previous three weeks.

Run: cd experiments/reports/analysis_v18 && python3 analysis_v18.py
"""

from pathlib import Path
import sys, json, re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
ROOT = REPORTS.parents[1]
ENRICHED = ROOT / "experiments/results/enriched"
for p in (REPORTS / "analysis_v13", REPORTS / "analysis_v15",
          REPORTS / "analysis_v16", REPORTS / "analysis_v17"):
    sys.path.insert(0, str(p))
import build_proxies_v13 as bp
import analysis_v15 as v15
import analysis_v17 as v17

OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

Y0, Y1 = 1800, 1900
BREAK, REFORM, LEVEL_BREAK = 1820, 1854, 1870
K_PERM = 5                       # years in the "permanent income" window (headline; 3 and 7 in robustness)
SEED = 7

NAVY, ORANGE, TEAL, GREY, RED = "#1f3b5c", "#b5530f", "#2b7a72", "#9aa4ad", "#8c2f39"

ERAS3 = [("1800-1819 (tight)", 1800, 1819),
         ("1820-1853 (loose, pre-reform)", 1820, 1853),
         ("1854-1900 (loose, post-reform)", 1854, 1900)]


# ===========================================================================
# 0. SERIES
# ===========================================================================

def build():
    """v17's annual panel, plus the income series with the two known bad extraction years repaired.

    v17 flagged 1859 and 1862 as incomplete INCOME extractions (1859 records GBP 32 of income against
    GBP 8,353 of spending on 15 income rows). Left in, they dominate every volatility and shock
    statistic in this file.

    The default here MASKS them (drops those years from every estimate) rather than interpolating
    them, because interpolation invents an income path that was never recorded and, as T0 shows, that
    invented path is not innocent -- it moves the break-date scan. Masking neither keeps known-bad
    data nor manufactures new data. T0 reports all three treatments side by side."""
    d = v17.build_series()
    bad = d["income_incomplete"].fillna(False)
    d["income_clean"] = d["income"].mask(bad)                 # DEFAULT: masked
    inc_i = d["income"].mask(bad)
    inc_i.loc[Y0:Y1] = inc_i.loc[Y0:Y1].interpolate(limit_area="inside")
    d["income_interp"] = inc_i                                # kept only for the T0 comparison
    d["n_bad"] = bad.astype(int)

    # permanent income = trailing geometric mean of the last K years (information available at t)
    for k in (3, 5, 7):
        d[f"perm{k}"] = np.exp(np.log(d["income_clean"]).rolling(k, min_periods=k).mean())
    d["perm"] = d[f"perm{K_PERM}"]
    d["transitory"] = np.log(d["income_clean"]) - np.log(d["perm"])   # deviation from permanent
    return d


def load_page_meta():
    """Per-year ledger-form and governance markers, read straight off the enriched pages.

    These are the observable traces of the NON-budgeting candidates: who is named (governance),
    how the page is organised and how much Latin survives (accounting practice), how many pages and
    rows there are per year (recording density)."""
    rows = []
    for fp in sorted(ENRICHED.glob("*_enriched.json")):
        d = json.loads(fp.read_text())
        tok = fp.name.split("_")[0]
        if not tok.isdigit():
            continue
        y = int(tok)
        if not (1600 < y < 2000):
            continue
        rr = [r for r in d.get("rows", []) if isinstance(r, dict)]
        if not rr:
            continue
        rt = [str(r.get("row_type", "")).lower() for r in rr]
        rows.append(dict(
            year=y, pages=1, n_rows=len(rr),
            n_entry=rt.count("entry"), n_total=rt.count("total"),
            n_sections=len({r.get("section_header") for r in rr if r.get("section_header")}),
            n_sig=sum(1 for r in rr if r.get("is_signature")),
            n_person=sum(1 for r in rr if r.get("person_name")),
            n_latin=sum(1 for r in rr if str(r.get("language")) in ("latin", "mixed")),
            n_lang=sum(1 for r in rr if r.get("language")),
            n_arrears=sum(1 for r in rr if r.get("is_arrears")),
            persons=len({str(r.get("person_name")).strip().lower()
                         for r in rr if r.get("person_name")}),
        ))
    m = pd.DataFrame(rows).groupby("year").sum(numeric_only=True)
    out = pd.DataFrame(index=m.index)
    out["rows_per_page"] = m.n_rows / m.pages
    out["sections_per_page"] = m.n_sections / m.pages
    out["subtotal_ratio"] = m.n_total / m.n_entry.clip(lower=1)
    out["sig_per_page"] = m.n_sig / m.pages
    out["person_share"] = m.n_person / m.n_rows
    out["latin_share"] = m.n_latin / m.n_lang.clip(lower=1)
    out["arrears_share"] = m.n_arrears / m.n_entry.clip(lower=1)
    out["distinct_persons_per_page"] = m.persons / m.pages
    # DENSITY-NORMALISED governance marker. distinct_persons_per_page moves with rows_per_page (a page
    # with fewer lines names fewer people), so on its own it cannot separate a governance change from a
    # change in page layout. persons_per_row is the same marker with the layout divided out.
    out["persons_per_row"] = m.persons / m.n_rows
    out["pages_per_year"] = m.pages
    out["rows_per_year"] = m.n_rows
    return out


HORIZON_YEARS = {"half_year": 0.5, "annual": 1.0, "sesquiannual": 1.5, "biennial": 2.0,
                 "triennial": 3.0, "quadrennial": 4.0, "quinquennial": 5.0, "multi_year": 5.0}


def build_proxies(d):
    """One yearly proxy per candidate mechanism -- the thing each story says should be moving."""
    df = bp.load_entries()
    inc = v15.classify_income(df)
    exp = v15.classify_expenditure(df)
    years = list(range(1700, Y1 + 1))
    P = pd.DataFrame(index=pd.Index(years, name="year"))

    # --- M2 DIVERSIFICATION: effective number of income sources (1/HHI over category shares) ------
    sh = (inc.groupby(["year", "category"]).amount_real.sum()
          .groupby(level=0).apply(lambda s: (s / s.sum())))
    hhi = sh.pow(2).groupby(level=0).sum()
    P["eff_income_sources"] = (1.0 / hhi).reindex(years)
    P["new_income_share"] = (d["new_income"] / d["income"]).reindex(years)

    # --- M4 PLANNING HORIZON: value-weighted commitment length of expenditure --------------------
    e = exp.copy()
    e["h"] = e.payment_period.map(HORIZON_YEARS)
    hz = e.dropna(subset=["h"])
    P["mean_horizon_yrs"] = ((hz.h * hz.amount_real).groupby(hz.year).sum()
                             / hz.groupby("year").amount_real.sum()).reindex(years)
    P["multiyear_share"] = ((hz[hz.h > 1].groupby("year").amount_real.sum()
                             / hz.groupby("year").amount_real.sum())
                            .reindex(years).fillna(0.0))

    # --- M5 FINANCIAL MANAGEMENT: securities/investment activity on both sides -------------------
    P["invest_spend_share"] = ((exp[exp.text.str.contains(v15.RX_INVEST)]
                                .groupby("year").amount_real.sum()
                                / exp.groupby("year").amount_real.sum())
                               .reindex(years).fillna(0.0))
    P["sec_income_share"] = ((inc[inc.text.str.contains(bp.RX_SEC)]
                              .groupby("year").amount_real.sum()
                              / inc.groupby("year").amount_real.sum())
                             .reindex(years).fillna(0.0))

    # --- M3 GOVERNANCE / M6 ACCOUNTING: from the page metadata -----------------------------------
    P = P.join(load_page_meta())
    # institutional (non-personal) share of expenditure value -- the governance/agency marker used in v4
    P["institutional_pay_share"] = (1 - (exp[exp.has_person].groupby("year").amount_real.sum()
                                         / exp.groupby("year").amount_real.sum())).reindex(years)
    return P


# ===========================================================================
# helpers
# ===========================================================================

def hac(y, X, lags=1):
    m = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(m) < 8:
        return None, m
    res = sm.OLS(m["y"], sm.add_constant(m[X.columns.tolist()])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return res, m


def sup_wald_break(y, lo=Y0, hi=Y1, trim=0.15):
    """Sup-Wald scan for a single mean shift in a yearly series. Returns (argmax year, max Wald,
    p at argmax by the naive chi2(1) -- reported only as a rough guide because the break date is
    estimated, and full scan-corrected critical values are not worth it for a supporting series)."""
    s = y.reindex(range(lo, hi + 1)).dropna()
    if len(s) < 20:
        return np.nan, np.nan, np.nan
    yrs = s.index.to_numpy()
    n = len(s)
    a, b = int(n * trim), int(n * (1 - trim))
    best = (np.nan, -np.inf, np.nan)
    for i in range(a, b):
        tau = yrs[i]
        X = pd.DataFrame({"post": (yrs >= tau).astype(float)}, index=yrs)
        res, _ = hac(s, X)
        if res is None:
            continue
        w = float((res.params["post"] / res.bse["post"]) ** 2)
        if w > best[1]:
            best = (int(tau), w, float(1 - stats.chi2.cdf(w, 1)))
    return best


def era_slice(s, lo, hi):
    return s.reindex(range(lo, hi + 1))


# ===========================================================================
# T0. HOW MUCH DOES THE INCOME REPAIR MATTER? (run before anything else)
# ===========================================================================

def t0_income_treatment(d):
    """Before testing mechanisms, establish which results are choices and which are facts.

    Two years (1859, 1862) are known-bad income extractions. Three defensible treatments:
        raw          -- keep them (what v16 and v17 did)
        masked       -- drop those years (the default here)
        interpolated -- fill them from neighbours
    For each, report the era-by-era adjustment speed, the 1820 interaction, and where an unrestricted
    scan puts the break. Anything that survives all three is a fact about the archive; anything that
    does not is a choice, and must be labelled as one."""
    L = np.log
    rows, scans = [], []
    for lab, inc in (("raw (v16/v17)", d.income), ("masked (default)", d.income_clean),
                     ("interpolated", d.income_interp)):
        dexp, dinc = L(d.expenditure).diff(), L(inc).diff()
        gap = (L(inc) - L(d.expenditure)).shift(1)
        idx = list(range(Y0, Y1 + 1))
        base = pd.DataFrame({"gap": gap.reindex(idx), "dexp1": dexp.shift(1).reindex(idx),
                             "dinc1": dinc.shift(1).reindex(idx)}, index=idx)
        y = dexp.reindex(idx)
        best = (np.nan, -np.inf)
        row20 = {}
        for tau in range(1812, 1886):
            post = pd.Series([(yy >= tau) * 1.0 for yy in idx], index=idx)
            # FULLY INTERACTED: every coefficient, not just the gap, is allowed to differ across the
            # two regimes. This matters -- a restricted scan that forces common short-run dynamics
            # puts the break in the 1860s instead (see FINDINGS), because the eras differ in their
            # short-run dynamics too and the restricted model absorbs that through the gap term.
            X = base.assign(post=post, gap_x_post=base.gap * post,
                            dexp1_x_post=base.dexp1 * post, dinc1_x_post=base.dinc1 * post)
            res, m = hac(y, X)
            if res is None:
                continue
            w = float((res.params["gap_x_post"] / res.bse["gap_x_post"]) ** 2)
            scans.append(dict(treatment=lab, tau=tau, wald=w))
            if w > best[1]:
                best = (tau, w)
            if tau == BREAK:
                row20 = dict(theta_1820=res.params["gap_x_post"], p_1820=res.pvalues["gap_x_post"],
                             n=len(m))
        lams = {}
        for elab, lo, hi in ERAS3:
            lam, p, n = v17.ecm_lambda(inc, d.expenditure, lo, hi, anchor=inc)
            lams[f"lambda_era{lo}"] = lam       # NB distinct prefix: p_1820 below is the
            lams[f"p_era{lo}"] = p              # INTERACTION p-value, not an era p-value
        ratios = {}
        for elab, lo, hi in ERAS3:
            both = pd.concat([era_slice(dinc, lo, hi).rename("i"),
                              era_slice(dexp, lo, hi).rename("e")], axis=1).dropna()
            ratios[f"smoothing_{lo}"] = both.e.std() / both.i.std()
        rows.append(dict(treatment=lab, scan_argmax=best[0], scan_wald=best[1], **row20,
                         **lams, **ratios))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "t0_income_treatment.csv", index=False)
    pd.DataFrame(scans).to_csv(OUT / "t0_break_scans.csv", index=False)
    return r, pd.DataFrame(scans)


# ===========================================================================
# T1. CURRENT vs PERMANENT INCOME
# ===========================================================================

def t1_current_vs_permanent(d):
    """The professor's test (1). Two specifications, each run era by era.

    (a) GROWTH horse race   Dlog exp_t = a + bc*Dlog income_t + bp*Dlog perm_t + e
        -- which income concept moves spending contemporaneously?
    (b) LEVELS horse race   Dlog exp_t = a + lam_c*(log inc - log exp)_{t-1}
                                           + lam_p*(log perm - log exp)_{t-1} + ...
        -- which income concept does spending error-correct BACK TO?
    M1 predicts bc, lam_c large pre-1820 and collapsing after, while bp, lam_p survive or strengthen."""
    L = np.log
    rows_g, rows_l = [], []
    for k in (3, 5, 7):
        perm = d[f"perm{k}"]
        dinc, dexp, dperm = L(d.income_clean).diff(), L(d.expenditure).diff(), L(perm).diff()
        for lab, lo, hi in ERAS3:
            X = pd.DataFrame({"d_current": era_slice(dinc, lo, hi),
                              "d_permanent": era_slice(dperm, lo, hi)})
            res, m = hac(era_slice(dexp, lo, hi), X)
            if res is None:
                continue
            # separate single-regressor fits, to report how much each concept explains on its own
            r_c, _ = hac(era_slice(dexp, lo, hi), X[["d_current"]])
            r_p, _ = hac(era_slice(dexp, lo, hi), X[["d_permanent"]])
            rows_g.append(dict(k=k, era=lab, n=len(m),
                               b_current=res.params["d_current"], p_current=res.pvalues["d_current"],
                               b_permanent=res.params["d_permanent"], p_permanent=res.pvalues["d_permanent"],
                               r2_current_only=r_c.rsquared, r2_permanent_only=r_p.rsquared,
                               r2_both=res.rsquared))

            gap_c = (L(d.income_clean) - L(d.expenditure)).shift(1)
            gap_p = (L(perm) - L(d.expenditure)).shift(1)
            Xl = pd.DataFrame({"gap_current": era_slice(gap_c, lo, hi),
                               "gap_permanent": era_slice(gap_p, lo, hi),
                               "dexp1": era_slice(dexp.shift(1), lo, hi)})
            rl, ml = hac(era_slice(dexp, lo, hi), Xl)
            if rl is not None:
                rows_l.append(dict(k=k, era=lab, n=len(ml),
                                   lambda_current=rl.params["gap_current"],
                                   p_current=rl.pvalues["gap_current"],
                                   lambda_permanent=rl.params["gap_permanent"],
                                   p_permanent=rl.pvalues["gap_permanent"]))
    g = pd.DataFrame(rows_g)
    l = pd.DataFrame(rows_l)
    g.to_csv(OUT / "t1_growth_horserace.csv", index=False)
    l.to_csv(OUT / "t1_levels_horserace.csv", index=False)

    # pooled formal test of the CHANGE (1800-1900, post-1820 interaction) at the headline k
    perm = d[f"perm{K_PERM}"]
    dinc, dexp, dperm = L(d.income_clean).diff(), L(d.expenditure).diff(), L(perm).diff()
    idx = range(Y0, Y1 + 1)
    post = pd.Series((np.array(idx) >= BREAK).astype(float), index=idx)
    X = pd.DataFrame({"d_current": era_slice(dinc, Y0, Y1),
                      "d_permanent": era_slice(dperm, Y0, Y1),
                      "post": post})
    X["cur_x_post"] = X.d_current * X.post
    X["perm_x_post"] = X.d_permanent * X.post
    res, m = hac(era_slice(dexp, Y0, Y1), X)
    pooled = dict(n=len(m),
                  b_current_pre=res.params["d_current"], p_current_pre=res.pvalues["d_current"],
                  d_current_post=res.params["cur_x_post"], p_change_current=res.pvalues["cur_x_post"],
                  b_current_post=res.params["d_current"] + res.params["cur_x_post"],
                  b_permanent_pre=res.params["d_permanent"], p_permanent_pre=res.pvalues["d_permanent"],
                  d_permanent_post=res.params["perm_x_post"], p_change_permanent=res.pvalues["perm_x_post"],
                  b_permanent_post=res.params["d_permanent"] + res.params["perm_x_post"])
    pd.DataFrame([pooled]).to_csv(OUT / "t1_pooled_interaction.csv", index=False)
    return g, l, pooled


# ===========================================================================
# T2. INCOME-SHOCK RESPONSE
# ===========================================================================

def t2_shocks(d):
    """The professor's test (2), in two parts.

    (a) TRANSITORY vs PERMANENT. This year's income growth splits EXACTLY into two pieces:
            Dlog income_t = Dlog perm_t + D(transitory_t)
        where perm is the 5-year trailing geometric mean and transitory is the deviation from it.
        Regressing spending growth on both pieces therefore decomposes the ordinary income
        elasticity into a permanent and a transitory response that are directly comparable (imposing
        equality returns the plain income-growth regression). M1 predicts the transitory response
        dies after 1820 while the permanent response survives.
        CAVEAT stated up front: the growth of a 5-year mean has roughly one-fifth the standard
        deviation of income growth itself, so the permanent coefficient is estimated far less
        precisely than the transitory one. The transitory coefficient is the informative half.

    (b) SIGNED ASYMMETRY. Shocks are the residuals of an AR(1) on income growth fitted on the FULL
        modern window (one shock definition for both eras, so the eras are comparable). Regress
        spending growth separately on the positive and the negative part.
        - a genuine planned/reserve rule damps BOTH signs after 1820;
        - a merely liquidity-constrained college damps the upside but still passes through the
          downside, because you cannot spend money you did not receive. This is the sharpest single
          discriminator between M1 (decision rule) and M5 (a reserve doing the work mechanically)."""
    L = np.log
    dexp = L(d.expenditure).diff()
    dinc = L(d.income_clean).diff()
    # (a) transitory vs permanent, in TWO parametrisations and three windows k.
    #     "growth"    : Dexp on (Dlog perm, D transitory)  -- an exact decomposition of income growth
    #     "deviation" : Dexp on (Dlog perm, transitory LEVEL) -- does spending react to a year that is
    #                   temporarily above trend?
    #     Both are reported because they disagree in power: with k=5 the 1800-19 era keeps only 15
    #     usable years (a 5-year trailing mean cannot start before 1805 -- 1750-99 is an archive gap),
    #     so the pre-1820 coefficients in the growth form are very imprecisely estimated.
    rows = []
    for k in (3, 5, 7):
        dperm = L(d[f"perm{k}"]).diff()
        tr_lvl = np.log(d.income_clean) - np.log(d[f"perm{k}"])
        for form, tr in (("growth", tr_lvl.diff()), ("deviation", tr_lvl)):
            for lab, lo, hi in ERAS3:
                X = pd.DataFrame({"permanent": era_slice(dperm, lo, hi),
                                  "transitory": era_slice(tr, lo, hi)})
                res, m = hac(era_slice(dexp, lo, hi), X)
                if res is None:
                    continue
                w = res.t_test("permanent - transitory = 0")
                rows.append(dict(k=k, form=form, era=lab, n=len(m),
                                 b_permanent=res.params["permanent"],
                                 p_permanent=res.pvalues["permanent"],
                                 b_transitory=res.params["transitory"],
                                 p_transitory=res.pvalues["transitory"],
                                 p_equal_response=float(np.squeeze(w.pvalue)), r2=res.rsquared))
    tp = pd.DataFrame(rows)
    tp.to_csv(OUT / "t2_transitory_vs_permanent.csv", index=False)
    dperm = L(d.perm).diff()
    trans = d.transitory.diff()

    # pooled interaction: is the fall in the transitory response significant?
    idx = range(Y0, Y1 + 1)
    post = pd.Series((np.array(idx) >= BREAK).astype(float), index=idx)
    X = pd.DataFrame({"permanent": era_slice(dperm, Y0, Y1),
                      "transitory": era_slice(trans, Y0, Y1), "post": post})
    X["perm_x_post"] = X.permanent * X.post
    X["trans_x_post"] = X.transitory * X.post
    res, m = hac(era_slice(dexp, Y0, Y1), X)
    tp_pooled = dict(n=len(m),
                     b_transitory_pre=res.params["transitory"],
                     d_transitory_post=res.params["trans_x_post"],
                     p_change_transitory=res.pvalues["trans_x_post"],
                     b_transitory_post=res.params["transitory"] + res.params["trans_x_post"],
                     b_permanent_pre=res.params["permanent"],
                     d_permanent_post=res.params["perm_x_post"],
                     p_change_permanent=res.pvalues["perm_x_post"],
                     b_permanent_post=res.params["permanent"] + res.params["perm_x_post"])
    pd.DataFrame([tp_pooled]).to_csv(OUT / "t2_transitory_pooled.csv", index=False)

    # (b) signed shocks from one common AR(1)
    ar = pd.DataFrame({"y": era_slice(dinc, Y0, Y1), "l1": era_slice(dinc.shift(1), Y0, Y1)}).dropna()
    arfit = sm.OLS(ar["y"], sm.add_constant(ar[["l1"]])).fit()
    shock = pd.Series(arfit.resid, index=ar.index)
    pos, neg = shock.clip(lower=0), shock.clip(upper=0)

    rows = []
    for lab, lo, hi in ERAS3:
        X = pd.DataFrame({"pos": era_slice(pos, lo, hi), "neg": era_slice(neg, lo, hi)})
        res, m = hac(era_slice(dexp, lo, hi), X)
        if res is None:
            continue
        # Wald test of symmetry, pos == neg
        w = res.t_test("pos - neg = 0")
        rows.append(dict(era=lab, n=len(m),
                         b_pos=res.params["pos"], p_pos=res.pvalues["pos"],
                         b_neg=res.params["neg"], p_neg=res.pvalues["neg"],
                         p_symmetry=float(np.squeeze(w.pvalue)),
                         n_pos=int((era_slice(pos, lo, hi) > 0).sum()),
                         n_neg=int((era_slice(neg, lo, hi) < 0).sum())))
    sg = pd.DataFrame(rows)
    sg.to_csv(OUT / "t2_signed_shocks.csv", index=False)

    # pooled interaction on the signed shocks
    X = pd.DataFrame({"pos": era_slice(pos, Y0, Y1), "neg": era_slice(neg, Y0, Y1), "post": post})
    X["pos_x_post"] = X.pos * X.post
    X["neg_x_post"] = X.neg * X.post
    res, m = hac(era_slice(dexp, Y0, Y1), X)
    sg_pooled = dict(n=len(m),
                     b_pos_pre=res.params["pos"], d_pos_post=res.params["pos_x_post"],
                     p_change_pos=res.pvalues["pos_x_post"],
                     b_pos_post=res.params["pos"] + res.params["pos_x_post"],
                     b_neg_pre=res.params["neg"], d_neg_post=res.params["neg_x_post"],
                     p_change_neg=res.pvalues["neg_x_post"],
                     b_neg_post=res.params["neg"] + res.params["neg_x_post"],
                     ar1_rho=float(arfit.params["l1"]))
    pd.DataFrame([sg_pooled]).to_csv(OUT / "t2_signed_pooled.csv", index=False)
    return tp, tp_pooled, sg, sg_pooled, shock


# ===========================================================================
# T3. EXCESS SMOOTHNESS  +  THE COMPOSITION PLACEBO
# ===========================================================================

def t3_smoothness(d):
    """The professor's test (3) with the discrimination he asked for built in.

    (a) Raw smoothing ratio sd(Dlog exp)/sd(Dlog inc) -- v17's headline, repeated on the repaired
        income series so it is comparable with everything else here.
    (b) EXCESS smoothness, in Deaton's sense. A rule that spends the permanent component of income is
        ALREADY smoother than income; that is not news. The benchmark is what a forward-looking
        spender SHOULD do given how persistent income shocks actually are. Fit an AR(1) to income
        growth within the era; an innovation eps moves the level of income permanently by
        eps/(1-rho), so a permanent-income rule implies sd(Dlog exp) = sd(eps)/(1-rho).
            excess_smoothness = actual sd(Dlog exp) / that benchmark
        Below 1 = spending is smoother than even a permanent-income rule requires (something is
        absorbing the rest); above 1 = spending over-reacts. Reported alongside the cruder ratio of
        spending volatility to the volatility of the 5-year mean.
    (c) The absolute-vs-relative decomposition. If the ratio fell only because income got noisier,
        M2 (diversification) is sufficient and M1 is unnecessary. So we report sd(exp) and sd(inc)
        separately, and the counterfactual ratio that would have obtained with the pre-1820 spending
        volatility and the post-1820 income volatility."""
    L = np.log
    dexp, dinc = L(d.expenditure).diff(), L(d.income_clean).diff()
    dperm = L(d.perm).diff()
    rows = []
    for lab, lo, hi in ERAS3:
        # NB the income/spending ratio is computed on the pairwise-complete income-spending sample,
        # NOT on the sample that also has a 5-year mean available. Restricting to the latter would
        # silently drop the first five years of the 1800-19 era and make this table disagree with T0
        # and with the placebo (1.77 instead of 1.56 for the same era).
        both = pd.concat([era_slice(dinc, lo, hi).rename("i"),
                          era_slice(dexp, lo, hi).rename("e")], axis=1).dropna()
        perm_sd = era_slice(dperm, lo, hi).dropna().std()
        # era-specific AR(1) on income growth -> persistence of an income innovation
        ar = pd.DataFrame({"y": era_slice(dinc, lo, hi),
                           "l1": era_slice(dinc.shift(1), lo, hi)}).dropna()
        f = sm.OLS(ar["y"], sm.add_constant(ar[["l1"]])).fit()
        rho = float(f.params["l1"])
        sd_eps = float(np.std(f.resid, ddof=2))
        bench = sd_eps / (1 - rho)                      # PIH-implied sd of spending growth
        rows.append(dict(era=lab, n=len(both),
                         sd_income=both.i.std(), sd_expenditure=both.e.std(),
                         sd_permanent=perm_sd,
                         smoothing_ratio=both.e.std() / both.i.std(),
                         ar1_rho=rho, sd_innovation=sd_eps, pih_benchmark_sd=bench,
                         excess_smoothness=both.e.std() / bench,
                         sd_ratio_vs_5yr_mean=both.e.std() / perm_sd,
                         persistence_share=perm_sd / both.i.std()))
    sm_ = pd.DataFrame(rows)
    # counterfactual: pre-1820 spending volatility against each era's income volatility
    sd_e_pre = sm_.loc[0, "sd_expenditure"]
    sm_["ratio_if_spending_unchanged"] = sd_e_pre / sm_["sd_income"]
    sm_.to_csv(OUT / "t3_smoothness.csv", index=False)
    return sm_


def t3b_placebo(d, n_boot=2000):
    """THE COMPOSITION PLACEBO -- the single most important test in this file.

    Rival M2 says: the rule never changed; income simply became lumpier, and a mechanical matching
    rule fed lumpier income would look exactly like what we see. Test it by simulation.

    Estimate the pre-1820 rule ONCE (the error-correction model on 1800-1819). Then run that FIXED
    rule forward from 1820, driven by the ACTUAL post-1820 income. If the simulated spending path is
    as smooth as the observed one, M2 is sufficient and M1 is unnecessary. If the simulated path is
    markedly more volatile than what Oxford actually spent, the composition change alone cannot
    account for the smoothing and the rule itself must have changed.

    Uncertainty: the pre-1820 rule is estimated on 19 observations, so the placebo ratio is
    bootstrapped by resampling the pre-1820 regression residuals AND redrawing the coefficients from
    their sampling distribution."""
    L = np.log
    inc, exp = d.income_clean, d.expenditure
    dexp, dinc = L(exp).diff(), L(inc).diff()
    gap = (L(inc) - L(exp)).shift(1)
    pre = pd.DataFrame({"y": dexp, "gap": gap, "dexp1": dexp.shift(1),
                        "dinc1": dinc.shift(1)}).loc[1800:1819].dropna()
    fit = sm.OLS(pre["y"], sm.add_constant(pre[["gap", "dexp1", "dinc1"]])).fit()
    b = fit.params
    resid_sd = float(np.std(fit.resid, ddof=len(b)))

    inc_l = L(inc)
    rng = np.random.default_rng(SEED)

    def simulate(params, shocks, yrs):
        """Run the fixed rule forward from the year before `yrs`, driven by ACTUAL income."""
        y0 = yrs[0]
        le = float(L(exp).loc[y0 - 1])
        le_prev_d = float(dexp.loc[y0 - 1]) if np.isfinite(dexp.loc[y0 - 1]) else 0.0
        path = {}
        for j, y in enumerate(yrs):
            g = float(inc_l.loc[y - 1] - le)
            di_1 = float(inc_l.loc[y - 1] - inc_l.loc[y - 2])
            step = (params["const"] + params["gap"] * g + params["dexp1"] * le_prev_d
                    + params["dinc1"] * di_1 + shocks[j])
            le += step
            le_prev_d = step
            path[y] = step
        return pd.Series(path)

    cov = fit.cov_params()

    def bootstrap(yrs):
        """Distribution of the smoothing ratio the FIXED pre-1820 rule would generate over `yrs`,
        with both parameter uncertainty and the rule's own innovation variance carried over."""
        sd_i = dinc.reindex(yrs).std()
        out = []
        for _ in range(n_boot):
            pb = pd.Series(rng.multivariate_normal(b.values, cov.values), index=b.index)
            s = simulate(pb, rng.normal(0, resid_sd, len(yrs)), yrs)
            if not np.isfinite(s).all() or s.abs().max() > 5:      # discard explosive draws
                continue
            out.append(s.std() / sd_i)
        return np.array(out)

    # calibration window starts at 1802: the rule needs income at t-2, and 1799 is inside the
    # 1750-1799 archive gap.
    yrs_pre = list(range(1802, 1820))
    yrs_post = list(range(BREAK, REFORM))
    obs_pre = float(dexp.reindex(yrs_pre).std() / dinc.reindex(yrs_pre).std())
    obs_post = float(dexp.reindex(yrs_post).std() / dinc.reindex(yrs_post).std())

    # CALIBRATION: fed its OWN era's income, does the simulator reproduce what Oxford actually did?
    # Without this the placebo is unfalsifiable -- a simulator that cannot match the era it was fitted
    # to tells us nothing about the era it was not.
    cal = bootstrap(yrs_pre)
    # THE PLACEBO: the same fixed rule, fed the actual 1820-53 income.
    draws = bootstrap(yrs_post)

    # deterministic-only variant (no innovations): how much of income's variance the rule mechanically
    # transmits. Reported for transparency, but it is NOT comparable to the observed ratio, which
    # contains spending's own innovations -- the stochastic version is the like-for-like comparison.
    det = simulate(b, np.zeros(len(yrs_post)), yrs_post)
    det_ratio = float(det.std() / dinc.reindex(yrs_post).std())

    med = float(np.median(draws))
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    total_fall = obs_pre - obs_post
    comp_fall = obs_pre - med                    # attributable to income composition alone
    out = dict(observed_ratio_1800_19=obs_pre,
               observed_ratio_1820_53=obs_post,
               calibration_median_1800_19=float(np.median(cal)),
               calibration_ci_lo=float(np.percentile(cal, 2.5)),
               calibration_ci_hi=float(np.percentile(cal, 97.5)),
               calibration_ok=bool(np.percentile(cal, 2.5) <= obs_pre <= np.percentile(cal, 97.5)),
               placebo_ratio_median=med, placebo_ci_lo=lo, placebo_ci_hi=hi,
               placebo_ratio_deterministic_only=det_ratio,
               share_of_draws_above_observed=float((draws > obs_post).mean()),
               n_valid_draws=int(len(draws)), pre1820_lambda=float(b["gap"]),
               total_fall_in_ratio=float(total_fall),
               fall_explained_by_income_composition=float(comp_fall),
               pct_explained_by_composition=float(100 * comp_fall / total_fall),
               pct_requiring_a_rule_change=float(100 * (med - obs_post) / total_fall),
               verdict=("composition alone does NOT reproduce the smoothing -- a rule change is "
                        "needed for the remainder" if lo > obs_post else
                        "composition alone CAN reproduce the smoothing -- M1 not needed"))
    pd.DataFrame([out]).to_csv(OUT / "t3b_composition_placebo.csv", index=False)
    return out, cal, draws


# ===========================================================================
# T4. BREAK-DATE HORSE RACE
# ===========================================================================

CANDIDATE_PROXIES = [
    ("M2 diversification", "eff_income_sources", "effective no. of income sources (1/HHI)"),
    ("M2 diversification", "new_income_share", "non-endowment share of income"),
    ("M3 governance", "institutional_pay_share", "share of spending to institutional (non-personal) payees"),
    ("M3 governance", "distinct_persons_per_page", "distinct named people per page"),
    ("M3 governance", "persons_per_row", "named people per ROW (page layout divided out)"),
    ("M3 governance", "sig_per_page", "signature rows per page"),
    ("M4 planning horizon", "mean_horizon_yrs", "value-weighted commitment length"),
    ("M4 planning horizon", "multiyear_share", "share of spend on >1-year commitments"),
    ("M5 financial mgmt", "invest_spend_share", "investment purchases / total spending"),
    ("M5 financial mgmt", "sec_income_share", "securities & annuity income / total income"),
    ("M6 accounting", "rows_per_page", "rows per page (physical layout)"),
    ("M6 accounting", "rows_per_year", "rows per YEAR (information actually recorded)"),
    ("M6 accounting", "pages_per_year", "pages per year"),
    ("M6 accounting", "sections_per_page", "sections per page"),
    ("M6 accounting", "subtotal_ratio", "subtotal rows per entry"),
    ("M6 accounting", "latin_share", "Latin/mixed-language share of rows"),
    ("M6 accounting", "arrears_share", "arrears-flagged share of entries"),
]


ANDREWS_CV5 = 8.85     # Andrews (1993) sup-Wald 5% critical value, 1 parameter, 15% trimming


def t4_break_dates(P):
    """Each candidate mechanism leaves a datable trace. If a candidate is to EXPLAIN the 1820
    coupling break, its own trace has to move at or before 1820 -- a proxy that only breaks in 1854
    or 1870 is a consequence of the reform, not a cause of the 1820 change.

    The coupling break itself is 1819-1822 (v16: sup-Wald argmax 1819, Bai-Perron 1822)."""
    rows = []
    for mech, col, desc in CANDIDATE_PROXIES:
        if col not in P.columns:
            continue
        yr, w, p = sup_wald_break(P[col])
        sig = bool(w == w and w > ANDREWS_CV5)
        rows.append(dict(mechanism=mech, proxy=col, description=desc,
                         break_year=yr, wald=w, significant_break=sig,
                         distance_from_1820=(abs(yr - BREAK) if yr == yr else np.nan),
                         timing=("NO detectable break" if not sig else
                                 "consistent with 1820" if abs(yr - BREAK) <= 5 else
                                 "too late (post-reform)" if yr >= REFORM - 3 else
                                 "between 1826 and 1850")))
    r = pd.DataFrame(rows).sort_values("break_year")
    # reference row: the coupling break itself, from v16's sup-Wald scan of the adjustment speed
    ref = REPORTS / "analysis_v16" / "supwald_scan.csv"
    if ref.exists():
        sc = pd.read_csv(ref)
        top = sc.loc[sc.wald.idxmax()]
        r = pd.concat([pd.DataFrame([dict(
            mechanism="M1 decision rule (REFERENCE)", proxy="income-spending coupling",
            description="v16 sup-Wald scan of the error-correction speed",
            break_year=int(top.break_year), wald=float(top.wald),
            significant_break=bool(top.wald > ANDREWS_CV5), distance_from_1820=0.0,
            timing="THE break being explained")]), r], ignore_index=True)
    r.to_csv(OUT / "t4_break_dates.csv", index=False)
    return r


# ===========================================================================
# T5. ABSORPTION HORSE RACE -- can any rival proxy absorb the lambda collapse?
# ===========================================================================

def t5_absorption(d, P):
    """The direct competition. The baseline is v16's finding written as one interaction:

        Dlog exp_t = a + lam*gap_{t-1} + theta*(gap_{t-1} x post1820) + controls

    theta is the collapse in the adjustment speed. Now add ONE rival proxy Z at a time, both on its
    own and interacted with the gap. If the rival is what really drives the decoupling, theta should
    shrink toward zero and lose significance once Z is in the model. If theta is untouched, the
    rival does not explain the change.

    (With ~100 annual observations we add one Z at a time -- a kitchen-sink model would be
    uninformative here, and saying so is more honest than reporting an over-fitted table.)"""
    L = np.log
    dexp, dinc = L(d.expenditure).diff(), L(d.income_clean).diff()
    gap = (L(d.income_clean) - L(d.expenditure)).shift(1)
    idx = list(range(Y0, Y1 + 1))
    post = pd.Series([(y >= BREAK) * 1.0 for y in idx], index=idx)

    base_X = pd.DataFrame({"gap": era_slice(gap, Y0, Y1), "dexp1": era_slice(dexp.shift(1), Y0, Y1),
                           "dinc1": era_slice(dinc.shift(1), Y0, Y1), "post": post})
    base_X["gap_x_post"] = base_X.gap * base_X.post
    base, mb = hac(era_slice(dexp, Y0, Y1), base_X)
    rows = [dict(model="baseline (no rival)", mechanism="M1 decision rule", proxy="-",
                 lambda_pre=base.params["gap"], theta=base.params["gap_x_post"],
                 p_theta=base.pvalues["gap_x_post"], gamma=np.nan, p_gamma=np.nan, n=len(mb))]

    # A LINEAR TIME TREND is included as a placebo rival. Any smoothly trending series interacted
    # with the gap can soak up part of a step change in a 99-observation sample. If a real rival
    # absorbs no more of theta than a meaningless trend does, its absorption is not evidence.
    P = P.copy()
    P["_trend_placebo"] = pd.Series(P.index.astype(float), index=P.index)
    for mech, col, _ in CANDIDATE_PROXIES + [("PLACEBO", "_trend_placebo", "linear time trend")]:
        if col not in P.columns:
            continue
        z = P[col].reindex(idx).astype(float)
        z = (z - z.mean()) / z.std()
        z = z.interpolate(limit_area="inside")
        X = base_X.copy()
        X["z"] = z.shift(1)
        X["gap_x_z"] = X.gap * X.z
        res, m = hac(era_slice(dexp, Y0, Y1), X)
        if res is None:
            continue
        rows.append(dict(model=f"+ {col}", mechanism=mech, proxy=col,
                         lambda_pre=res.params["gap"], theta=res.params["gap_x_post"],
                         p_theta=res.pvalues["gap_x_post"],
                         gamma=res.params["gap_x_z"], p_gamma=res.pvalues["gap_x_z"], n=len(m)))
    r = pd.DataFrame(rows)
    r["theta_retained_pct"] = 100 * r["theta"] / r.loc[0, "theta"]
    r.to_csv(OUT / "t5_absorption.csv", index=False)
    return r


def t5b_step_vs_trend(d):
    """The absorption race in T5 turns out to have no power (see the trend placebo: a meaningless
    linear trend absorbs the whole effect). That failure is itself informative, but it leaves one
    question that CAN be answered from a single time series:

        did the coupling change ABRUPTLY at a date, or DRIFT smoothly across the century?

    A drift is what most of the rivals imply -- diversification, lengthening horizons and the
    Latin-to-English shift are all gradual. An abrupt change is what a decision rule implies: rules
    are adopted, not drifted into. So we compare, on BIC:

        TREND        every coefficient moves linearly with the year
        STEP@1820    every coefficient shifts once, at 1820
        STEP@best    every coefficient shifts once, at the best-fitting scanned date
        TWO STEPS    a second regime is allowed after the first

    SPECIFICATION WARNING, learned the hard way in this analysis. The comparison must be FULLY
    INTERACTED -- the short-run terms have to be allowed to differ across regimes, not just the
    error-correction term. In the restricted version (only the gap interacted) the best break lands
    in the 1860s and the 1820 step looks no better than a trend; that is the restriction talking, not
    the data. The eras genuinely differ in their short-run dynamics, and when the model is not
    allowed to say so it launders that difference through the gap coefficient. Both versions are
    reported below so the difference is visible rather than asserted."""
    L = np.log
    dexp, dinc = L(d.expenditure).diff(), L(d.income_clean).diff()
    gap = (L(d.income_clean) - L(d.expenditure)).shift(1)
    idx = list(range(Y0, Y1 + 1))
    base = pd.DataFrame({"gap": era_slice(gap, Y0, Y1), "dexp1": era_slice(dexp.shift(1), Y0, Y1),
                         "dinc1": era_slice(dinc.shift(1), Y0, Y1)}, index=idx)
    y = era_slice(dexp, Y0, Y1)
    CORE = ["gap", "dexp1", "dinc1"]

    def fit(X):
        m = pd.concat([y.rename("y"), X], axis=1).dropna()
        r = sm.OLS(m["y"], sm.add_constant(m[X.columns.tolist()])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 1})
        return r, len(m)

    def shifted(sw, suffix, full):
        """Add regime interactions: `full` decides whether all core terms shift or only the gap."""
        X = base.copy()
        X["D" + suffix] = sw
        for c in (CORE if full else ["gap"]):
            X[f"{c}_x{suffix}"] = base[c] * sw
        return X

    rows = []
    for full in (True, False):
        tag = "fully interacted" if full else "restricted (gap only)"
        tr = (pd.Series(idx, index=idx) - Y0) / 100.0
        rt, nt = fit(shifted(tr, "T", full))
        rows.append(dict(spec=tag, model="TREND", tau=np.nan, tau2=np.nan,
                         bic=rt.bic, adj_r2=rt.rsquared_adj, n=nt,
                         theta=rt.params["gap_xT"], p_theta=rt.pvalues["gap_xT"]))

        best = None
        for tau in range(1810, 1886):
            sw = pd.Series([(yy >= tau) * 1.0 for yy in idx], index=idx)
            r, n = fit(shifted(sw, "S", full))
            if best is None or r.bic < best[1].bic:
                best = (tau, r, n)
        tau, r, n = best
        rows.append(dict(spec=tag, model="STEP (best scanned)", tau=tau, tau2=np.nan,
                         bic=r.bic, adj_r2=r.rsquared_adj, n=n,
                         theta=r.params["gap_xS"], p_theta=r.pvalues["gap_xS"]))

        sw = pd.Series([(yy >= BREAK) * 1.0 for yy in idx], index=idx)
        r20, n20 = fit(shifted(sw, "S", full))
        rows.append(dict(spec=tag, model="STEP @1820", tau=BREAK, tau2=np.nan,
                         bic=r20.bic, adj_r2=r20.rsquared_adj, n=n20,
                         theta=r20.params["gap_xS"], p_theta=r20.pvalues["gap_xS"]))

        # two regimes: 1820 plus a scanned second break
        best2 = None
        for tau2 in range(1845, 1886):
            X = shifted(sw, "S", full)
            sw2 = pd.Series([(yy >= tau2) * 1.0 for yy in idx], index=idx)
            X["DS2"] = sw2
            for c in (CORE if full else ["gap"]):
                X[f"{c}_xS2"] = base[c] * sw2
            r2, n2 = fit(X)
            if best2 is None or r2.bic < best2[1].bic:
                best2 = (tau2, r2, n2)
        tau2, r2, n2 = best2
        rows.append(dict(spec=tag, model="TWO STEPS (1820 + scanned)", tau=BREAK, tau2=tau2,
                         bic=r2.bic, adj_r2=r2.rsquared_adj, n=n2,
                         theta=r2.params["gap_xS2"], p_theta=r2.pvalues["gap_xS2"]))

    r = pd.DataFrame(rows)
    r["delta_bic_vs_best_in_spec"] = r.bic - r.groupby("spec").bic.transform("min")
    r.to_csv(OUT / "t5b_step_vs_trend.csv", index=False)
    return r


# ===========================================================================
# T6. VERDICT TABLE
# ===========================================================================

def t6_verdicts(brk, absorb, placebo, t1l, sg, shape):
    """One row per candidate: what it predicts, whether the archive can test it, what we found."""
    b = brk.set_index("proxy")
    fi = shape[shape.spec == "fully interacted"]

    def yr(col):
        if col not in b.index:
            return "n/a"
        row = b.loc[col]
        return f"{int(row.break_year)}" + ("" if row.significant_break else " (n.s.)")

    kl = t1l[t1l.k == K_PERM].reset_index(drop=True)
    rows = [
        dict(mechanism="M1 decision rule (current -> permanent income)",
             observable_implication="spending should error-correct to CURRENT income before the break "
                                    "and to MULTI-YEAR income after it; the change should be abrupt",
             testable="yes",
             evidence=f"anchor switches exactly as predicted: lambda_current {kl.loc[0,'lambda_current']:.2f} "
                      f"(p={kl.loc[0,'p_current']:.3f}) -> {kl.loc[1,'lambda_current']:.2f} (n.s.); "
                      f"lambda_permanent {kl.loc[0,'lambda_permanent']:.2f} (n.s.) -> "
                      f"{kl.loc[1,'lambda_permanent']:.2f} (p={kl.loc[1,'p_permanent']:.3f}). "
                      f"The change is ABRUPT, not a drift: with every coefficient free to differ "
                      f"across regimes, the best-fitting single step is {int(fi[fi.model=='STEP (best scanned)'].tau.iloc[0])} "
                      f"(theta={fi[fi.model=='STEP (best scanned)'].theta.iloc[0]:.2f}, "
                      f"p={fi[fi.model=='STEP (best scanned)'].p_theta.iloc[0]:.3f}); a smooth trend "
                      f"fits worse (dBIC={fi[fi.model=='TREND'].delta_bic_vs_best_in_spec.iloc[0]:.1f}) "
                      f"and a second regime is not supported "
                      f"(dBIC={fi[fi.model=='TWO STEPS (1820 + scanned)'].delta_bic_vs_best_in_spec.iloc[0]:.1f}).",
             verdict="SUPPORTED -- the only candidate whose distinctive prediction is confirmed",
             strength="STRONG"),
        dict(mechanism="M2 revenue diversification (income simply got lumpier)",
             observable_implication="the smoothing ratio should fall on its own once income becomes "
                                    "noisier, with no change in the rule; diversification indices "
                                    "should break at 1820",
             testable="yes -- simulate the unchanged rule on the actual later income",
             evidence=f"the composition placebo reproduces only "
                      f"{placebo['pct_explained_by_composition']:.0f}% of the fall in the smoothing "
                      f"ratio ({placebo['observed_ratio_1800_19']:.2f} -> "
                      f"{placebo['observed_ratio_1820_53']:.2f}); the unchanged rule fed the actual "
                      f"1820-53 income still gives {placebo['placebo_ratio_median']:.2f} "
                      f"[{placebo['placebo_ci_lo']:.2f}, {placebo['placebo_ci_hi']:.2f}]. Effective "
                      f"number of income sources breaks in {yr('eff_income_sources')}, "
                      f"non-endowment share in {yr('new_income_share')} -- both far too late.",
             verdict="PARTIAL -- a real contributor (about three-tenths of the effect) but "
                     "insufficient on its own",
             strength="STRONG"),
        dict(mechanism="M3 governance (who authorises spending changed)",
             observable_implication="markers of who is named and who signs should break at 1820",
             testable="partly -- the ledger names payees and flags signatures, but records no "
                      "committee structure, no statutes and no minutes",
             evidence=f"distinct people per PAGE breaks at {yr('distinct_persons_per_page')}, but that "
                      f"is page layout, not governance: normalised per row the same marker breaks at "
                      f"{yr('persons_per_row')}. Signature rows break at {yr('sig_per_page')}, "
                      f"institutional payee share at {yr('institutional_pay_share')}.",
             verdict="NOT SUPPORTED at 1820 -- the governance markers move with the 1854 reform, "
                     "i.e. they follow the change rather than cause it",
             strength="MEDIUM (proxies are indirect; the decisive records are outside this archive)"),
        dict(mechanism="M4 planning horizon (commitments lengthened)",
             observable_implication="the value-weighted length of commitments, and the share of "
                                    "spending on multi-year obligations, should rise at 1820",
             testable="yes -- every entry carries an enriched payment_period",
             evidence=f"commitment length breaks at {yr('mean_horizon_yrs')}, multi-year share at "
                      f"{yr('multiyear_share')}; multi-year spending is only 2.0% of the total in "
                      f"1800-19 and 3.9% in 1820-53 -- far too small to move the aggregate.",
             verdict="CONTRADICTED -- horizons lengthen half a century too late and at a trivial scale",
             strength="STRONG"),
        dict(mechanism="M5 financial management (a reserve absorbed the swings)",
             observable_implication="securities and investment activity should appear at 1820; and a "
                                    "reserve should protect spending against NEGATIVE income shocks "
                                    "in particular",
             testable="partly -- purchases and securities income are recorded; the reserve STOCK is "
                      "not (v17 T3: balance rows are sparse)",
             evidence=f"investment purchases break at {yr('invest_spend_share')}, securities income at "
                      f"{yr('sec_income_share')}. The signed-shock test points the other way: what "
                      f"changed at 1820 is the response to POSITIVE shocks "
                      f"({sg.loc[0,'b_pos']:.2f} -> {sg.loc[1,'b_pos']:.2f}), while negative shocks "
                      f"were never passed through in either era ({sg.loc[0,'b_neg']:.2f} -> "
                      f"{sg.loc[1,'b_neg']:.2f}, both n.s.).",
             verdict="NOT SUPPORTED as the cause -- the college stopped spending windfalls, which is "
                     "a rule, not a buffer; financial assets arrive with the reform",
             strength="MEDIUM (the stock side is unobserved -- an honest data limit)"),
        dict(mechanism="M6 accounting practice (the ledger changed, not the college)",
             observable_implication="recording density, page structure, subtotalling, language or the "
                                    "treatment of arrears should break at 1820",
             testable="yes -- all are measured directly from the pages",
             evidence=f"rows per year {yr('rows_per_year')}, rows per page {yr('rows_per_page')}, "
                      f"pages per year {yr('pages_per_year')}, sections per page "
                      f"{yr('sections_per_page')}, subtotal ratio {yr('subtotal_ratio')}, Latin share "
                      f"{yr('latin_share')}, arrears share {yr('arrears_share')}. None breaks at 1820. "
                      f"v16 additionally showed the result survives a thinning placebo, nominal "
                      f"(undeflated) money, and dropping interpolated years.",
             verdict="NOT SUPPORTED -- the ledger's form changes at 1854-1882, never at 1820",
             strength="STRONG"),
    ]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "t6_verdicts.csv", index=False)
    return r


# ===========================================================================
# HTML REPORT
# ===========================================================================

CSS = """
* { box-sizing:border-box; }
body { margin:0; background:#ffffff; color:#000000; font-family:Arial, Helvetica, sans-serif; line-height:1.7; }
.wrap { max-width:800px; margin:0 auto; padding:56px 28px 90px; }
h1 { font-size:1.85rem; line-height:1.25; margin:0 0 14px; font-weight:bold; }
.standfirst { font-size:1.08rem; margin:0 0 8px; }
hr.rule { border:0; border-top:1px solid #cccccc; margin:42px 0; }
h2 { font-size:1.2rem; font-weight:bold; margin:42px 0 10px; }
h3 { font-size:1.0rem; font-weight:bold; margin:26px 0 4px; }
h4 { font-size:.94rem; font-weight:bold; margin:20px 0 2px; color:#1f3b5c; }
p { margin:13px 0; }
.box { background:#f6f7f8; border:1px solid #d8dce0; border-left:3px solid #1f3b5c; padding:16px 20px; margin:24px 0; font-size:.9rem; line-height:1.6; }
.box h3 { margin:0 0 8px; font-size:.82rem; letter-spacing:.04em; text-transform:uppercase; color:#1f3b5c; }
.box p { margin:6px 0; }
.box-warning { border-left-color:#b5530f; }
.what { background:#fbfcfd; border:1px dashed #b9c4ce; padding:14px 18px; margin:18px 0; font-size:.9rem; }
.what h3 { margin:0 0 6px; font-size:.8rem; letter-spacing:.04em; text-transform:uppercase; color:#5a6b7b; }
.what p { margin:6px 0; }
.caveat { color:#5a6b7b; font-size:.85rem; }
table { width:100%; border-collapse:collapse; margin:20px 0; font-size:.88rem; }
th { text-align:left; font-weight:bold; padding:0 10px 8px; border-bottom:2px solid #000; vertical-align:bottom; }
td { text-align:left; padding:9px 10px; border-bottom:1px solid #ccc; vertical-align:top; line-height:1.5; }
.num { text-align:right; }
.key { background:#eef2f6; }
figure { margin:28px 0; }
img { max-width:100%; display:block; margin:0 auto; border:1px solid #ccc; }
figcaption { font-size:.85rem; margin-top:10px; text-align:center; font-style:italic; }
ul { margin:13px 0; padding-left:22px; }
li { margin:8px 0; }
strong { font-weight:bold; }
code { font-family:"SFMono-Regular",Consolas,monospace; font-size:.86em; background:#f0f2f4; padding:1px 4px; }
@media print {
  @page { margin: 12mm; }
  .wrap { max-width:100%; padding:0; }
  h2, h3, figure, table { page-break-inside:avoid; }
}
"""


def build_html_report(results):
    """Build the complete HTML report from analysis results."""
    t1l = results["t1l"]
    sg = results["sg"]
    sm_ = results["sm"]
    placebo = results["placebo"]
    shape = results["shape"]

    # Get key values
    kl = t1l[t1l.k == K_PERM].reset_index(drop=True)
    fi = shape[shape.spec == "fully interacted"].reset_index(drop=True)
    best_step = fi[fi.model == "STEP (best scanned)"].iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which Mechanism? Testing the Competing Explanations</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<h1>Which Mechanism? Testing the Competing Explanations</h1>
<p class="standfirst">The question for this week: <strong>last week proposed that Oxford's budgeting
rule changed around 1820. But several other explanations would fit the same evidence. Which of them
does the archive actually support?</strong> This week treats the budgeting rule as one candidate
among six, works out what each one would have left behind in the ledger, and goes looking.</p>

<div class="box">
  <h3>Summary</h3>
  <p><strong>One candidate survives, one is a partial contributor, and four are contradicted.</strong>
  The decision-rule explanation survives because it makes a prediction the others do not: that
  what spending is <em>anchored to</em> should switch from this year's income to a multi-year average.
  That switch is exactly what we find ({kl.iloc[0]['lambda_current']:.2f} to current income before 1820,
  {kl.iloc[1]['lambda_permanent']:.2f} to five-year income after). Revenue diversification turns out to
  be real but partial: simulating the unchanged old rule against the actual later income reproduces
  <strong>only about {placebo['pct_explained_by_composition']:.0f}%</strong> of the smoothing, leaving
  roughly <strong>{placebo['pct_requiring_a_rule_change']:.0f}%</strong> that needs the rule itself
  to have changed. Governance, planning horizon, financial management and accounting practice all
  leave datable traces in the ledger, and <strong>none of them moves at 1820</strong>: they move
  between 1846 and 1883, following the 1854 reform rather than preceding the 1820 change.</p>
  <p>Two things this week did <em>not</em> deliver, stated up front: one of the tests I built
  <strong>failed</strong> (a meaningless time trend absorbs 93% of the effect, so that design cannot
  attribute anything), and the transitory versus permanent income test came back <strong>inconclusive</strong>.
  Both are reported as such below.</p>
</div>

<hr class="rule">

<h2>What changed between last week and this week</h2>

<p>Last week's answer was a mechanism: the rule changed from "match spending to this year's
receipts" to "hold spending on a smoothed path." The instruction this week was to stop building
support for that answer and start trying to break it. So the budgeting rule goes back on the table
as one candidate among several, and each candidate is asked the same three questions:</p>

<div class="box box-warning">
  <p><strong>What would we observe if this mechanism were true? Can we test that in the archive?
  Does the evidence support it, contradict it, or fail to speak?</strong></p>
</div>

<table>
  <tr><th></th><th>Candidate mechanism</th><th>What it would predict</th></tr>
  <tr><td><strong>M1</strong></td><td>Decision rule</td><td>spending stops being anchored to current receipts and becomes anchored to longer-run income; the change is abrupt</td></tr>
  <tr><td><strong>M2</strong></td><td>Revenue diversification</td><td>nothing organisational changed: income simply became lumpier, and any fixed rule fed lumpier income looks like this</td></tr>
  <tr><td><strong>M3</strong></td><td>Governance</td><td>who authorises and receives spending changes at 1820</td></tr>
  <tr><td><strong>M4</strong></td><td>Planning horizon</td><td>commitments lengthen, so spending mechanically stops tracking the single year</td></tr>
  <tr><td><strong>M5</strong></td><td>Financial management</td><td>real reserves and securities appear and absorb the swings</td></tr>
  <tr><td><strong>M6</strong></td><td>Accounting practice</td><td>the college did not change; the ledger changed how it records</td></tr>
</table>

<hr class="rule">

<h2>Test 1: Is spending driven by current income, or by permanent income?</h2>

<div class="what">
  <h3>What we did</h3>
  <p>If the decision rule changed, then what spending corrects <em>back to</em> should change. Before
  1820 spending should be pulled back toward this year's receipts; afterwards toward a multi-year
  average. We estimate both anchors in the same model, era by era, and let them compete.</p>
</div>

<h3>Result</h3>

<table>
  <tr><th>Era</th><th class="num">Pull toward <strong>current</strong> income</th><th class="num">Pull toward <strong>5-year</strong> income</th></tr>
  <tr class="key"><td>1800-1819</td><td class="num"><strong>{kl.iloc[0]['lambda_current']:.2f}</strong> (p = {kl.iloc[0]['p_current']:.3f})</td><td class="num">{kl.iloc[0]['lambda_permanent']:.2f} (not significant)</td></tr>
  <tr class="key"><td>1820-1853</td><td class="num">{kl.iloc[1]['lambda_current']:.2f} (not significant)</td><td class="num"><strong>{kl.iloc[1]['lambda_permanent']:.2f}</strong> (p = {kl.iloc[1]['p_permanent']:.3f})</td></tr>
  <tr><td>1854-1900</td><td class="num">{kl.iloc[2]['lambda_current']:.2f} (not significant)</td><td class="num">{kl.iloc[2]['lambda_permanent']:.2f} (not significant)</td></tr>
</table>

<p><strong>The two anchors swap places at 1820, exactly as the decision-rule explanation predicts.</strong>
This is the clearest single result of the week, and it holds whether permanent income is measured over
three, five or seven years. None of the rival mechanisms predicts this particular pattern.</p>

<p>The third row was not predicted by anyone. After 1854 spending is anchored to <strong>neither</strong>
income concept. That is either a third regime beginning at the reform, or a sign that the post-1854 income
data are too compromised to detect an anchor. The two are currently indistinguishable.</p>

<p class="caveat">Honest limit: the pre-1820 era supplies only 15 usable observations, because a five-year
trailing average cannot begin before 1805 and 1750-99 is an archive gap.</p>

<hr class="rule">

<h2>Test 2: How does spending respond to income shocks?</h2>

<div class="what">
  <h3>What we did</h3>
  <p>Two versions. First, split income into its permanent and temporary parts and ask which one moves
  spending. Second, split shocks by <em>sign</em>: do good years and bad years get treated
  differently, and does that treatment change at 1820?</p>
</div>

<h3>Result 1: the transitory/permanent split is inconclusive</h3>

<p>Reported plainly because it is the honest answer. Two reasonable ways of writing the same test
disagree with each other, and the pre-1820 era has too few observations to separate them (p = 0.17 on
the change). This is the professor's test (2) in its first form, and this archive does not settle it.</p>

<h3>Result 2: what changed was the treatment of windfalls</h3>

<table>
  <tr><th>Era</th><th class="num">Response to a <strong>positive</strong> shock</th><th class="num">Response to a <strong>negative</strong> shock</th></tr>
  <tr class="key"><td>1800-1819</td><td class="num"><strong>{sg.iloc[0]['b_pos']:.2f}</strong> (p = {sg.iloc[0]['p_pos']:.3f})</td><td class="num">{sg.iloc[0]['b_neg']:.2f} (not significant)</td></tr>
  <tr><td>1820-1853</td><td class="num">{sg.iloc[1]['b_pos']:.2f} (p = {sg.iloc[1]['p_pos']:.3f})</td><td class="num">{sg.iloc[1]['b_neg']:.2f} (not significant)</td></tr>
  <tr><td>1854-1900</td><td class="num">{sg.iloc[2]['b_pos']:.2f} (p = {sg.iloc[2]['p_pos']:.3f})</td><td class="num">{sg.iloc[2]['b_neg']:.2f} (not significant)</td></tr>
</table>

<p>Before 1820 an unexpectedly good year was spent, and then some. After 1820, less than half as much of
it was. Meanwhile <strong>bad years were never passed through to spending, in any era</strong>.</p>

<p>This matters for telling two stories apart. If the mechanism were "a reserve appeared"
(M5), the first thing we should see is protection on the <em>downside</em>: that is what reserves
are for. The downside is precisely where nothing changes. What changed is that the college stopped
automatically spending money it happened to receive. That is a rule, not a buffer.</p>

<p class="caveat">Strength: suggestive. The direction is consistent across all three eras, but the formal
test of the change in the windfall response is p = 0.12.</p>

<hr class="rule">

<h2>Test 3: Is spending smoother than income, and is that the rule or just lumpier income?</h2>

<div class="what">
  <h3>What we did</h3>
  <p>The smoothing ratio itself was established last week ({sm_.iloc[0]['smoothing_ratio']:.2f} before
  1820, {sm_.iloc[1]['smoothing_ratio']:.2f} after). The real question this week is <em>why</em> it moves.
  Look at the two halves separately: spending volatility falls modestly ({sm_.iloc[0]['sd_expenditure']:.2f}
  to {sm_.iloc[1]['sd_expenditure']:.2f}), but income volatility rises sharply ({sm_.iloc[0]['sd_income']:.2f}
  to {sm_.iloc[1]['sd_income']:.2f}). Most of the ratio's movement comes from the income side, which is
  exactly what the diversification rival predicts, and the reason it has to be taken seriously.</p>
  <p>So we test it directly. Estimate the pre-1820 rule once. Then run that <strong>fixed, unchanged</strong>
  rule forward from 1820, driven by the <strong>actual</strong> 1820-53 income. If lumpier income
  alone explains the smoothing, the simulation should reproduce it.</p>
</div>

<h3>Result</h3>

<table>
  <tr><th></th><th class="num">Smoothing ratio</th><th>Reading</th></tr>
  <tr><td>Calibration check: the rule fed its <em>own</em> 1800-19 income</td><td class="num">{placebo['calibration_median_1800_19']:.2f} [{placebo['calibration_ci_lo']:.2f}, {placebo['calibration_ci_hi']:.2f}]</td><td>against an observed {placebo['observed_ratio_1800_19']:.2f}: the simulator reproduces the era it was fitted to</td></tr>
  <tr class="key"><td><strong>Placebo: the unchanged rule fed the actual 1820-53 income</strong></td><td class="num"><strong>{placebo['placebo_ratio_median']:.2f} [{placebo['placebo_ci_lo']:.2f}, {placebo['placebo_ci_hi']:.2f}]</strong></td><td>against an observed <strong>{placebo['observed_ratio_1820_53']:.2f}</strong>; {placebo['share_of_draws_above_observed']*100:.0f}% of simulations are less smooth than what Oxford actually did</td></tr>
</table>

<p>Breaking the fall from {placebo['observed_ratio_1800_19']:.2f} to {placebo['observed_ratio_1820_53']:.2f}
into its parts: <strong>income composition accounts for about {placebo['pct_explained_by_composition']:.0f}%
of it; the remaining {placebo['pct_requiring_a_rule_change']:.0f}% requires the rule itself to have changed.</strong></p>

<p>This is a more useful answer than either "diversification explains it" or "diversification is irrelevant."
Diversification is a real contributor of roughly three-tenths. It is not sufficient.</p>

<figure>
  <img src="mechanism_discrimination.png" alt="Four-panel figure: growth horse race, anchor switch, composition placebo, break-date horse race">
  <figcaption>The four discriminating tests. Panel B is the anchor switch (Test 1); panel C is the
  composition placebo (Test 3); panel D is the break-date race below.</figcaption>
</figure>

<hr class="rule">

<h2>Test 4: Whose trace actually moves at 1820?</h2>

<div class="what">
  <h3>What we did</h3>
  <p>Every rival mechanism, if true, leaves something datable in the ledger. A mechanism whose own
  trace only moves in 1868 cannot explain a change in 1820. So we run the same break-detection scan on
  each proxy and compare the dates.</p>
</div>

<h3>Result</h3>

<table>
  <tr><th>Mechanism</th><th>Trace in the ledger</th><th class="num">Break</th></tr>
  <tr class="key"><td><em>the thing being explained</em></td><td>income-spending coupling</td><td class="num"><strong>1819</strong></td></tr>
  <tr><td>M2 diversification</td><td>non-endowment share of income</td><td class="num">1846</td></tr>
  <tr><td>M3 governance</td><td>institutional payee share</td><td class="num">1849</td></tr>
  <tr><td>M5 financial management</td><td>securities income</td><td class="num">1854</td></tr>
  <tr><td>M6 accounting</td><td>arrears share</td><td class="num">1857</td></tr>
  <tr><td>M3 governance</td><td>signature rows per page</td><td class="num">1868</td></tr>
  <tr><td>M2 diversification</td><td>effective number of income sources</td><td class="num">1881</td></tr>
  <tr><td>M3 governance</td><td>named people per row</td><td class="num">1882</td></tr>
  <tr><td>M6 accounting</td><td>Latin share, sections, subtotals, pages per year</td><td class="num">1882</td></tr>
  <tr><td>M4 planning horizon</td><td>multi-year share of spending</td><td class="num">1883</td></tr>
  <tr><td colspan="3" class="caveat">Rows per year, rows per page, investment purchases and commitment length show no statistically detectable break anywhere.</td></tr>
</table>

<p><strong>Nothing comes near 1820.</strong> Every rival's trace moves between 1846 and 1883, clustered
on the 1854 reform and the 1870s-80s. The change in how Oxford <em>spent</em> has no observable
companion in how it was governed, how long it committed, what it invested in, or how it kept its books.</p>

<hr class="rule">

<h2>A test that failed, reported as failed</h2>

<p>The natural next move is to put each rival proxy into the model as a control and see whether the 1820
effect survives. I built that test, and then added a <strong>linear time trend</strong> as a meaningless
placebo rival, to check the test's power.</p>

<div class="box box-warning">
  <p>The meaningless trend absorbs <strong>93% of the effect</strong>: the 1820 coefficient goes from
  -0.42 (p = 0.021) to -0.03 (p = 0.95). Not one real rival comes close.</p>
</div>

<p>With a single 99-year series, any smoothly-moving variable can soak up a step change. So the test has
essentially no power, and the fact that the Latin-share proxy also reduces the effect by 68% is
<strong>not</strong> evidence for an accounting explanation: the meaningless trend does better.</p>

<p>I am reporting this rather than deleting it because the failure is itself informative. <strong>This
archive cannot discriminate mechanisms by controlling for them.</strong> It can discriminate only by
distinctive predictions (Test 1), by break-date coincidence (Test 4), and by simulation (Test 3).</p>

<h3>What the shape of the change can still tell us</h3>

<p>A drift is what most rivals imply: diversification, lengthening horizons and the shift from Latin to
English are all gradual. An abrupt change is what a <em>rule</em> implies: rules are adopted, not
drifted into. Comparing model fit: a single abrupt change at <strong>{int(best_step['tau'])}</strong>
is preferred over a smooth trend, and a second change is decisively rejected.</p>

<figure>
  <img src="absorption_horserace.png" alt="Bar chart of how much of the 1820 effect survives each rival control">
  <figcaption>The failed test. Read the placebo bar alongside every other bar: a meaningless trend
  absorbs more than any real rival does.</figcaption>
</figure>

<hr class="rule">

<h2>Where this leaves the six candidates</h2>

<table>
  <tr><th>Mechanism</th><th>Verdict</th><th>Confidence</th></tr>
  <tr class="key"><td><strong>M1 decision rule</strong></td><td><strong>Supported</strong>: the only candidate whose distinctive prediction (the anchor switch) is confirmed, and the change is abrupt as a rule change implies</td><td>strong</td></tr>
  <tr class="key"><td><strong>M2 diversification</strong></td><td><strong>Partial</strong>: a real contributor of about {placebo['pct_explained_by_composition']:.0f}% of the smoothing, but insufficient on its own</td><td>strong</td></tr>
  <tr><td>M3 governance</td><td>Not supported at 1820: the markers move with the 1854 reform; the single 1818 hit is a page-layout artefact</td><td>medium (proxies are indirect)</td></tr>
  <tr><td>M4 planning horizon</td><td>Contradicted: horizons lengthen half a century too late, and multi-year commitments are only 2-4% of spending</td><td>strong</td></tr>
  <tr><td>M5 financial management</td><td>Not supported as the cause: what changed is the response to windfalls, not to shortfalls; financial assets arrive with the reform</td><td>medium (the reserve stock is unobserved)</td></tr>
  <tr><td>M6 accounting practice</td><td>Not supported: the ledger's form changes between 1854 and 1882, never at 1820</td><td>strong</td></tr>
</table>

<hr class="rule">

<h2>Summary</h2>

<div class="box">
  <h3>The question</h3>
  <p>Last week we proposed that Oxford's budgeting rule changed around 1820. This week the task was to
  stop confirming that story and try to break it, by testing six competing explanations.</p>
</div>

<div class="box box-warning">
  <h3>The answer</h3>
  <p><strong>One candidate survives, one is a partial contributor, and four are ruled out.</strong></p>
  <ul>
    <li><strong>M1 (decision rule):</strong> Supported. The anchor switched from current-year income
    ({kl.iloc[0]['lambda_current']:.2f}) to 5-year permanent income ({kl.iloc[1]['lambda_permanent']:.2f})
    exactly at 1820. No other mechanism predicts this.</li>
    <li><strong>M2 (diversification):</strong> Partial. Explains about {placebo['pct_explained_by_composition']:.0f}%
    of the smoothing, but the remaining {placebo['pct_requiring_a_rule_change']:.0f}% requires a rule change.</li>
    <li><strong>M3, M4, M5, M6:</strong> Not supported. Their traces all break between 1846 and 1883,
    not at 1820. They follow the change, they do not cause it.</li>
  </ul>
</div>

<p>In one sentence: <strong>around 1820, Oxford changed from "spend what we earn this year" to
"spend according to our long-run average income." The other explanations either come too late
or explain too little.</strong></p>

</div>
</body>
</html>
"""

    return html


# ===========================================================================
# FIGURES
# ===========================================================================

def figures(d, t1g, t1l, tp, sg, sm_, placebo, brk, absorb):
    k = t1g[t1g.k == K_PERM]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # A. current vs permanent income coefficients
    a = ax[0, 0]
    x = np.arange(len(k))
    a.bar(x - 0.19, k.b_current, 0.36, color=ORANGE, label="current-year income")
    a.bar(x + 0.19, k.b_permanent, 0.36, color=NAVY, label=f"{K_PERM}-yr permanent income")
    a.axhline(0, color="k", lw=.8)
    a.set_xticks(x); a.set_xticklabels(["1800-19", "1820-53", "1854-1900"], fontsize=9)
    a.set_title("A. Growth horse race (the weaker of the two specifications)")
    a.legend(fontsize=8)

    # B. what does spending error-correct BACK to? (the levels horse race -- the cleanest result)
    a = ax[0, 1]
    kl = t1l[t1l.k == K_PERM]
    x = np.arange(len(kl))
    a.bar(x - 0.19, kl.lambda_current, 0.36, color=ORANGE, label="anchor = current income")
    a.bar(x + 0.19, kl.lambda_permanent, 0.36, color=NAVY,
          label=f"anchor = {K_PERM}-yr permanent income")
    for xi, (_, r) in zip(x, kl.iterrows()):
        for off, p, v in ((-0.19, r.p_current, r.lambda_current),
                          (0.19, r.p_permanent, r.lambda_permanent)):
            a.text(xi + off, v + .03, "*" if p < .05 else "", ha="center", fontsize=14)
    a.axhline(0, color="k", lw=.8)
    a.set_ylim(top=max(kl.lambda_current.max(), kl.lambda_permanent.max()) * 1.22)
    a.set_xticks(x); a.set_xticklabels(["1800-19", "1820-53", "1854-1900"], fontsize=9)
    a.set_title("B. What does spending correct back to? (* = p<0.05)")
    a.legend(fontsize=8)

    # C. composition placebo
    a = ax[1, 0]
    a.hist(placebo["_cal"], bins=40, color=TEAL, alpha=.45,
           label="calibration: rule fed its OWN 1800-19 income")
    a.hist(placebo["_draws"], bins=40, color=GREY, alpha=.75,
           label="placebo: unchanged pre-1820 rule\nfed the actual 1820-53 income")
    a.axvline(placebo["observed_ratio_1800_19"], color=TEAL, lw=2, ls=":",
              label=f"observed 1800-19 {placebo['observed_ratio_1800_19']:.2f}")
    a.axvline(placebo["observed_ratio_1820_53"], color=ORANGE, lw=2.4,
              label=f"observed 1820-53 {placebo['observed_ratio_1820_53']:.2f}")
    a.axvline(placebo["placebo_ratio_median"], color=NAVY, lw=2, ls="--",
              label=f"placebo median {placebo['placebo_ratio_median']:.2f}")
    a.set_xlim(0, 3.2)
    a.set_title("C. Could lumpier income alone explain the smoothing?")
    a.set_xlabel("sd(spending growth) / sd(income growth), 1820-53")
    a.legend(fontsize=7.5)

    # D. break-date horse race
    a = ax[1, 1]
    b = brk.dropna(subset=["break_year"]).sort_values("break_year")
    y = np.arange(len(b))
    cols = {"M2 diversification": NAVY, "M3 governance": TEAL, "M4 planning horizon": ORANGE,
            "M5 financial mgmt": RED, "M6 accounting": GREY, "PLACEBO": "#000000"}
    sig = b.significant_break.fillna(False).to_numpy()
    cc = [cols.get(m, GREY) for m in b.mechanism]
    a.scatter(b.break_year[sig], y[sig], c=[c for c, k in zip(cc, sig) if k], s=60, zorder=3)
    a.scatter(b.break_year[~sig], y[~sig], facecolors="none", s=60, zorder=3,
              edgecolors=[c for c, k in zip(cc, sig) if not k],
              label="hollow = no significant break")
    a.axvline(BREAK, color=ORANGE, lw=2, ls="--", label="coupling break (1820)")
    a.axvline(REFORM, color=GREY, lw=1.2, ls=":", label="reform (1854)")
    a.set_yticks(y); a.set_yticklabels(b.proxy, fontsize=7)
    a.set_xlim(1805, 1900)
    a.set_title("D. When does each rival's own trace break?")
    a.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("v18: discriminating among competing mechanisms for the ~1820 decoupling",
                 fontsize=13, y=.995)
    fig.tight_layout()
    fig.savefig(OUT / "mechanism_discrimination.png", dpi=130)
    plt.close(fig)

    # second figure: the absorption horse race
    fig, a = plt.subplots(figsize=(9, 6))
    r = absorb.iloc[1:].sort_values("theta_retained_pct")
    y = np.arange(len(r))
    a.barh(y, r.theta_retained_pct, color=[cols.get(m, GREY) for m in r.mechanism])
    a.axvline(100, color="k", lw=1.2, ls="--")
    a.set_yticks(y); a.set_yticklabels(r.proxy, fontsize=8)
    a.set_xlabel("% of the post-1820 collapse in adjustment speed that SURVIVES the rival control\n"
                 "(100% = rival explains none of it; 0% = rival fully absorbs it)")
    a.set_title("Does any rival mechanism absorb the 1820 collapse in the coupling?", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "absorption_horserace.png", dpi=130)
    plt.close(fig)


# ===========================================================================
def main():
    np.random.seed(SEED)
    d = build()
    P = build_proxies(d)
    d.join(P, rsuffix="_p").to_csv(OUT / "series_v18.csv")
    P.to_csv(OUT / "mechanism_proxies.csv")

    fmt = lambda x: f"{x:.3f}"

    print("=" * 90)
    print("T0  INCOME-REPAIR ROBUSTNESS -- which results are facts and which are choices?")
    t0, t0scan = t0_income_treatment(d)
    print(t0[["treatment", "lambda_era1800", "p_era1800", "lambda_era1820", "p_era1820",
              "lambda_era1854", "p_era1854"]].to_string(index=False, float_format=fmt))
    print(t0[["treatment", "theta_1820", "p_1820", "scan_argmax", "scan_wald",
              "smoothing_1800", "smoothing_1820", "smoothing_1854"]]
          .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 90)
    print("T1  CURRENT vs PERMANENT INCOME  (professor's test 1)")
    t1g, t1l, pooled = t1_current_vs_permanent(d)
    print("\n(a) growth horse race, k=5:")
    print(t1g[t1g.k == K_PERM].drop(columns="k").to_string(index=False, float_format=fmt))
    print("\n(b) levels (error-correction) horse race, k=5:")
    print(t1l[t1l.k == K_PERM].drop(columns="k").to_string(index=False, float_format=fmt))
    print("\n(c) pooled post-1820 interaction test:")
    for kk, vv in pooled.items():
        print(f"      {kk:24s} {vv:.4f}")

    print("\n" + "=" * 90)
    print("T2  INCOME-SHOCK RESPONSE  (professor's test 2)")
    tp, tp_pooled, sg, sg_pooled, shock = t2_shocks(d)
    print("\n(a) transitory vs permanent (headline k=5, both parametrisations):")
    print(tp[tp.k == K_PERM].drop(columns="k").to_string(index=False, float_format=fmt))
    print("\n    robustness over the permanent-income window k:")
    print(tp.pivot_table(index=["form", "era"], columns="k",
                         values="b_transitory").to_string(float_format=fmt))
    print("\n    pooled change test:")
    for kk, vv in tp_pooled.items():
        print(f"      {kk:24s} {vv:.4f}")
    print("\n(b) signed shocks (positive vs negative):")
    print(sg.to_string(index=False, float_format=fmt))
    print("\n    pooled change test:")
    for kk, vv in sg_pooled.items():
        print(f"      {kk:24s} {vv:.4f}")

    print("\n" + "=" * 90)
    print("T3  EXCESS SMOOTHNESS  (professor's test 3)")
    sm_ = t3_smoothness(d)
    print(sm_.to_string(index=False, float_format=fmt))

    print("\nT3b COMPOSITION PLACEBO -- can lumpier income alone reproduce the smoothing?")
    placebo, cal, draws = t3b_placebo(d)
    for kk, vv in placebo.items():
        print(f"      {kk:38s} {vv if isinstance(vv, str) else round(float(vv), 4)}")
    placebo["_draws"], placebo["_cal"] = draws, cal

    print("\n" + "=" * 90)
    print("T4  BREAK-DATE HORSE RACE -- when does each rival's own trace move?")
    brk = t4_break_dates(P)
    print(brk[["mechanism", "proxy", "break_year", "wald", "timing"]]
          .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 90)
    print("T5  ABSORPTION HORSE RACE -- can any rival absorb the lambda collapse?")
    absorb = t5_absorption(d, P)
    print(absorb[["model", "mechanism", "lambda_pre", "theta", "p_theta", "gamma", "p_gamma",
                  "theta_retained_pct"]].to_string(index=False, float_format=fmt))
    trend_ret = absorb.loc[absorb.mechanism == "PLACEBO", "theta_retained_pct"]
    if len(trend_ret):
        print(f"\n    *** READ THE PLACEBO ROW FIRST: a MEANINGLESS linear time trend absorbs the "
              f"effect down to {trend_ret.iloc[0]:.0f}% of baseline. ***")
        print("    With one 99-year series, any smoothly-moving covariate can soak up a step change,")
        print("    so absorption here is NOT evidence for or against any rival. The test fails; the")
        print("    honest response is T5b (what SHAPE do the data prefer?) plus T4 (break dates).")

    print("\n" + "=" * 90)
    print("T5b SHAPE TEST -- abrupt step or smooth drift?")
    shape = t5b_step_vs_trend(d)
    print(shape.to_string(index=False, float_format=fmt))

    print("\n" + "=" * 90)
    print("T6  VERDICTS")
    verdicts = t6_verdicts(brk, absorb, placebo, t1l, sg, shape)
    for _, r in verdicts.iterrows():
        print(f"\n  {r.mechanism}\n     testable : {r.testable}\n     verdict  : {r.verdict} "
              f"[{r.strength}]")

    figures(d, t1g, t1l, tp, sg, sm_, placebo, brk, absorb)

    # Build and save HTML report
    results = dict(d=d, P=P, t1g=t1g, t1l=t1l, pooled=pooled, tp=tp, tp_pooled=tp_pooled,
                   sg=sg, sg_pooled=sg_pooled, sm=sm_, placebo=placebo, brk=brk, absorb=absorb,
                   shape=shape, verdicts=verdicts)

    print("\n" + "=" * 90)
    print("GENERATING HTML REPORT...")
    html = build_html_report(results)
    report_path = OUT / "analysis_v18_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"HTML report written to: {report_path}")

    print("\nWrote CSVs + figures + HTML report to", OUT)
    return results


if __name__ == "__main__":
    main()
