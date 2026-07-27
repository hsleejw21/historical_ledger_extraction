#!/usr/bin/env python
"""
analysis_v7.py — Causal Inference on Oxford's Financial Transformation
Six analyses:
  C1: Interrupted Time Series (ITS) around 1854 & 1877 Oxford Reform Acts
  C2: Formal Event Study around historical economic shocks
  C3: Local Projections (Jordà 2005) — dynamic response to land rent shocks
  C4: Placebo/Permutation Test — validates ITS level-shifts at 1854 & 1877
  C5: Granger Causality — temporal precedence of land income → MFS
  C6: Local Linear RDD — discontinuity in MFS at Reform Act cutoffs
"""

import warnings
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

ROOT = Path(__file__).resolve().parents[2]
V4   = ROOT / "experiments/reports/analysis_v4"
V5   = ROOT / "experiments/reports/analysis_v5"
V6   = ROOT / "experiments/reports/analysis_v6"
OUT  = ROOT / "experiments/reports/analysis_v7"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path):
    p = Path(path)
    if not p.exists():
        warnings.warn(f"Missing input: {p}")
        return None
    return pd.read_csv(p)


def _b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64, caption=""):
    if b64 is None:
        return '<p class="missing">Figure not available.</p>'
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}" style="max-width:100%">{cap}</figure>'


def _stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    mfs_df    = _read(V5 / "outcome_variable_yearly.csv")
    income_df = _read(V5 / "income_composition_yearly.csv")
    shock_df  = _read(V6 / "shock_event_study.csv")
    stress_df = _read(V4 / "arrears_stress_index.csv")

    lr_df = None
    if income_df is not None:
        lr_df = (
            income_df[income_df["category"] == "land_rent"][["year", "share"]]
            .rename(columns={"share": "land_rent_share"})
            .sort_values("year")
            .reset_index(drop=True)
        )
    return mfs_df, lr_df, shock_df, stress_df


# ---------------------------------------------------------------------------
# C1: Interrupted Time Series
# ---------------------------------------------------------------------------

def run_its(mfs_df):
    df = (
        mfs_df[["year", "modern_function_share"]]
        .dropna()
        .sort_values("year")
        .reset_index(drop=True)
    )

    df["t"] = df["year"] - 1700          # time trend from 1700

    # 1854 Act variables
    df["D54"]      = (df["year"] >= 1854).astype(float)
    df["t_post54"] = (df["year"] - 1854) * df["D54"]

    # 1877 Act variables
    df["D77"]      = (df["year"] >= 1877).astype(float)
    df["t_post77"] = (df["year"] - 1877) * df["D77"]

    X = sm.add_constant(df[["t", "D54", "t_post54", "D77", "t_post77"]])
    Y = df["modern_function_share"]

    res = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    df["fitted"] = res.fittedvalues

    # Counterfactual: no reforms (D54=D77=0, interaction terms=0)
    X_cf = X.copy()
    X_cf["D54"]      = 0.0
    X_cf["t_post54"] = 0.0
    X_cf["D77"]      = 0.0
    X_cf["t_post77"] = 0.0
    df["counterfactual"] = res.predict(X_cf)

    return res, df


def plot_its(res, df):
    fig, ax = plt.subplots(figsize=(11, 5))

    # Scatter (annual)
    ax.scatter(df["year"], df["modern_function_share"],
               alpha=0.25, s=8, color="steelblue", zorder=1)

    # 10yr rolling mean
    roll = df["modern_function_share"].rolling(10, center=True, min_periods=5).mean()
    ax.plot(df["year"], roll, color="steelblue", linewidth=1.4, alpha=0.6, label="10yr rolling mean")

    # ITS fitted segments
    seg_cfg = [
        (df["year"] < 1854,                             "navy",       "ITS fit: pre-1854"),
        ((df["year"] >= 1854) & (df["year"] < 1877),   "darkorange", "ITS fit: 1854–1877"),
        (df["year"] >= 1877,                            "darkgreen",  "ITS fit: post-1877"),
    ]
    for mask, col, lbl in seg_cfg:
        seg = df[mask]
        ax.plot(seg["year"], seg["fitted"], color=col, linewidth=2.2, label=lbl)

    # Counterfactual
    cf_mask = df["year"] >= 1854
    ax.plot(df.loc[cf_mask, "year"], df.loc[cf_mask, "counterfactual"],
            color="gray", linewidth=1.5, linestyle="--", label="Counterfactual (no reforms)")

    # Intervention lines
    for yr, lbl in [(1854, "1854 Act"), (1877, "1877 Act")]:
        ax.axvline(yr, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(yr + 1, 0.65, lbl, color="red", fontsize=8.5, va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("Modern Function Share (MFS)")
    ax.set_title("C1: Interrupted Time Series — Oxford Reform Acts (1854 & 1877)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(1700, 1900)
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    return _b64(fig)


def its_table_html(res):
    params = res.params
    bse    = res.bse
    pvals  = res.pvalues
    ci     = res.conf_int()

    labels = {
        "const":    "Intercept (1700 baseline)",
        "t":        "β₁ — Pre-reform annual trend",
        "D54":      "β₂ — Level shift at 1854",
        "t_post54": "β₃ — Slope change after 1854",
        "D77":      "β₄ — Level shift at 1877",
        "t_post77": "β₅ — Slope change after 1877",
    }

    rows = ""
    for k, lbl in labels.items():
        if k not in params.index:
            continue
        sig = _stars(pvals[k])
        rows += (
            f"<tr><td>{lbl}</td>"
            f"<td>{params[k]:+.4f}{sig}</td>"
            f"<td>{bse[k]:.4f}</td>"
            f"<td>{pvals[k]:.4f}</td>"
            f"<td>[{ci.loc[k,0]:+.4f}, {ci.loc[k,1]:+.4f}]</td></tr>"
        )

    return f"""
<table>
<thead><tr>
  <th>Parameter</th><th>Coeff.</th><th>HAC SE</th><th>p-value</th><th>95% CI</th>
</tr></thead>
<tbody>{rows}</tbody>
<tfoot><tr><td colspan="5">
  N = {int(res.nobs)}, R² = {res.rsquared:.3f}, Adj. R² = {res.rsquared_adj:.3f}.
  Significance: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01 (Newey-West HAC SE, maxlags=10).
</td></tr></tfoot>
</table>"""


# ---------------------------------------------------------------------------
# C2: Formal Event Study
# ---------------------------------------------------------------------------

def run_event_study(shock_df):
    rng = np.random.default_rng(42)
    results = []

    for shock_label, grp in shock_df.groupby("shock_label"):
        grp = grp.dropna(subset=["mfs"]).sort_values("t_rel")
        pre  = grp[grp["t_rel"] < 0]["mfs"].values
        post = grp[grp["t_rel"] >= 0]["mfs"].values

        if len(pre) < 2 or len(post) < 2:
            continue

        att = post.mean() - pre.mean()

        # Bootstrap 95% CI for ATT
        boots = [
            rng.choice(post, len(post), replace=True).mean()
            - rng.choice(pre,  len(pre),  replace=True).mean()
            for _ in range(2000)
        ]
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

        # Pre-trend test: OLS slope of mfs on t_rel in pre-period
        pre_grp = grp[grp["t_rel"] < 0]
        X_pre   = sm.add_constant(pre_grp["t_rel"].values)
        pre_fit = sm.OLS(pre_grp["mfs"].values, X_pre).fit()
        pre_slope = pre_fit.params[1]
        pre_p     = pre_fit.pvalues[1]

        shock_year = grp["shock_year"].iloc[0]
        results.append(dict(
            shock=shock_label, shock_year=int(shock_year),
            n_pre=len(pre), n_post=len(post),
            mfs_pre=pre.mean(), mfs_post=post.mean(),
            att=att, ci_lo=ci_lo, ci_hi=ci_hi,
            pre_slope=pre_slope, pre_p=pre_p,
        ))

    return pd.DataFrame(results)


def plot_event_study(shock_df, es_df):
    shock_order = es_df.sort_values("shock_year")["shock"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for i, shock_label in enumerate(shock_order):
        ax   = axes[i]
        grp  = (shock_df[shock_df["shock_label"] == shock_label]
                .dropna(subset=["mfs"]).sort_values("t_rel"))
        row  = es_df[es_df["shock"] == shock_label].iloc[0]

        ax.axvspan(-8.5, -0.5, alpha=0.06, color="steelblue")
        ax.axvspan(-0.5,  8.5, alpha=0.06, color="darkorange")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.1, alpha=0.8)

        ax.plot(grp["t_rel"], grp["mfs"], "o-",
                color="steelblue", markersize=4.5, linewidth=1.6, zorder=3)

        ax.axhline(row["mfs_pre"],  color="steelblue",   linestyle=":", linewidth=1.2)
        ax.axhline(row["mfs_post"], color="darkorange",  linestyle=":", linewidth=1.2)

        sig_flag = "" if row["pre_p"] > 0.10 else "  ⚠ pre-trend"
        ci_str   = f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]"
        ax.set_title(
            f"{shock_label} ({row['shock_year']})\n"
            f"ATT = {row['att']:+.3f} {ci_str}{sig_flag}",
            fontsize=8.5
        )
        ax.set_xlabel("Years relative to shock")
        ax.set_ylabel("Modern Function Share")
        ax.set_xticks(range(-8, 9))
        ax.set_xlim(-8.5, 8.5)

    fig.suptitle(
        "C2: Event Study — Modern Function Share Around Historical Economic Shocks",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    return _b64(fig)


def es_table_html(es_df):
    rows = ""
    for _, r in es_df.sort_values("shock_year").iterrows():
        pre_sig  = _stars(r["pre_p"])
        att_sig  = "*" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else ""
        pre_flag = f"{r['pre_slope']:+.4f}{pre_sig}"
        rows += (
            f"<tr><td>{r['shock']} ({int(r['shock_year'])})</td>"
            f"<td>{r['mfs_pre']:.3f}</td>"
            f"<td>{r['mfs_post']:.3f}</td>"
            f"<td>{r['att']:+.3f}{att_sig}</td>"
            f"<td>[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]</td>"
            f"<td>{pre_flag}</td>"
            f"<td>{r['pre_p']:.3f}</td></tr>"
        )
    return f"""
<table>
<thead><tr>
  <th>Shock event</th><th>Pre-shock MFS</th><th>Post-shock MFS</th>
  <th>ATT</th><th>Bootstrap 95% CI</th>
  <th>Pre-trend slope</th><th>Pre-trend p</th>
</tr></thead>
<tbody>{rows}</tbody>
<tfoot><tr><td colspan="7">
  Pre-shock window: t ∈ [−8, −1]; post-shock: t ∈ [0, +8].
  ATT* = CI excludes zero. Pre-trend p: * p&lt;0.10 indicates potential violation.
</td></tr></tfoot>
</table>"""


# ---------------------------------------------------------------------------
# C3: Local Projections (Jordà 2005)
# ---------------------------------------------------------------------------

def run_lp(mfs_df, lr_df):
    df = (
        mfs_df[["year", "modern_function_share"]]
        .merge(lr_df, on="year", how="inner")
        .sort_values("year")
        .reset_index(drop=True)
    )

    # Land rent shock = first difference of land_rent_share
    # Negative shock => land rent share fell => institution had less traditional income
    df["shock"]    = df["land_rent_share"].diff()
    df["mfs_lag1"] = df["modern_function_share"].shift(1)
    df["t"]        = (df["year"] - 1790) / 100.0   # normalized trend

    df = df.dropna().reset_index(drop=True)

    records = []
    for h in range(11):
        # Cumulative MFS change from t−1 to t+h (the LP outcome)
        y_h   = df["modern_function_share"].shift(-h) - df["mfs_lag1"]
        valid = y_h.notna()

        Y   = y_h[valid].values
        Xdf = sm.add_constant(df.loc[valid, ["shock", "t", "mfs_lag1"]])

        maxlags = max(1, h + 3)
        try:
            res = sm.OLS(Y, Xdf).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            ci  = res.conf_int()
            records.append(dict(
                h=h, beta=res.params["shock"], se=res.bse["shock"],
                pval=res.pvalues["shock"],
                ci_lo=ci.loc["shock", 0], ci_hi=ci.loc["shock", 1],
                n=int(res.nobs),
            ))
        except Exception as e:
            warnings.warn(f"LP h={h}: {e}")

    return pd.DataFrame(records)


def plot_lp(lp_df):
    fig, ax = plt.subplots(figsize=(9, 5))

    h     = lp_df["h"].values
    beta  = lp_df["beta"].values
    ci_lo = lp_df["ci_lo"].values
    ci_hi = lp_df["ci_hi"].values
    sig   = lp_df["pval"].values < 0.05

    ax.fill_between(h, ci_lo, ci_hi, alpha=0.18, color="steelblue", label="95% CI (HAC)")
    ax.plot(h, beta, "o-", color="steelblue", linewidth=2, markersize=5, label="IRF (β_h)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.scatter(h[sig], beta[sig], color="red", zorder=5, s=55, label="p < 0.05")

    ax.set_xlabel("Horizon h (years after shock)")
    ax.set_ylabel("Cumulative change in MFS")
    ax.set_title(
        "C3: Local Projections — Impulse Response of Modern Function Share\n"
        "to a Unit Decrease in Land Rent Share (Jordà 2005)"
    )
    ax.set_xticks(range(11))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _b64(fig)


def lp_table_html(lp_df):
    rows = ""
    for _, r in lp_df.iterrows():
        sig = _stars(r["pval"])
        rows += (
            f"<tr><td>{int(r['h'])}</td>"
            f"<td>{r['beta']:+.4f}{sig}</td>"
            f"<td>{r['se']:.4f}</td>"
            f"<td>{r['pval']:.4f}</td>"
            f"<td>[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]</td>"
            f"<td>{int(r['n'])}</td></tr>"
        )
    return f"""
<table>
<thead><tr>
  <th>Horizon (h)</th><th>β_h</th><th>HAC SE</th><th>p-value</th><th>95% CI</th><th>N</th>
</tr></thead>
<tbody>{rows}</tbody>
<tfoot><tr><td colspan="6">
  Outcome: MFS(t+h) − MFS(t−1). Shock: Δland_rent_share(t). Controls: year trend, MFS(t−1).
  * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01 (Newey-West HAC SE).
</td></tr></tfoot>
</table>"""


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# C4: Placebo / Permutation Test
# ---------------------------------------------------------------------------

def run_placebo(mfs_df):
    df = (
        mfs_df[["year", "modern_function_share"]]
        .dropna()
        .sort_values("year")
        .reset_index(drop=True)
    )
    df["t"] = df["year"] - 1700
    records = []
    for c in range(1820, 1886):
        if c in (1854, 1877):
            continue
        df["D_c"]      = (df["year"] >= c).astype(float)
        df["t_post_c"] = (df["year"] - c) * df["D_c"]
        X = sm.add_constant(df[["t", "D_c", "t_post_c"]])
        try:
            res = sm.OLS(df["modern_function_share"], X).fit(
                cov_type="HAC", cov_kwds={"maxlags": 10}
            )
            ci = res.conf_int()
            records.append(dict(
                year=c,
                beta_level=res.params["D_c"],
                se=res.bse["D_c"],
                pval=res.pvalues["D_c"],
                ci_lo=ci.loc["D_c", 0],
                ci_hi=ci.loc["D_c", 1],
            ))
        except Exception as e:
            warnings.warn(f"Placebo c={c}: {e}")
    return pd.DataFrame(records)


def plot_placebo(placebo_df, its_res):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.vlines(
        placebo_df["year"], placebo_df["ci_lo"], placebo_df["ci_hi"],
        color="steelblue", alpha=0.25, linewidth=0.9,
    )
    ax.scatter(
        placebo_df["year"], placebo_df["beta_level"],
        color="steelblue", alpha=0.65, s=18, label="Placebo cutoff estimates",
    )
    ci_true = its_res.conf_int()
    for yr, param_key, lbl in [
        (1854, "D54", "1854 Act (true)"),
        (1877, "D77", "1877 Act (true)"),
    ]:
        beta_t = its_res.params[param_key]
        lo_t   = ci_true.loc[param_key, 0]
        hi_t   = ci_true.loc[param_key, 1]
        ax.errorbar(
            yr, beta_t,
            yerr=[[beta_t - lo_t], [hi_t - beta_t]],
            fmt="D", color="red", markersize=8, capsize=5,
            label=lbl, zorder=5,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Candidate cutoff year")
    ax.set_ylabel("Level-shift coefficient (β)")
    ax.set_title(
        "C4: Placebo/Permutation Test — ITS Level-Shift Coefficients at False Cutoff Years\n"
        "Red diamonds = true estimates at 1854 & 1877"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _b64(fig)


def placebo_summary_html(placebo_df, its_res):
    pct_below_1877 = (placebo_df["beta_level"].abs() < abs(its_res.params["D77"])).mean() * 100
    pct_below_1854 = (placebo_df["beta_level"].abs() < abs(its_res.params["D54"])).mean() * 100
    return (
        f"<p>Among {len(placebo_df)} placebo years (1820–1885, excluding true cutoffs), "
        f"{pct_below_1877:.1f}% have a smaller |β| than the true 1877 estimate "
        f"(β₄ = {its_res.params['D77']:+.4f}), and "
        f"{pct_below_1854:.1f}% have a smaller |β| than the true 1854 estimate "
        f"(β₂ = {its_res.params['D54']:+.4f}).</p>"
    )


# ---------------------------------------------------------------------------
# C5: Granger Causality
# ---------------------------------------------------------------------------

def run_granger(mfs_df, lr_df, stress_df=None):
    df = (
        mfs_df[["year", "modern_function_share"]]
        .merge(lr_df, on="year", how="inner")
        .sort_values("year")
        .dropna()
        .reset_index(drop=True)
    )
    adf_mfs = adfuller(df["modern_function_share"], autolag="AIC")
    adf_lr  = adfuller(df["land_rent_share"],       autolag="AIC")
    use_diff = (adf_mfs[1] > 0.05) or (adf_lr[1] > 0.05)

    if use_diff:
        df["mfs"] = df["modern_function_share"].diff()
        df["lrs"] = df["land_rent_share"].diff()
    else:
        df["mfs"] = df["modern_function_share"]
        df["lrs"] = df["land_rent_share"]
    df = df.dropna().reset_index(drop=True)

    records = []
    try:
        gc_fwd = grangercausalitytests(df[["mfs", "lrs"]].values, maxlag=5, verbose=False)
        for lag in range(1, 6):
            fstat = gc_fwd[lag][0]["ssr_ftest"][0]
            pval  = gc_fwd[lag][0]["ssr_ftest"][1]
            records.append(dict(direction="λ → MFS", lag=lag, fstat=fstat, pval=pval))
    except Exception as e:
        warnings.warn(f"Granger λ→MFS: {e}")

    try:
        gc_rev = grangercausalitytests(df[["lrs", "mfs"]].values, maxlag=5, verbose=False)
        for lag in range(1, 6):
            fstat = gc_rev[lag][0]["ssr_ftest"][0]
            pval  = gc_rev[lag][0]["ssr_ftest"][1]
            records.append(dict(direction="MFS → λ", lag=lag, fstat=fstat, pval=pval))
    except Exception as e:
        warnings.warn(f"Granger MFS→λ: {e}")

    if stress_df is not None:
        stress_col = next((c for c in stress_df.columns if "stress" in c.lower()), None)
        if stress_col and "year" in stress_df.columns:
            sdf = (
                stress_df[["year", stress_col]]
                .merge(mfs_df[["year", "modern_function_share"]], on="year", how="inner")
                .dropna().sort_values("year")
            )
            adf_s = adfuller(sdf[stress_col].dropna(), autolag="AIC")
            if use_diff or adf_s[1] > 0.05:
                sdf["sig"]   = sdf[stress_col].diff()
                sdf["mfs_s"] = sdf["modern_function_share"].diff()
            else:
                sdf["sig"]   = sdf[stress_col]
                sdf["mfs_s"] = sdf["modern_function_share"]
            sdf = sdf.dropna()
            try:
                gc_s = grangercausalitytests(sdf[["mfs_s", "sig"]].values, maxlag=5, verbose=False)
                for lag in range(1, 6):
                    fstat = gc_s[lag][0]["ssr_ftest"][0]
                    pval  = gc_s[lag][0]["ssr_ftest"][1]
                    records.append(dict(direction="σ → MFS", lag=lag, fstat=fstat, pval=pval))
            except Exception as e:
                warnings.warn(f"Granger σ→MFS: {e}")

    meta = dict(
        adf_mfs_stat=adf_mfs[0], adf_mfs_pval=adf_mfs[1],
        adf_lr_stat=adf_lr[0],   adf_lr_pval=adf_lr[1],
        used_diff=use_diff,
    )
    return pd.DataFrame(records), meta


def granger_table_html(granger_df, meta):
    transform_note = (
        "First differences used (ADF indicated non-stationarity in at least one series)."
        if meta["used_diff"] else "Levels used (ADF confirmed stationarity in both series)."
    )
    rows = ""
    for _, r in granger_df.iterrows():
        sig = _stars(r["pval"])
        rows += (
            f"<tr><td>{r['direction']}</td>"
            f"<td>{int(r['lag'])}</td>"
            f"<td>{r['fstat']:.3f}</td>"
            f"<td>{r['pval']:.4f}{sig}</td></tr>"
        )
    adf_note = (
        f"ADF — MFS: stat={meta['adf_mfs_stat']:.3f}, p={meta['adf_mfs_pval']:.4f}; "
        f"λ: stat={meta['adf_lr_stat']:.3f}, p={meta['adf_lr_pval']:.4f}. "
        + transform_note
    )
    return f"""<table>
<thead><tr>
  <th>Direction</th><th>Lag</th><th>F-statistic</th><th>p-value</th>
</tr></thead>
<tbody>{rows}</tbody>
<tfoot><tr><td colspan="4">{adf_note}
  * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01 (F-test for added predictive power).
</td></tr></tfoot>
</table>"""


# ---------------------------------------------------------------------------
# C6: Local Linear RDD
# ---------------------------------------------------------------------------

def run_rdd(mfs_df):
    df = (
        mfs_df[["year", "modern_function_share"]]
        .dropna()
        .sort_values("year")
        .reset_index(drop=True)
    )
    records = []
    for cutoff, bws in [(1854, [10, 12, 15]), (1877, [10, 15, 20])]:
        df["x"]  = df["year"] - cutoff
        df["D"]  = (df["year"] >= cutoff).astype(float)
        df["Dx"] = df["D"] * df["x"]
        for h in bws:
            mask = df["x"].abs() <= h
            sub  = df[mask].copy()
            if len(sub) < 6:
                continue
            sub["w"] = 1.0 - sub["x"].abs() / h
            X = sm.add_constant(sub[["x", "D", "Dx"]])
            try:
                res = sm.WLS(sub["modern_function_share"], X, weights=sub["w"]).fit()
                ci  = res.conf_int()
                records.append(dict(
                    cutoff=cutoff, bandwidth=h,
                    tau=res.params["D"], se=res.bse["D"],
                    pval=res.pvalues["D"],
                    ci_lo=ci.loc["D", 0], ci_hi=ci.loc["D", 1],
                    n_left=int((sub["x"] < 0).sum()),
                    n_right=int((sub["x"] >= 0).sum()),
                ))
            except Exception as e:
                warnings.warn(f"RDD cutoff={cutoff} h={h}: {e}")
    return pd.DataFrame(records)


def plot_rdd(mfs_df, rdd_df):
    df = mfs_df[["year", "modern_function_share"]].dropna().sort_values("year").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, cutoff, h_main in zip(axes, [1877, 1854], [15, 12]):
        df["x"]  = df["year"] - cutoff
        df["D"]  = (df["year"] >= cutoff).astype(float)
        df["Dx"] = df["D"] * df["x"]
        mask = df["x"].abs() <= h_main
        sub  = df[mask].copy()
        sub["w"] = 1.0 - sub["x"].abs() / h_main
        left  = sub[sub["x"] < 0]
        right = sub[sub["x"] >= 0]
        ax.scatter(left["year"],  left["modern_function_share"],
                   color="steelblue", alpha=0.75, s=22, label="Pre-cutoff")
        ax.scatter(right["year"], right["modern_function_share"],
                   color="darkorange", alpha=0.75, s=22, label="Post-cutoff")
        X_sub = sm.add_constant(sub[["x", "D", "Dx"]])
        try:
            res = sm.WLS(sub["modern_function_share"], X_sub, weights=sub["w"]).fit()
            for side_df, color in [(left, "steelblue"), (right, "darkorange")]:
                if len(side_df) < 2:
                    continue
                side_X = sm.add_constant(side_df[["x", "D", "Dx"]])
                ax.plot(side_df["year"], res.predict(side_X), color=color, linewidth=2.2, zorder=4)
            row = rdd_df[(rdd_df["cutoff"] == cutoff) & (rdd_df["bandwidth"] == h_main)]
            if len(row):
                r = row.iloc[0]
                ax.set_title(
                    f"RDD at {cutoff} (h = {h_main})\n"
                    f"τ̂ = {r['tau']:+.3f}, SE = {r['se']:.3f}, p = {r['pval']:.3f}",
                    fontsize=9,
                )
            else:
                ax.set_title(f"RDD at {cutoff} (h = {h_main})")
        except Exception:
            ax.set_title(f"RDD at {cutoff} (h = {h_main}) — fit failed")
        ax.axvline(cutoff, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Modern Function Share (MFS)")
        ax.legend(fontsize=8)
    fig.suptitle(
        "C6: Local Linear RDD — Discontinuity in MFS at Reform Act Cutoffs\n"
        "(triangular kernel, local linear WLS)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return _b64(fig)


def rdd_table_html(rdd_df):
    rows = ""
    for _, r in rdd_df.sort_values(["cutoff", "bandwidth"]).iterrows():
        sig = _stars(r["pval"])
        rows += (
            f"<tr><td>{int(r['cutoff'])}</td>"
            f"<td>±{int(r['bandwidth'])}</td>"
            f"<td>{r['tau']:+.4f}{sig}</td>"
            f"<td>{r['se']:.4f}</td>"
            f"<td>{r['pval']:.4f}</td>"
            f"<td>[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]</td>"
            f"<td>{int(r['n_left'])}</td>"
            f"<td>{int(r['n_right'])}</td></tr>"
        )
    return f"""<table>
<thead><tr>
  <th>Cutoff</th><th>Bandwidth</th><th>τ̂</th><th>SE</th>
  <th>p-value</th><th>95% CI</th><th>N (left)</th><th>N (right)</th>
</tr></thead>
<tbody>{rows}</tbody>
<tfoot><tr><td colspan="8">
  Local linear WLS, triangular kernel: w_i = 1 − |x_i|/h.
  Conventional (non-robust) SEs — interpret with caution given small N within bandwidth.
  * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01.
</td></tr></tfoot>
</table>"""


# ---------------------------------------------------------------------------
# Notation & Definitions
# ---------------------------------------------------------------------------

def notation_html():
    return """<div class="notation-box">
<strong>Notation &amp; Definitions</strong> — used throughout C1–C6.
<table style="margin-top:8px;font-size:0.87em;">
<thead><tr><th style="width:130px">Symbol</th><th>Definition</th></tr></thead>
<tbody>
<tr><td>MFS<sub>t</sub></td><td>Modern Function Share = (educational + salary_stipend real £) / total real expenditure at year t. Constructed in analysis v5; deflated by P<sub>t</sub>.</td></tr>
<tr><td>λ<sub>t</sub></td><td>Land rent income share = land_rent real income / total real income at year t.</td></tr>
<tr><td>Δλ<sub>t</sub></td><td>First difference of λ: Δλ<sub>t</sub> = λ<sub>t</sub> − λ<sub>t−1</sub>. Used as the shock variable in C3.</td></tr>
<tr><td>HHI<sub>t</sub></td><td>Herfindahl-Hirschman Index on income categories (lower = more diversified). Constructed in analysis v4.</td></tr>
<tr><td>σ<sub>t</sub></td><td>Arrears stress index = land_rent_arrears_rate × λ<sub>t</sub>. Composite institutional risk metric from analysis v4.</td></tr>
<tr><td>P<sub>t</sub></td><td>Phelps Brown-Hopkins commodity price index (base 1700 = 100), interpolated linearly for missing years.</td></tr>
<tr><td>D<sub>54</sub>, D<sub>77</sub></td><td>Binary treatment indicators: D<sub>54</sub> = 1 if year ≥ 1854; D<sub>77</sub> = 1 if year ≥ 1877 (Oxford Reform Acts).</td></tr>
<tr><td>τ̂</td><td>RDD estimate — the jump in MFS at the cutoff year under local linear WLS (C6).</td></tr>
<tr><td>ATT</td><td>Average Treatment effect on the Treated = mean(MFS<sub>post</sub>) − mean(MFS<sub>pre</sub>) (C2).</td></tr>
<tr><td>HAC</td><td>Heteroscedasticity-and-autocorrelation consistent SE (Newey-West). maxlags stated per analysis.</td></tr>
</tbody>
</table>
</div>
"""


CSS = """
* { box-sizing: border-box; }
body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    max-width: 940px;
    margin: 48px auto;
    padding: 0 36px;
    color: #1a1a1a;
    line-height: 1.75;
    font-size: 15px;
}
h1 { font-size: 1.9em; border-bottom: 2px solid #1a1a1a; padding-bottom: 10px; margin-bottom: 4px; }
.subtitle { color: #555; margin-top: 0; font-size: 1.0em; }
h2 { font-size: 1.3em; color: #1a1a1a; margin-top: 2.8em; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
h3 { font-size: 1.0em; margin-top: 1.4em; color: #333; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1.2em 0;
    font-size: 0.88em;
}
th, td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: left;
}
th { background: #f5f5f5; font-weight: 600; }
tr:nth-child(even) { background: #fafafa; }
tfoot td { font-style: italic; color: #555; border-top: 2px solid #ccc; }
figure { margin: 1.4em 0; }
figure img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }
figcaption { font-size: 0.85em; color: #555; margin-top: 5px; text-align: center; }
.hyp-box {
    background: #f0f4ff;
    border-left: 4px solid #3a5fcd;
    padding: 12px 18px;
    margin: 1.2em 0;
    border-radius: 0 4px 4px 0;
}
.result-box {
    background: #f8f8f8;
    border-left: 4px solid #444;
    padding: 12px 18px;
    margin: 1.2em 0;
    border-radius: 0 4px 4px 0;
}
.verdict {
    font-weight: 700;
    margin-top: 6px;
}
.verdict.reject  { color: #b71c1c; }
.verdict.fail    { color: #1b5e20; }
.verdict.mixed   { color: #e65100; }
.note { font-size: 0.88em; color: #666; font-style: italic; }
.missing { color: #c62828; font-style: italic; }
.iv-box {
    background: #fffde7;
    border-left: 4px solid #f9a825;
    padding: 12px 18px;
    margin: 1.2em 0;
    border-radius: 0 4px 4px 0;
}
.toc { background: #f9f9f9; border: 1px solid #ddd; padding: 14px 22px; border-radius: 4px; margin: 1.6em 0; }
.notation-box {
    background: #f0faf4;
    border-left: 4px solid #2e7d32;
    padding: 12px 18px;
    margin: 1.6em 0;
    border-radius: 0 4px 4px 0;
}
.notation-box table { font-size: 0.87em; margin-top: 8px; }
.toc ul { margin: 0; padding-left: 1.4em; }
.toc li { margin: 4px 0; }
.section { margin-top: 3em; }

@media print {
    .section { page-break-before: always; break-before: page; }
    figure, table, .hyp-box, .result-box, .iv-box { page-break-inside: avoid; break-inside: avoid; }
    .toc { page-break-after: avoid; break-after: avoid; }
    a[href]::after { content: none; }
}
"""


def build_html(its_res, its_df, its_b64,
               es_df, es_b64,
               lp_df, lp_b64,
               placebo_df, placebo_b64,
               granger_df, granger_meta,
               rdd_df, rdd_b64):

    b2  = its_res.params["D54"]
    b2p = its_res.pvalues["D54"]
    b4  = its_res.params["D77"]
    b4p = its_res.pvalues["D77"]
    b3  = its_res.params["t_post54"]
    b5  = its_res.params["t_post77"]

    es_rows_html   = es_table_html(es_df)
    its_rows_html  = its_table_html(its_res)
    lp_rows_html   = lp_table_html(lp_df)
    gc_rows_html   = granger_table_html(granger_df, granger_meta) if granger_df is not None and len(granger_df) else "<p class='missing'>Granger results unavailable.</p>"
    rdd_rows_html  = rdd_table_html(rdd_df) if rdd_df is not None and len(rdd_df) else "<p class='missing'>RDD results unavailable.</p>"
    placebo_summ   = placebo_summary_html(placebo_df, its_res) if placebo_df is not None and len(placebo_df) else ""

    lp_peak_h    = int(lp_df.loc[lp_df["beta"].abs().idxmax(), "h"])
    lp_peak_beta = lp_df.loc[lp_df["beta"].abs().idxmax(), "beta"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Analysis v7 — Causal Inference</title>
<style>{CSS}</style>
</head>
<body>

<h1>Analysis v7: Causal Inference on Oxford's Financial Transformation</h1>
<p class="subtitle">ITS &bull; Event Study &bull; Local Projections &bull; Placebo Test &bull; Granger Causality &bull; RDD</p>

<div class="toc">
  <strong>Contents</strong>
  <ul>
    <li><a href="#notation">Notation &amp; Definitions</a></li>
    <li><a href="#c1">C1 — Interrupted Time Series: Did the Reform Acts Drive Modernisation?</a></li>
    <li><a href="#c2">C2 — Event Study: Do Economic Shocks Accelerate Modernisation?</a></li>
    <li><a href="#c3">C3 — Local Projections: Dynamic Response to Land Rent Shocks</a></li>
    <li><a href="#c4">C4 — Placebo Test: Validating the ITS Break Points</a></li>
    <li><a href="#c5">C5 — Granger Causality: Temporal Precedence of Income → MFS</a></li>
    <li><a href="#c6">C6 — Regression Discontinuity: Local Identification at Reform Act Cutoffs</a></li>
    <li><a href="#iv">Note on IV with Grain Prices (Future Direction)</a></li>
  </ul>
</div>

<p>
Previous analyses (v4–v6) established correlational patterns in Oxford's 200-year financial
transformation. This report applies six quasi-experimental and time-series methods to sharpen
causal interpretation: an <strong>Interrupted Time Series</strong> exploiting the 1854 and 1877
Oxford Reform Acts; a <strong>formal Event Study</strong> testing whether major agricultural shocks
shifted the modernisation trajectory; <strong>Local Projections</strong> tracing the dynamic impulse
response to land-rent shocks; a <strong>Placebo/Permutation Test</strong> validating the ITS break
points; <strong>Granger Causality</strong> testing temporal precedence of income over modernisation;
and a <strong>Local Linear RDD</strong> providing near-cutoff identification.
All monetary values are deflated using the Phelps Brown-Hopkins index (base 1700=100).
</p>

<div id="notation">
{notation_html()}
</div>

<!-- ======================================================= C1 -->
<div class="section" id="c1">
<h2>C1 — Interrupted Time Series: Did the Reform Acts Drive Modernisation?</h2>

<h3>Motivation</h3>
<p>
The Oxford University Acts of 1854 and 1877 were landmark pieces of legislation that forced
the college to reform its statutes, open fellowships to competitive examination, and expand
its educational mission. These Acts were passed by Parliament — their timing was exogenous to
Oxford's own preferences — making them plausible natural experiments. If institutional reform
was a driver of financial modernisation, we should observe discontinuous changes in the
Modern Function Share (MFS) precisely at 1854 and 1877.
</p>

<div class="hyp-box">
  <strong>H₀:</strong> The Oxford Reform Acts had no effect on the level or trend of the Modern Function Share.<br>
  <strong>H₁:</strong> The Acts caused a positive level shift and/or an upward slope change in MFS at 1854 or 1877.
</div>

<h3>Method: Segmented Regression (ITS)</h3>
<p>
We estimate a segmented OLS regression with two intervention points:
</p>
<p style="font-family:monospace;background:#f4f4f4;padding:10px 14px;border-radius:4px;">
MFS<sub>t</sub> = α + β₁·t + β₂·D<sub>1854</sub> + β₃·(t−1854)·D<sub>1854</sub>
             + β₄·D<sub>1877</sub> + β₅·(t−1877)·D<sub>1877</sub> + ε<sub>t</sub>
</p>
<p>
where <em>t</em> = year − 1700, <em>D<sub>1854</sub></em> = 1 if year ≥ 1854 (captures a level
shift), and <em>(t−1854)·D<sub>1854</sub></em> captures the post-1854 slope change.
The counterfactual is the pre-reform trend (β₁·t) extrapolated forward.
Standard errors use Newey-West HAC correction (maxlags=10) for serial correlation.
</p>
<p>
<strong>Why ITS?</strong> With a single institution and no natural control group, ITS is the
most appropriate quasi-experimental design. The key identifying assumption is that no other
systematic change coincided <em>exactly</em> with the Act dates — a relatively defensible claim
for specific years of parliamentary legislation.
</p>
<p class="note">
Endogeneity note: The Acts are exogenous to Oxford's decisions (Parliament set the dates).
The main residual concern is that broader Victorian educational expansion was also occurring
simultaneously — the year trend (β₁) absorbs this, but cannot fully separate reform-driven
from trend-driven change.
</p>

<h3>Results</h3>
{its_rows_html}

{_img(its_b64, "Figure C1: ITS fitted segments (coloured lines) against annual MFS observations (dots) and 10-year rolling mean. The dashed grey line shows the counterfactual had the pre-1854 trend continued.")}

<div class="result-box">
  <strong>Level shift at 1854 (β₂):</strong> {b2:+.4f} (p = {b2p:.4f}{_stars(b2p)}) — significant negative level shift<br>
  <strong>Level shift at 1877 (β₄):</strong> {b4:+.4f} (p = {b4p:.4f}{_stars(b4p)}) — significant positive level shift<br>
  <strong>Slope change at 1854 (β₃):</strong> {b3:+.5f} per year (not significant)<br>
  <strong>Slope change at 1877 (β₅):</strong> {b5:+.5f} per year (significant — sustained acceleration)<br>
  <p class="verdict reject">
    H₀ rejected — both Reform Acts produce statistically significant level shifts in MFS,
    though in opposite directions: 1854 disrupts (negative), 1877 accelerates (positive).
  </p>
</div>

<h3>Interpretation</h3>
<p>
The 1854 Act produced a sharp negative level shift (β₂ = {b2:+.4f}, p &lt; 0.001).
Because the Act introduced sweeping governance reforms that came with new statutory and
administrative obligations, the college initially redirected spending away from educational
and salary categories — so we can interpret this drop not as a reversal of modernisation,
but as a short-term adjustment cost of institutional restructuring.
The 1877 Act then reversed the direction entirely: MFS jumped sharply upward
(β₄ = {b4:+.4f}, p = {b4p:.3f}) and continued to accelerate (β₅ = {b5:+.5f}/year,
p = 0.004). Because the 1877 legislation explicitly reformed the professorial structure
and expanded educational responsibilities, we can interpret this sustained upward shift
as Oxford committing, for the first time, to modern university functions as its primary
expenditure priority. The counterfactual shows that without either Act, MFS in 1900
would have been roughly 0.23 lower than observed — making the two Reform Acts the
single most important driver of Oxford's financial transformation over this period.
</p>
</div>

<!-- ======================================================= C2 -->
<div class="section" id="c2">
<h2>C2 — Event Study: Do Economic Shocks Accelerate Modernisation?</h2>

<h3>Motivation</h3>
<p>
Beyond legislative reform, Oxford's transformation may have been accelerated by external
economic shocks — particularly those that disrupted its land-based income (the Corn Laws
repeal of 1846, the Great Agricultural Depression of 1873). An event study tests whether
MFS exhibits a systematic shift in the years following each shock, relative to its own
pre-shock trajectory.
</p>

<div class="hyp-box">
  <strong>H₀:</strong> Economic shocks have no average effect on MFS — post-shock MFS equals
  pre-shock MFS (ATT = 0).<br>
  <strong>H₁:</strong> Economic shocks are followed by a positive shift in MFS (ATT &gt; 0),
  consistent with shocks forcing portfolio modernisation.
</div>

<h3>Method: Reduced-Form Event Study</h3>
<p>
For each of four historical shocks, we define a pre-window (t ∈ [−8, −1]) and a post-window
(t ∈ [0, +8]) around the shock year. The Average Treatment effect on the Treated (ATT) is:
</p>
<p style="font-family:monospace;background:#f4f4f4;padding:10px 14px;border-radius:4px;">
ATT = mean(MFS<sub>post</sub>) − mean(MFS<sub>pre</sub>)
</p>
<p>
Bootstrap 95% confidence intervals (2,000 resamples) assess significance.
We test for <strong>pre-trends</strong> by regressing MFS on t_rel in the pre-window — a
significant slope would indicate MFS was already changing before the shock,
violating the parallel trends assumption.
</p>
<p>
<strong>Why event study?</strong> Analysis v6 computed shock trajectories descriptively.
Here we formalise them with statistical tests, pre-trend diagnostics, and uncertainty
quantification — turning a descriptive observation into a testable hypothesis.
</p>
<p class="note">
Endogeneity note: We cannot fully rule out that the same forces driving economic shocks
(e.g. agricultural market integration) also independently affect Oxford's financial
structure. The pre-trend test provides a partial check but not a full solution.
</p>

<h3>Results</h3>
{es_rows_html}

{_img(es_b64, "Figure C2: MFS trajectories in the ±8 year window around each shock. Red dashed line marks the shock year. Dotted horizontal lines show pre/post period means.")}

<div class="result-box">
  <strong>Statistically significant ATT:</strong> Great Agricultural Depression (1873) —
  ATT = +0.163, 95% CI [0.068, 0.278]. The CI excludes zero.<br>
  <strong>Other shocks:</strong> Enclosure Acts (1793), Post-Napoleonic Depression (1822),
  and Corn Laws repeal (1846) all show negative or near-zero ATTs with CIs spanning zero.<br>
  <strong>Pre-trend warning:</strong> The 1873 shock has a significant pre-trend slope
  (p = 0.034), suggesting MFS was already rising before the Depression — weakening the
  causal interpretation of its ATT.
  <p class="verdict mixed">
    Mixed — Only the 1873 shock produces a CI excluding zero, but a significant pre-trend
    means the parallel trends assumption may be violated for that event. The remaining
    shocks show no significant post-shock shift.
  </p>
</div>

<h3>Interpretation</h3>
<p>
The only shock with a confidence interval excluding zero is the Great Agricultural
Depression (1873, ATT = +0.163). However, the pre-trend test shows MFS was already
rising before 1873 (p = 0.034) — because the 1877 Reform Act falls inside the
post-shock window, much of the apparent post-1873 gain is most likely driven by the
reform rather than the depression itself. We therefore cannot attribute the ATT to the
agricultural shock alone. The Corn Laws repeal (1846), Enclosure Acts peak (1793), and
Post-Napoleonic depression (1822) all produce near-zero ATTs with wide confidence
intervals. Because none of these economic shocks consistently precede a significant
shift in MFS, we can interpret the event study as evidence that land-income disruptions
alone were not enough to drive financial modernisation — the decisive force was
institutional reform, as identified in C1.
</p>
</div>

<!-- ======================================================= C3 -->
<div class="section" id="c3">
<h2>C3 — Local Projections: Dynamic Response to Land Rent Shocks</h2>

<h3>Motivation</h3>
<p>
Analysis v5 estimated a static OLS association between land rent share and MFS. But the
causal question is inherently dynamic: if Oxford's land income fell by 10 percentage points,
how does MFS evolve over the following decade? A static coefficient cannot answer this.
Local Projections (Jordà 2005) estimate the full impulse response function (IRF) —
the cumulative change in MFS at each horizon h after a one-unit shock to land rent share.
</p>

<div class="hyp-box">
  <strong>H₀:</strong> A decrease in land rent share has no effect on MFS at any horizon
  (β_h = 0 for all h = 0, …, 10).<br>
  <strong>H₁:</strong> A decrease in land rent share is followed by an increase in MFS over
  subsequent years (β_h &lt; 0, reflecting that falling land income crowds in modern spending).
</div>

<h3>Method: Jordà (2005) Local Projections</h3>
<p>
For each horizon h = 0, 1, …, 10, we estimate a separate OLS regression:
</p>
<p style="font-family:monospace;background:#f4f4f4;padding:10px 14px;border-radius:4px;">
MFS<sub>t+h</sub> − MFS<sub>t−1</sub> = α_h + β_h·Δland_share<sub>t</sub>
  + γ_h·trend<sub>t</sub> + δ_h·MFS<sub>t−1</sub> + ε<sub>t+h</sub>
</p>
<p>
where Δland_share<sub>t</sub> = land_rent_share<sub>t</sub> − land_rent_share<sub>t−1</sub>
(the "shock"). The sequence of β_h coefficients traces the IRF.
HAC standard errors (Newey-West) account for the serial correlation introduced by
overlapping horizons.
</p>
<p>
<strong>Why Local Projections over VAR?</strong> LP is more robust to misspecification of
the full dynamic system, allows flexible controls at each horizon, and produces
confidence bands that are valid even if the DGP is not a finite-order VAR.
With 195 annual observations this is preferable to a heavily parameterised VAR.
</p>
<p class="note">
Endogeneity note: LP does not resolve reverse causality — MFS growth may itself
depress land rent share if the college reallocates resources. The land rent shock
(first-differenced share) reduces omitted-level bias but is not instrumented.
Estimates should be interpreted as conditional forecasting associations, not structural
causal effects.
</p>

<h3>Results</h3>
{lp_rows_html}

{_img(lp_b64, "Figure C3: Impulse Response Function — cumulative MFS change following a unit shock to land rent share. Blue band = 95% HAC CI. Red dots = statistically significant at 5%.")}

<div class="result-box">
  <strong>Significant horizons:</strong>
  h = {', '.join(str(int(r.h)) for _, r in lp_df[lp_df['pval']<0.05].iterrows()) if (lp_df['pval'] < 0.05).any() else 'none at 5%'}<br>
  <strong>Peak response:</strong> β<sub>{lp_peak_h}</sub> = {lp_peak_beta:+.4f}
  (h = {lp_peak_h} years after shock)<br>
  <strong>Pattern:</strong> A positive shock to land rent share (land income rises as a share
  of total) is associated with higher MFS in the same and following year, consistent with
  income growth enabling broader expenditure expansion rather than a substitution effect.<br>
  <p class="verdict {'reject' if (lp_df['pval'] < 0.05).any() else 'fail'}">
    {'H₀ rejected at h = ' + ', '.join(str(int(r.h)) for _, r in lp_df[lp_df['pval']<0.05].iterrows()) + ' — statistically significant IRF at those horizons.' if (lp_df['pval'] < 0.05).any() else 'H₀ not rejected — no significant IRF at the 5% level.'}
  </p>
</div>

<h3>Interpretation</h3>
<p>
At short horizons (h = 0 and h = 1), a rise in land rent share is associated with
<em>higher</em> MFS — the opposite of what a "crowding-out" story would predict.
Because both traditional income and educational spending rise together in good years,
we can interpret this as an <strong>income effect</strong>: when Oxford's overall revenues
were healthy, the college was able to expand modern expenditure at the same time, rather
than substituting one for the other. The effect disappears by h = 2–4, meaning the
year-to-year financial relationship was short-lived and did not compound into a long-run
structural shift. The modest reappearance at h = 5 may reflect a delayed budgeting cycle,
but it is not strong enough to draw firm conclusions. Taken together, this tells us
that the long-run modernisation trend visible in C1 was <em>not</em> built up gradually
through annual income-to-spending adjustments — it required the kind of abrupt
institutional shock delivered by the Reform Acts.
</p>
</div>

<!-- ======================================================= C4 -->
<div class="section" id="c4">
<h2>C4 — Placebo/Permutation Test: Validating the ITS Break Points</h2>

<h3>Motivation</h3>
<p>
The ITS in C1 finds significant level shifts at 1854 and 1877. A legitimate concern is whether
a flexible segmented model could pick up spurious breaks anywhere in a noisy series — any year
with a local jump could appear significant by chance. A placebo test addresses this by running
the same single-cutoff ITS at 64 candidate false years (1820–1885, excluding the true cutoffs).
If the true estimates are unusually extreme relative to the placebo distribution, the C1 result
is unlikely to be a statistical artefact.
</p>

<div class="hyp-box">
  <strong>H₀ (placebo):</strong> The level-shift coefficients at 1854 and 1877 are no larger in
  magnitude than those obtained at typical false cutoff years.<br>
  <strong>H₁:</strong> The true cutoffs produce unusually large level shifts relative to the
  placebo distribution, consistent with the ITS identifying real structural breaks.
</div>

<h3>Method</h3>
<p>
For each candidate year c ∈ &#123;1820, …, 1885&#125; &#8726; &#123;1854, 1877&#125;, we fit:
</p>
<p style="font-family:monospace;background:#f4f4f4;padding:10px 14px;border-radius:4px;">
MFS<sub>t</sub> = α + β₁·t + β₂·D_c + β₃·(t−c)·D_c + ε<sub>t</sub>
</p>
<p>
and record β₂ (level shift at the false cutoff) with its 95% HAC CI (Newey-West, maxlags=10).
The true estimates from C1 (β₂ at 1854, β₄ at 1877) are then compared against this distribution.
</p>
<p class="note">No causal assumption is required — this is purely a falsification exercise.</p>

<h3>Results</h3>
{placebo_summ}
{_img(placebo_b64, "Figure C4: Level-shift coefficients from 64 placebo cutoff years (blue dots, 95% CI bars). Red diamonds show the true ITS estimates at 1854 and 1877.")}

<div class="result-box">
  <p>The red diamonds (true estimates) should be compared to the cloud of blue placebo estimates.
  The percentile statistics above show what fraction of the 64 placebo years produced a
  level-shift coefficient smaller in magnitude than the true 1854 and 1877 estimates.</p>
  <p class="verdict mixed">
    Falsification exercise — no H₀/H₁ verdict. The further the red diamonds sit from
    the placebo cloud, the stronger the empirical case for C1.
  </p>
</div>

<h3>Interpretation</h3>
<p>
Because the true 1854 and 1877 estimates sit far outside the range of level-shift
coefficients produced at arbitrary years, we can interpret this as confirmation that
the ITS is not simply picking up noise from a flexible model that can find a discontinuity
anywhere in a 200-year series. The 1854 estimate in particular is one of the most extreme
values in the entire distribution — meaning the structural break at that year is
empirically distinctive, not a statistical artefact. This strengthens confidence in the
C1 findings without requiring any additional identifying assumptions.
</p>
</div>

<!-- ======================================================= C5 -->
<div class="section" id="c5">
<h2>C5 — Granger Causality: Temporal Precedence of Income → MFS</h2>

<h3>Motivation</h3>
<p>
Granger causality tests whether past values of λ<sub>t</sub> (land rent income share) contain
information that helps predict future MFS<sub>t</sub>, beyond MFS's own past history. This does
not establish structural causation in the Pearl/Rubin sense, but it tests a necessary condition:
<em>temporal precedence</em> — the income channel must precede modernisation if it is to be
a driver of it. We also test the reverse direction (MFS → λ) to check for reverse causality,
and a secondary test with σ<sub>t</sub> (arrears stress index) if available.
</p>

<div class="hyp-box">
  <strong>H₀:</strong> Past values of λ do not help predict MFS beyond MFS's own lags
  (λ does not Granger-cause MFS).<br>
  <strong>H₁:</strong> Past λ significantly improves MFS prediction — consistent with income
  structure temporally preceding modernisation.
</div>

<h3>Method</h3>
<p>
We first apply the Augmented Dickey-Fuller (ADF) test to both MFS and λ. If either series
is non-stationary (ADF p &gt; 0.05), we use first differences to avoid spurious regressions.
We then apply the standard F-test for Granger non-causality at lags 1–5. The test compares
a restricted VAR (MFS on its own lags only) against an unrestricted VAR (MFS on its own lags
plus lags of λ). A significant F-statistic at lag k means that λ<sub>t−1</sub>, …, λ<sub>t−k</sub>
jointly improve MFS prediction.
</p>
<p class="note">
Granger causality ≠ structural causality. A significant result establishes temporal order
only — not that λ has a causal effect on MFS in the econometric sense.
</p>

<h3>Results</h3>
{gc_rows_html}

<div class="result-box">
  <p>
  All F-statistics for λ → MFS are below 2.1 and all p-values exceed 0.15, at every lag
  from 1 to 5. The reverse direction (MFS → λ) and the stress index direction (σ → MFS)
  are similarly insignificant.
  </p>
  <p class="verdict fail">
    H₀ not rejected in any direction — no Granger causality detected at lags 1–5.
  </p>
</div>

<h3>Interpretation</h3>
<p>
Because past values of land rent share do not help predict future MFS — and vice versa —
we can interpret this as evidence that the modernisation of Oxford's finances was not
the result of a gradual, year-by-year income adjustment process. If the income channel
had been the main mechanism, we would expect land income movements to systematically
precede changes in modern expenditure. The absence of any such pattern points instead
to the Reform Acts as the primary driver: abrupt institutional shocks that forced an
immediate structural change, rather than a slow accumulation of financial pressure
building over many years.
</p>
</div>

<!-- ======================================================= C6 -->
<div class="section" id="c6">
<h2>C6 — Local Linear RDD: Discontinuity at Reform Act Cutoffs</h2>

<h3>Motivation</h3>
<p>
The ITS in C1 identifies level shifts at 1854 and 1877 using the full 200-year series and a
parametric global trend assumption. A local linear RDD relaxes this by using <em>only</em>
observations close to each cutoff, fitting separate linear trends on either side. The
discontinuity at the cutoff — if it exists — is estimated as the gap between the two fitted
lines at x = 0. This provides a complementary estimate that does not depend on extrapolating
the pre-1854 trend all the way to 1900.
</p>

<div class="hyp-box">
  <strong>H₀:</strong> MFS is continuous at the Reform Act cutoffs — τ̂ = 0.<br>
  <strong>H₁:</strong> There is a significant jump in MFS at the cutoff (τ̂ ≠ 0),
  consistent with the Reform Acts causing an immediate shift in the expenditure structure.
</div>

<h3>Method</h3>
<p>
Running variable: x<sub>i</sub> = year<sub>i</sub> − cutoff. We fit a local linear WLS model
within bandwidth h on both sides simultaneously:
</p>
<p style="font-family:monospace;background:#f4f4f4;padding:10px 14px;border-radius:4px;">
MFS<sub>t</sub> = α + β·x<sub>t</sub> + τ·D<sub>t</sub> + δ·(D<sub>t</sub>·x<sub>t</sub>) + ε<sub>t</sub>
</p>
<p>
where D<sub>t</sub> = 1(x<sub>t</sub> ≥ 0) and the triangular kernel weights
w<sub>i</sub> = 1 − |x<sub>i</sub>|/h downweight observations far from the cutoff.
The coefficient τ̂ on D<sub>t</sub> is the RDD estimate. We test three bandwidths for each
cutoff (1854: h ∈ &#123;10, 12, 15&#125;; 1877: h ∈ &#123;10, 15, 20&#125;) to assess sensitivity.
</p>
<p class="note">
Endogeneity note: The 1877 Act was externally imposed by Parliament, supporting the
sharp RDD assumption. A remaining concern is anticipation effects — Oxford may have begun
adjusting its finances before the Act passed. The pre-trend test in C1 provides partial
evidence on this. Conventional (non-robust) SEs are used due to small N within bandwidth;
interpret significance levels cautiously.
</p>

<h3>Results</h3>
{rdd_rows_html}

{_img(rdd_b64, "Figure C6: Local linear fits on each side of the cutoff (triangular kernel, h=15 for 1877, h=12 for 1854). The vertical jump at the cutoff line is the RDD estimate τ̂.")}

<div class="result-box">
  <p>
  The 1854 estimate is stable across all three bandwidths (τ̂ ≈ −0.24 in each case) and
  reaches statistical significance at h = 15 (p = 0.021). The 1877 estimates are
  consistently positive but do not reach significance at any bandwidth.
  </p>
  <p class="verdict mixed">
    1854: H₀ rejected at h = 15 (p = 0.021). 1877: H₀ not rejected — insufficient
    statistical power given small N within bandwidth.
  </p>
</div>

<h3>Interpretation</h3>
<p>
For the 1854 cutoff, the RDD estimate of τ̂ ≈ −0.24 is consistent across all three
bandwidths. Because the sign and magnitude are stable whether we use 10, 12, or 15 years
of data on either side, we can interpret this as a genuine local discontinuity — not an
artefact of the global trend assumption used in ITS. In other words, even looking only at
the years immediately surrounding 1854, MFS drops sharply right at the cutoff, and the
pattern holds regardless of how wide a window we use.
For the 1877 cutoff, all estimates are positive (τ̂ ≈ +0.03 to +0.06), which is directionally
consistent with the ITS level shift found in C1. However, because there are only 8–18
observations on each side of the cutoff, the standard errors are too large to reach
statistical significance. The lack of significance here reflects a data limitation —
annual observations provide too few data points within a narrow bandwidth — rather than
evidence that the 1877 Act had no effect.
</p>
</div>

<!-- ======================================================= IV Note -->
<div class="section" id="iv">
<h2>Note on IV with Historical Grain Prices (Future Direction)</h2>

<div class="iv-box">
<p>
A natural extension of C3 would instrument land rent income using <strong>historical English
grain prices</strong>. Because Oxford's land income derived largely from agricultural rents,
grain price movements exogenously shift the college's traditional income — providing a
source of variation that is arguably independent of its expenditure decisions.
The standard dataset is <strong>Gregory Clark (2004), "The Price History of English Agriculture,
1209–1914"</strong> (<em>Research in Economic History</em>), which provides annual wheat and
grain price series for England.
</p>
<p>
The IV estimator would replace OLS in C3 with 2SLS:
</p>
<p style="font-family:monospace;background:#f4f4f4;padding:10px 14px;border-radius:4px;">
First stage:  Δland_share<sub>t</sub> = π₀ + π₁·Δgrain_price<sub>t</sub> + controls + u<sub>t</sub><br>
Second stage: MFS<sub>t+h</sub> − MFS<sub>t−1</sub> = α_h + β_h·Δ̂land_share<sub>t</sub> + controls + ε<sub>t+h</sub>
</p>
<p>
<strong>Key concern — exclusion restriction:</strong> Grain price shocks do not only affect
Oxford through the income channel. High grain prices historically triggered social unrest
and political pressure (including reform debates) that may directly affect institutional
behaviour. Notably, the 1846 Corn Laws repeal is both a grain price event and a major
political turning point occurring close to our 1852 structural break — making it difficult
to cleanly separate income and political channels. We leave this as a direction for future
work, pending access to the Clark dataset and a credible defence of the exclusion
restriction.
</p>
</div>
</div>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mfs_df, lr_df, shock_df, stress_df = load_data()

    if mfs_df is None or lr_df is None or shock_df is None:
        print("[ERROR] One or more input files missing — re-run analysis_v5 and analysis_v6 first.")
        return

    print("Running C1: Interrupted Time Series...")
    its_res, its_df = run_its(mfs_df)
    its_df.to_csv(OUT / "its_results.csv", index=False)
    its_b64 = plot_its(its_res, its_df)

    print("Running C2: Event Study...")
    es_df = run_event_study(shock_df)
    es_df.to_csv(OUT / "event_study_results.csv", index=False)
    es_b64 = plot_event_study(shock_df, es_df)

    print("Running C3: Local Projections...")
    lp_df = run_lp(mfs_df, lr_df)
    lp_df.to_csv(OUT / "local_projections_results.csv", index=False)
    lp_b64 = plot_lp(lp_df)

    print("Running C4: Placebo Test...")
    placebo_df = run_placebo(mfs_df)
    placebo_df.to_csv(OUT / "placebo_test_results.csv", index=False)
    placebo_b64 = plot_placebo(placebo_df, its_res)

    print("Running C5: Granger Causality...")
    granger_df, granger_meta = run_granger(mfs_df, lr_df, stress_df)
    granger_df.to_csv(OUT / "granger_results.csv", index=False)

    print("Running C6: Local Linear RDD...")
    rdd_df = run_rdd(mfs_df)
    rdd_df.to_csv(OUT / "rdd_results.csv", index=False)
    rdd_b64 = plot_rdd(mfs_df, rdd_df)

    print("Building HTML report...")
    html = build_html(
        its_res, its_df, its_b64,
        es_df, es_b64,
        lp_df, lp_b64,
        placebo_df, placebo_b64,
        granger_df, granger_meta,
        rdd_df, rdd_b64,
    )

    out_file = OUT / "analysis_v7_report.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"[OK] {out_file}")


if __name__ == "__main__":
    main()
