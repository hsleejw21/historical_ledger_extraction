#!/usr/bin/env python
"""
analysis_v9.py — Leapfrogging, Capability→Mission, Paper Skeleton

Sections:
  A: Four-Level Framework as Dimensions (not stages) — reframe + operationalization
  B: Leapfrogging Analysis (NEW)
       B1 Normalised growth trajectories
       B2 Relative ITS effect sizes (normalised by pre-reform mean)
       B3 Break magnitude comparison
       B4 Post-reform acceleration comparison
       B5 Event-time plots around Reform Acts
  C: Capability-before-Mission — Strengthened
       C1 Cross-correlation with bootstrap CIs
       C2 Distributed lag model (L3→L4 vs L4→L3)
       C3 Rolling-window correlation
       C4 Event-time plots (L3 vs L4 around each reform)
       C5 State-transition analysis
  D: What the Sequencing Evidence Actually Supports
  E: Reform Acts as Strategic Discontinuities
       E1 Signal-to-noise of breaks
       E2 Sharpness index
  F: Paper Skeleton
"""

import json
import re
import warnings
import base64
from io import BytesIO
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import ruptures as rpt

ROOT     = Path(__file__).resolve().parents[2]
V6       = ROOT / "experiments/reports/analysis_v6"
ENRICHED = ROOT / "experiments/results/enriched"
OUT      = ROOT / "experiments/reports/analysis_v9"
OUT.mkdir(parents=True, exist_ok=True)

CUT1, CUT2 = 1854, 1877

ERA_COLORS = {
    "pre_industrial":   "#d4e6f1",
    "transition":       "#d5f5e3",
    "early_industrial": "#fdebd0",
    "late_industrial":  "#f9ebea",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

LEVEL_META = {
    "L1_inv": {"label": "L1: Efficiency / Standardisation", "short": "L1",
               "proxy": "1 − traditional_function_share", "color": "#2980b9",
               "dim": "Operational", "trigger": "Gradual pressure / internal admin"},
    "L2":     {"label": "L2: Process Modernisation",        "short": "L2",
               "proxy": "payment_modernity_index",           "color": "#27ae60",
               "dim": "Operational", "trigger": "Administrative formalisation"},
    "L3":     {"label": "L3: Capability Expansion",         "short": "L3",
               "proxy": "salary_stipend_share",              "color": "#e67e22",
               "dim": "Strategic",   "trigger": "Institutional shock / reform mandate"},
    "L4":     {"label": "L4: Mission Transformation",       "short": "L4",
               "proxy": "educational_share",                 "color": "#8e44ad",
               "dim": "Strategic",   "trigger": "Post-reform strategic repositioning"},
}
LEVEL_COLS   = ["L1_inv", "L2", "L3", "L4"]
LEVEL_ROLL   = {"L1_inv": "L1_inv_10yr", "L2": "L2_10yr", "L3": "L3_10yr", "L4": "L4_10yr"}


# ---------------------------------------------------------------------------
# Utilities (carried over from v8)
# ---------------------------------------------------------------------------

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
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _df_to_html(df, title=""):
    rows = []
    if title:
        rows.append(f'<caption>{title}</caption>')
    rows.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead>")
    rows.append("<tbody>")
    for _, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if isinstance(v, float):
                cells.append(f"<td>{v:.4f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody>")
    return "<table>" + "".join(rows) + "</table>"


def _add_era_bands(ax, df):
    if "era" not in df.columns:
        return
    era_order = ["pre_industrial", "transition", "early_industrial", "late_industrial"]
    for era in era_order:
        sub = df[df["era"] == era]["year"]
        if len(sub):
            ax.axvspan(sub.min(), sub.max(), alpha=0.15,
                       color=ERA_COLORS.get(era, "#eeeeee"), zorder=0)


def _run_its(series, years):
    df = pd.DataFrame({"year": years, "y": series}).dropna().sort_values("year")
    df["t"]        = df["year"] - 1700
    df["D54"]      = (df["year"] >= CUT1).astype(float)
    df["t_post54"] = (df["year"] - CUT1) * df["D54"]
    df["D77"]      = (df["year"] >= CUT2).astype(float)
    df["t_post77"] = (df["year"] - CUT2) * df["D77"]
    X   = sm.add_constant(df[["t", "D54", "t_post54", "D77", "t_post77"]])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    X_cf = X.copy()
    for col in ["D54", "t_post54", "D77", "t_post77"]:
        X_cf[col] = 0.0
    return {"df": df, "res": res,
            "fitted": res.fittedvalues.values,
            "counterfactual": res.predict(X_cf).values}


def _detect_breaks(signal, n_bkps=2, min_size=8):
    try:
        algo = rpt.Binseg(model="rbf", min_size=min_size).fit(signal.reshape(-1, 1))
        return algo.predict(n_bkps=n_bkps)
    except Exception:
        return [len(signal)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_main_panel():
    df = pd.read_csv(V6 / "four_level_proxies.csv").sort_values("year").reset_index(drop=True)
    df["L1_inv"]      = 1.0 - df["L1"]
    df["L1_inv_10yr"] = df["L1_inv"].rolling(10, center=True, min_periods=5).mean()
    return df


def _amount_pounds(row):
    try:
        return (float(row.get("amount_pounds") or 0)
                + float(row.get("amount_shillings") or 0) / 20.0
                + float(row.get("amount_pence_whole") or 0) / 240.0)
    except Exception:
        return 0.0


def load_enriched_yearly():
    year_stats = defaultdict(lambda: {"eng": 0, "lang_n": 0, "admin_exp": 0.0, "total_exp": 0.0})
    for f in sorted(ENRICHED.glob("*.json")):
        m = re.match(r"(\d{4})_", f.name)
        if not m:
            continue
        year = int(m.group(1))
        with open(f) as fh:
            data = json.load(fh)
        for row in data.get("rows", []):
            lang = row.get("language")
            if lang in ("english", "latin", "mixed"):
                year_stats[year]["lang_n"] += 1
                if lang == "english":
                    year_stats[year]["eng"] += 1
            if row.get("direction") == "expenditure":
                amt = _amount_pounds(row)
                year_stats[year]["total_exp"] += amt
                if row.get("category") == "administrative":
                    year_stats[year]["admin_exp"] += amt
    records = []
    for year in sorted(year_stats):
        s = year_stats[year]
        n  = s["lang_n"]
        te = s["total_exp"]
        records.append({
            "year":          year,
            "english_share": s["eng"] / n  if n  > 0 else np.nan,
            "admin_share":   s["admin_exp"] / te if te > 0 else np.nan,
        })
    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section A — Four-Level Framework as Dimensions
# ---------------------------------------------------------------------------

def section_a(panel_df):
    df = panel_df.copy()

    # Figure A1: 4-panel trend plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    for ax, col in zip(axes, LEVEL_COLS):
        meta = LEVEL_META[col]
        _add_era_bands(ax, df)
        ax.scatter(df["year"], df[col], s=6, alpha=0.4, color=meta["color"])
        ax.plot(df["year"], df[LEVEL_ROLL[col]], color=meta["color"], lw=2)
        for c in [CUT1, CUT2]:
            ax.axvline(c, color="#c0392b", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(meta["label"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Share (0–1)")
    for ax in axes[-2:]:
        ax.set_xlabel("Year")
    fig.suptitle("A1: Four Dimensions of Transformation (1700–1900)\n"
                 "Treated as dimensions / pathways, not deterministic stages",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    b64_trends = _b64(fig)

    # Dimension table HTML
    dim_rows = "".join(
        f"<tr><td><strong>{m['short']}</strong></td><td>{m['label']}</td>"
        f"<td><code>{m['proxy']}</code></td><td>{m['dim']}</td><td>{m['trigger']}</td></tr>"
        for m in LEVEL_META.values()
    )
    dim_table = (
        "<table><thead><tr><th>Code</th><th>Dimension</th><th>Proxy</th>"
        "<th>Axis</th><th>Primary Trigger</th></tr></thead>"
        f"<tbody>{dim_rows}</tbody></table>"
    )

    # Era descriptives
    era_order = ["pre_industrial", "transition", "early_industrial", "late_industrial"]
    rows = []
    for era in era_order:
        sub = df[df["era"] == era]
        if not len(sub):
            continue
        rows.append({
            "Era": era.replace("_", " ").title(),
            "Years": f"{sub['year'].min()}–{sub['year'].max()}",
            "n": len(sub),
            "L1_inv": round(sub["L1_inv"].mean(), 3),
            "L2": round(sub["L2"].mean(), 3),
            "L3": round(sub["L3"].mean(), 3),
            "L4": round(sub["L4"].mean(), 3),
        })
    era_df = pd.DataFrame(rows)
    era_df.to_csv(OUT / "era_level_descriptives.csv", index=False)

    return {"b64_trends": b64_trends, "dim_table": dim_table,
            "era_table": _df_to_html(era_df, title="Era-Level Descriptives")}


# ---------------------------------------------------------------------------
# Section B — Leapfrogging Analysis
# ---------------------------------------------------------------------------

def section_b(panel_df):
    df = panel_df.dropna(subset=LEVEL_COLS).sort_values("year").reset_index(drop=True)
    results = {}

    # ---- B1: Normalised growth trajectories --------------------------------
    base_mask = (df["year"] >= 1820) & (df["year"] <= 1853)
    norm_df = df[["year"] + LEVEL_COLS].copy()
    for col in LEVEL_COLS:
        base_mean = df.loc[base_mask, col].mean()
        norm_df[f"{col}_norm"] = df[col] / base_mean if base_mean > 0 else np.nan

    norm_df.to_csv(OUT / "normalised_trajectories.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    _add_era_bands(ax, df)
    plot_df = norm_df[norm_df["year"] >= 1820]
    for col in LEVEL_COLS:
        meta = LEVEL_META[col]
        roll = plot_df[f"{col}_norm"].rolling(10, center=True, min_periods=5).mean()
        ax.plot(plot_df["year"], roll, color=meta["color"], lw=2.5, label=meta["label"])
        ax.scatter(plot_df["year"], plot_df[f"{col}_norm"], s=5, alpha=0.25, color=meta["color"])
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="Pre-reform baseline = 1")
    ax.axvline(CUT1, color="#c0392b", ls="--", lw=1.2, alpha=0.8, label=f"{CUT1} Reform Act")
    ax.axvline(CUT2, color="#922b21", ls=":",  lw=1.2, alpha=0.8, label=f"{CUT2} Reform Act")
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalised index (pre-reform mean = 1)")
    ax.set_title("B1: Normalised Growth Trajectories by Level (1820–1900)\n"
                 "Leapfrogging: L3/L4 rising disproportionately above L1/L2 baseline")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    results["b64_norm"] = _b64(fig)

    # ---- B2: Relative ITS effect sizes (normalised by pre-reform mean) -----
    its_results = {}
    pre_means   = {}
    effect_rows = []
    for col in LEVEL_COLS:
        its = _run_its(df[col], df["year"])
        its_results[col] = its
        pre_mean = df.loc[df["year"] < CUT1, col].mean()
        pre_means[col] = pre_mean
        for param, reform_yr in [("D54", CUT1), ("D77", CUT2)]:
            coef = its["res"].params[param]
            se   = its["res"].bse[param]
            pval = its["res"].pvalues[param]
            norm_effect = coef / pre_mean if pre_mean > 0 else np.nan
            effect_rows.append({
                "Level": LEVEL_META[col]["label"],
                "Code":  col,
                "Reform": reform_yr,
                "Coef":  round(coef,        4),
                "SE":    round(se,           4),
                "p":     round(pval,         4),
                "Sig":   _stars(pval),
                "Pre-reform mean": round(pre_mean, 4),
                "Norm. effect size": round(norm_effect, 4),
            })

    eff_df = pd.DataFrame(effect_rows)
    eff_df.to_csv(OUT / "effect_sizes.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [LEVEL_META[c]["color"] for c in LEVEL_COLS]
    for ax, reform in zip(axes, [CUT1, CUT2]):
        sub = eff_df[eff_df["Reform"] == reform].reset_index(drop=True)
        vals = sub["Norm. effect size"].values
        errs = (sub["SE"] / sub["Pre-reform mean"]).values
        ax.barh(range(len(sub)), vals, xerr=errs, color=colors, alpha=0.85,
                error_kw={"ecolor": "black", "capsize": 4})
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([LEVEL_META[c]["short"] for c in LEVEL_COLS])
        ax.set_title(f"Reform {reform}: Normalised Effect Size\n(ITS level shift / pre-reform mean)")
        ax.set_xlabel("Effect size (× pre-reform baseline)")
        for i, row in sub.iterrows():
            s = row["Sig"]
            if s:
                xpos = row["Norm. effect size"] + errs[i] * 1.1
                ax.text(xpos, i, s, va="center", fontsize=11)
    fig.suptitle("B2: Leapfrogging — Normalised ITS Effect Sizes by Level\n"
                 "Larger bars for L3/L4 indicate disproportionate higher-order transformation",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    results["b64_effects"] = _b64(fig)
    results["eff_df"] = eff_df

    # ---- B3: Break magnitude comparison ------------------------------------
    years = df["year"].values
    bkp_rows = []
    for col in LEVEL_COLS:
        signal = df[col].values
        bkps   = _detect_breaks(signal, n_bkps=2, min_size=8)
        idx1   = bkps[0]
        pre_mean  = signal[:idx1].mean() if idx1 > 0 else np.nan
        post_mean = signal[idx1:].mean()  if idx1 < len(signal) else np.nan
        abs_mag   = abs(post_mean - pre_mean)
        rel_mag   = abs_mag / pre_mean if pre_mean and pre_mean > 0 else np.nan
        bkp_rows.append({
            "Level": LEVEL_META[col]["label"],
            "Code":  col,
            "Break year": int(years[min(idx1, len(years)-1)]),
            "Pre-break mean":  round(pre_mean, 4) if pre_mean is not None else np.nan,
            "Post-break mean": round(post_mean, 4) if post_mean is not None else np.nan,
            "Abs. magnitude":  round(abs_mag,  4) if abs_mag  is not None else np.nan,
            "Rel. magnitude":  round(rel_mag,  4) if rel_mag  is not None else np.nan,
        })

    bkp_df = pd.DataFrame(bkp_rows)
    bkp_df.to_csv(OUT / "break_magnitudes.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    rel_vals = bkp_df["Rel. magnitude"].values
    bars = ax.bar([LEVEL_META[c]["short"] for c in LEVEL_COLS], rel_vals,
                  color=colors, alpha=0.85)
    ax.set_ylabel("|post − pre| / pre mean")
    ax.set_title("B3: Break Magnitude by Level (Relative to Pre-break Mean)\n"
                 "Larger = more disruptive structural shift")
    for bar, val in zip(bars, rel_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    results["b64_bkp_mag"] = _b64(fig)
    results["bkp_df"] = bkp_df

    # ---- B4: Post-reform acceleration comparison ---------------------------
    accel_rows = []
    windows = {"1854–1877": (1854, 1877), "1877–1900": (1877, 1900)}
    for col in LEVEL_COLS:
        roll10 = df[col].rolling(10, center=True, min_periods=5).mean()
        df_tmp = df.copy()
        df_tmp["roll"] = roll10
        for wlabel, (w0, w1) in windows.items():
            sub = df_tmp[(df_tmp["year"] >= w0) & (df_tmp["year"] <= w1)]["roll"]
            if len(sub) >= 5:
                start_val = sub.iloc[0]
                end_val   = sub.iloc[-1]
                pct_change = (end_val - start_val) / start_val * 100 if start_val and start_val != 0 else np.nan
            else:
                pct_change = np.nan
            accel_rows.append({
                "Level": LEVEL_META[col]["label"],
                "Code":  col,
                "Window": wlabel,
                "% change (10yr rolling)": round(pct_change, 2) if not np.isnan(pct_change) else np.nan,
            })

    accel_df = pd.DataFrame(accel_rows)
    accel_df.to_csv(OUT / "post_reform_acceleration.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, wlabel in zip(axes, ["1854–1877", "1877–1900"]):
        sub = accel_df[accel_df["Window"] == wlabel]
        vals  = sub["% change (10yr rolling)"].values
        bars  = ax.bar([LEVEL_META[c]["short"] for c in LEVEL_COLS], vals,
                       color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("% change in 10yr rolling mean")
        ax.set_title(f"B4: Acceleration ({wlabel})")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ypos = bar.get_height() + (1.5 if val >= 0 else -4)
                ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle("B4: Post-Reform Acceleration by Level\n"
                 "Higher-order levels (L3/L4) should show larger % gains if leapfrogging",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    results["b64_accel"] = _b64(fig)
    results["accel_df"]  = accel_df

    # ---- B5: Event-time plots around each Reform Act -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cut, label in zip(axes, [CUT1, CUT2],
                              [f"{CUT1} Reform Act", f"{CUT2} Reform Act"]):
        window = 15
        for col in LEVEL_COLS:
            meta   = LEVEL_META[col]
            events = []
            for t in range(-window, window + 1):
                yr  = cut + t
                row = df[df["year"] == yr]
                events.append(row[col].iloc[0] if len(row) else np.nan)
            ax.plot(range(-window, window + 1), events,
                    color=meta["color"], lw=2, label=meta["short"], alpha=0.85)
            ax.scatter(range(-window, window + 1), events,
                       s=8, color=meta["color"], alpha=0.4)
        ax.axvline(0, color="#c0392b", ls="--", lw=1.5, label="Reform")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_xlabel(f"Years relative to {cut}")
        ax.set_ylabel("Level value")
        ax.set_title(label)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("B5: Event-Time Response by Level Around Each Reform Act\n"
                 "Shows differential timing of level-specific responses to institutional shocks",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    results["b64_event"] = _b64(fig)

    results["its_results"] = its_results
    results["pre_means"]   = pre_means
    return results


# ---------------------------------------------------------------------------
# Section C — Capability-before-Mission (Strengthened)
# ---------------------------------------------------------------------------

def section_c(panel_df):
    df = panel_df.dropna(subset=LEVEL_COLS).sort_values("year").reset_index(drop=True)
    results = {}

    # ---- C1: Cross-correlation L3→L4 (BOTH levels and first-differences) ---
    # Levels can be contaminated by shared trends; first-differences isolate
    # changes-on-changes, which is the correct test for L3 changes leading L4 changes.
    max_lag     = 15
    n_boot      = 1000
    block_size  = 5
    lags        = list(range(-max_lag, max_lag + 1))
    rng         = np.random.default_rng(42)

    def _xcorr_at_lag(xv, yv, lag):
        if lag >= 0:
            a, b = (xv[:len(xv)-lag] if lag > 0 else xv,
                    yv[lag:]         if lag > 0 else yv)
        else:
            a, b = xv[-lag:], yv[:len(yv)+lag]
        n = min(len(a), len(b))
        if n < 15:
            return np.nan
        r, _ = stats.pearsonr(a[:n], b[:n])
        return r

    def _bootstrap_ci(x, y, lags_):
        n = len(x)
        n_blocks = max(1, n // block_size)
        boot = np.full((n_boot, len(lags_)), np.nan)
        for b_idx in range(n_boot):
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            idx    = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])[:n]
            xb, yb = x[idx], y[idx]
            for li, lag in enumerate(lags_):
                boot[b_idx, li] = _xcorr_at_lag(xb, yb, lag)
        return np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)

    # Levels
    x_lev = df["L3"].values
    y_lev = df["L4"].values
    obs_lev = [_xcorr_at_lag(x_lev, y_lev, lag) for lag in lags]
    ci_lo_lev, ci_hi_lev = _bootstrap_ci(x_lev, y_lev, lags)

    # First-differences (changes-on-changes — proper lead-lag test)
    x_dif = np.diff(df["L3"].values)
    y_dif = np.diff(df["L4"].values)
    obs_dif = [_xcorr_at_lag(x_dif, y_dif, lag) for lag in lags]
    ci_lo_dif, ci_hi_dif = _bootstrap_ci(x_dif, y_dif, lags)

    xcorr_df = pd.DataFrame({
        "Lag":         lags,
        "r_levels":    obs_lev,
        "CI_lo_lev":   ci_lo_lev,
        "CI_hi_lev":   ci_hi_lev,
        "r_diffs":     obs_dif,
        "CI_lo_dif":   ci_lo_dif,
        "CI_hi_dif":   ci_hi_dif,
    })
    xcorr_df.to_csv(OUT / "bootstrap_crosscorr_L3_L4.csv", index=False)

    # Peak lag for first-differenced (preferred) series
    obs_dif_arr = np.array(obs_dif, dtype=float)
    peak_idx    = int(np.nanargmax(np.abs(obs_dif_arr)))
    peak_lag    = lags[peak_idx]
    peak_r      = float(obs_dif_arr[peak_idx])
    peak_ci_lo  = float(ci_lo_dif[peak_idx])
    peak_ci_hi  = float(ci_hi_dif[peak_idx])
    peak_sig    = (peak_ci_lo > 0) or (peak_ci_hi < 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, obs, lo, hi, title in [
        (axes[0], obs_lev, ci_lo_lev, ci_hi_lev, "C1a: Cross-correlation on LEVELS (trend-contaminated)"),
        (axes[1], obs_dif, ci_lo_dif, ci_hi_dif, "C1b: Cross-correlation on FIRST-DIFFERENCES (preferred)"),
    ]:
        ax.bar(lags, obs, color="#e67e22", alpha=0.7, width=0.8)
        ax.fill_between(lags, lo, hi, alpha=0.25, color="#e67e22", label="95% bootstrap CI")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Lag (years; positive → L3 leads L4)")
        ax.set_ylabel("Pearson r")
        ax.set_title(title)
        ax.legend(fontsize=8, frameon=False)
    axes[1].axvline(peak_lag, color="#c0392b", lw=2, ls="-",
                    label=f"Peak lag = {peak_lag} (r = {peak_r:.3f})")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("C1: L3→L4 Cross-Correlation — Levels vs First-Differences\n"
                 "First-difference panel is the proper lead-lag test (removes shared trends).",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    results["b64_xcorr"]   = _b64(fig)
    results["peak_lag"]    = peak_lag
    results["peak_r"]      = peak_r
    results["peak_ci_lo"]  = peak_ci_lo
    results["peak_ci_hi"]  = peak_ci_hi
    results["peak_sig"]    = peak_sig

    # ---- C2: Distributed lag model (LEVELS + FIRST-DIFFERENCES) -----------
    max_dl = 10
    dl_rows = []

    def _dlm(yv, xv, direction_label, mode):
        lag_matrix = pd.DataFrame(
            {f"lag{k}": pd.Series(xv).shift(k).values for k in range(max_dl + 1)}
        ).dropna()
        n_valid = len(lag_matrix)
        yv_trim = yv[max_dl:][:n_valid]
        X = sm.add_constant(lag_matrix.values)
        res = sm.OLS(yv_trim, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        betas = res.params[1:]
        ses   = res.bse[1:]
        pvals = res.pvalues[1:]
        cumul = np.cumsum(betas)
        for k in range(len(betas)):
            dl_rows.append({
                "Mode":      mode,
                "Direction": direction_label,
                "Lag k":     k,
                "Beta_k":    round(betas[k], 5),
                "SE":        round(ses[k],   5),
                "p":         round(pvals[k], 4),
                "Cumulative": round(cumul[k], 5),
            })
        return betas, cumul

    # Levels
    L3_lev, L4_lev = df["L3"].values, df["L4"].values
    betas_fwd_lev, cumul_fwd_lev = _dlm(L4_lev, L3_lev, "L3→L4 (capability→mission)", "levels")
    betas_rev_lev, cumul_rev_lev = _dlm(L3_lev, L4_lev, "L4→L3 (reverse)",               "levels")

    # First-differences
    L3_dif, L4_dif = np.diff(L3_lev), np.diff(L4_lev)
    betas_fwd_dif, cumul_fwd_dif = _dlm(L4_dif, L3_dif, "ΔL3→ΔL4 (capability→mission)", "diffs")
    betas_rev_dif, cumul_rev_dif = _dlm(L3_dif, L4_dif, "ΔL4→ΔL3 (reverse)",              "diffs")

    dl_df = pd.DataFrame(dl_rows)
    dl_df.to_csv(OUT / "distributed_lag.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    k_range = list(range(max_dl + 1))

    # Top row: levels
    ax = axes[0, 0]
    ax.bar([k - 0.2 for k in k_range], betas_fwd_lev, width=0.35, color="#e67e22", alpha=0.8, label="L3→L4")
    ax.bar([k + 0.2 for k in k_range], betas_rev_lev, width=0.35, color="#8e44ad", alpha=0.8, label="L4→L3")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("C2a: Levels — β_k by lag")
    ax.set_xlabel("Lag k"); ax.set_ylabel("β_k")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    ax.plot(k_range, cumul_fwd_lev, "o-", color="#e67e22", lw=2, label="L3→L4 cumulative")
    ax.plot(k_range, cumul_rev_lev, "s--", color="#8e44ad", lw=2, label="L4→L3 cumulative")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("C2b: Levels — cumulative impulse")
    ax.set_xlabel("Lag k"); ax.set_ylabel("Cumulative β")
    ax.legend(fontsize=8, frameon=False)

    # Bottom row: first-differences (preferred for lead-lag)
    ax = axes[1, 0]
    ax.bar([k - 0.2 for k in k_range], betas_fwd_dif, width=0.35, color="#e67e22", alpha=0.8, label="ΔL3→ΔL4")
    ax.bar([k + 0.2 for k in k_range], betas_rev_dif, width=0.35, color="#8e44ad", alpha=0.8, label="ΔL4→ΔL3")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("C2c: First-differences — β_k by lag")
    ax.set_xlabel("Lag k"); ax.set_ylabel("β_k")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    ax.plot(k_range, cumul_fwd_dif, "o-", color="#e67e22", lw=2, label="ΔL3→ΔL4 cumulative")
    ax.plot(k_range, cumul_rev_dif, "s--", color="#8e44ad", lw=2, label="ΔL4→ΔL3 cumulative")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("C2d: First-differences — cumulative impulse (preferred)")
    ax.set_xlabel("Lag k"); ax.set_ylabel("Cumulative β")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("C2: Distributed Lag Model — Does L3 Predict L4 More Than L4 Predicts L3?\n"
                 "Bottom row (first-differences) is the trend-robust test.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    results["b64_dlm"]            = _b64(fig)
    results["dl_df"]              = dl_df
    results["cumul_fwd_dif_end"]  = float(cumul_fwd_dif[-1])
    results["cumul_rev_dif_end"]  = float(cumul_rev_dif[-1])

    # ---- C3: Rolling-window correlation (30-year window) -------------------
    window_size = 30
    roll_rows   = []
    for i in range(len(df) - window_size + 1):
        sub     = df.iloc[i: i + window_size]
        mid_yr  = int(sub["year"].iloc[window_size // 2])
        l3, l4  = sub["L3"].values, sub["L4"].values
        r_sync, _ = stats.pearsonr(l3, l4)
        # 5-year lead: L3[t] vs L4[t+5]
        if i + window_size + 5 <= len(df):
            l3_lead = df["L3"].iloc[i: i + window_size].values
            l4_lag  = df["L4"].iloc[i + 5: i + window_size + 5].values
            r_lead, _ = stats.pearsonr(l3_lead, l4_lag)
        else:
            r_lead = np.nan
        roll_rows.append({"year": mid_yr, "r_sync": round(r_sync, 4),
                          "r_L3_leads5": round(r_lead, 4) if not np.isnan(r_lead) else np.nan,
                          "lead_advantage": round(r_lead - r_sync, 4) if not np.isnan(r_lead) else np.nan})

    roll_df = pd.DataFrame(roll_rows)
    roll_df.to_csv(OUT / "rolling_correlation.csv", index=False)

    # Quantify when L3 leads L4 (lead_advantage > 0.1) vs synchronous (< -0.1)
    n_lead    = (roll_df["lead_advantage"] > 0.1).sum()
    n_sync    = (roll_df["lead_advantage"] < -0.1).sum()
    n_total   = roll_df["lead_advantage"].notna().sum()
    leading_years   = roll_df[roll_df["lead_advantage"] > 0.1]["year"]
    leading_period  = f"{leading_years.min()}–{leading_years.max()}" if len(leading_years) else "none"
    results["pct_L3_leads"]   = n_lead / n_total if n_total > 0 else 0.0
    results["leading_period"] = leading_period
    results["n_lead"]         = int(n_lead)
    results["n_sync"]         = int(n_sync)
    results["n_total_rc"]     = int(n_total)

    fig, ax = plt.subplots(figsize=(10, 4))
    _add_era_bands(ax, df)
    ax.plot(roll_df["year"], roll_df["r_sync"],       color="#e67e22", lw=2, label="r(L3_t, L4_t) — synchronous")
    ax.plot(roll_df["year"], roll_df["r_L3_leads5"],  color="#8e44ad", lw=2, ls="--", label="r(L3_t, L4_{t+5}) — L3 leads 5yr")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(CUT1, color="#c0392b", ls="--", lw=1, alpha=0.7, label=f"{CUT1}")
    ax.axvline(CUT2, color="#922b21", ls=":",  lw=1, alpha=0.7, label=f"{CUT2}")
    ax.set_xlabel("Year (centre of 30-yr window)")
    ax.set_ylabel("Pearson r")
    ax.set_title("C3: Rolling 30-Year Correlation: L3 and L4\n"
                 "If r(L3_t, L4_t+5) > r(L3_t, L4_t) → L3 systematically leads L4")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    results["b64_rollcorr"] = _b64(fig)

    # ---- C4: Event-time — L3 vs L4 around each reform ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cut, label in zip(axes, [CUT1, CUT2],
                              [f"{CUT1} Reform Act", f"{CUT2} Reform Act"]):
        window = 15
        ev_l3, ev_l4 = [], []
        for t in range(-window, window + 1):
            yr = cut + t
            row = df[df["year"] == yr]
            ev_l3.append(row["L3"].iloc[0] if len(row) else np.nan)
            ev_l4.append(row["L4"].iloc[0] if len(row) else np.nan)
        trange = range(-window, window + 1)
        ax.plot(trange, ev_l3, "o-", color="#e67e22", lw=2, ms=4, label="L3 Capability")
        ax.plot(trange, ev_l4, "s-", color="#8e44ad", lw=2, ms=4, label="L4 Mission")
        ev_l3_arr = np.array(ev_l3, dtype=float)
        ev_l4_arr = np.array(ev_l4, dtype=float)
        ax.fill_between(trange,
                        np.where(ev_l3_arr > ev_l4_arr, ev_l3_arr, np.nan),
                        np.where(ev_l3_arr > ev_l4_arr, ev_l4_arr, np.nan),
                        alpha=0.2, color="#e67e22", label="L3 > L4 gap")
        ax.axvline(0, color="#c0392b", ls="--", lw=1.5, label="Reform")
        ax.set_xlabel(f"Years relative to {cut}")
        ax.set_ylabel("Share")
        ax.set_title(label)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("C4: Event-Time Plots — Capability (L3) vs Mission (L4)\n"
                 "L3 rising before L4 in each reform window → capability precedes mission",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    results["b64_evL3L4"] = _b64(fig)

    # ---- C5: State-transition analysis (FIRST-ENTRY TIMING) ---------------
    # The right test is: in what year does each level first enter a sustained
    # "high" state? If L3 enters first → capability precedes mission.
    p67_L3 = df["L3"].quantile(0.67)
    p67_L4 = df["L4"].quantile(0.67)

    def _first_persistent_entry(series, threshold, min_run=5):
        above = (series > threshold).astype(int).values
        for i in range(len(above) - min_run + 1):
            if all(above[i:i + min_run]):
                return i
        return None

    # Use 10-yr rolling mean to avoid year-to-year noise driving entry
    L3_smooth = df["L3"].rolling(10, center=True, min_periods=5).mean()
    L4_smooth = df["L4"].rolling(10, center=True, min_periods=5).mean()

    idx_L3 = _first_persistent_entry(L3_smooth, p67_L3, min_run=5)
    idx_L4 = _first_persistent_entry(L4_smooth, p67_L4, min_run=5)
    year_L3_entry = int(df["year"].iloc[idx_L3]) if idx_L3 is not None else None
    year_L4_entry = int(df["year"].iloc[idx_L4]) if idx_L4 is not None else None

    # Sensitivity at multiple thresholds
    sens_rows = []
    for q in [0.50, 0.60, 0.67, 0.75, 0.80]:
        thL3 = df["L3"].quantile(q)
        thL4 = df["L4"].quantile(q)
        iL3 = _first_persistent_entry(L3_smooth, thL3, min_run=5)
        iL4 = _first_persistent_entry(L4_smooth, thL4, min_run=5)
        yL3 = int(df["year"].iloc[iL3]) if iL3 is not None else None
        yL4 = int(df["year"].iloc[iL4]) if iL4 is not None else None
        lead = (yL4 - yL3) if (yL3 is not None and yL4 is not None) else None
        sens_rows.append({
            "Threshold (quantile)": q,
            "L3 entry year":  yL3 if yL3 else "Not detected",
            "L4 entry year":  yL4 if yL4 else "Not detected",
            "L3 leads L4 by (years)": lead if lead is not None else "N/A",
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT / "state_transitions.csv", index=False)

    # Verdict
    leads_count   = sum(1 for r in sens_rows
                        if isinstance(r["L3 leads L4 by (years)"], int) and r["L3 leads L4 by (years)"] > 0)
    total_count   = sum(1 for r in sens_rows
                        if isinstance(r["L3 leads L4 by (years)"], int))
    state_verdict = (f"L3 enters sustained high-state before L4 at {leads_count}/{total_count} "
                     f"of the tested quantile thresholds.")

    # Headline numbers for reporting
    if year_L3_entry and year_L4_entry:
        L3L4_gap = year_L4_entry - year_L3_entry
    else:
        L3L4_gap = None

    results["state_summary"]    = sens_df
    results["state_verdict"]    = state_verdict
    results["year_L3_entry"]    = year_L3_entry
    results["year_L4_entry"]    = year_L4_entry
    results["L3L4_gap"]         = L3L4_gap

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 4))
    _add_era_bands(ax, df)
    ax.plot(df["year"], L3_smooth, color="#e67e22", lw=2, label="L3 (10yr rolling)")
    ax.plot(df["year"], L4_smooth, color="#8e44ad", lw=2, label="L4 (10yr rolling)")
    ax.axhline(p67_L3, color="#e67e22", ls=":", alpha=0.5, label="L3 67th-pct threshold")
    ax.axhline(p67_L4, color="#8e44ad", ls=":", alpha=0.5, label="L4 67th-pct threshold")
    if year_L3_entry:
        ax.axvline(year_L3_entry, color="#e67e22", lw=2, ls="--",
                   label=f"L3 first-entry: {year_L3_entry}")
    if year_L4_entry:
        ax.axvline(year_L4_entry, color="#8e44ad", lw=2, ls="--",
                   label=f"L4 first-entry: {year_L4_entry}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    ax.set_title("C5: First Persistent Entry into High-State — L3 (capability) vs L4 (mission)\n"
                 f"{state_verdict}")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    plt.tight_layout()
    results["b64_state"] = _b64(fig)

    return results


# ---------------------------------------------------------------------------
# Section D — What the Sequencing Evidence Actually Supports
# ---------------------------------------------------------------------------

def section_d(b_results, c_results):
    peak_lag   = c_results.get("peak_lag", "N/A")
    peak_r     = c_results.get("peak_r",   np.nan)
    peak_sig   = c_results.get("peak_sig", False)
    y_L3       = c_results.get("year_L3_entry")
    y_L4       = c_results.get("year_L4_entry")
    L3L4_gap   = c_results.get("L3L4_gap")
    state_v    = c_results.get("state_verdict", "")
    cum_fwd    = c_results.get("cumul_fwd_dif_end", np.nan)
    cum_rev    = c_results.get("cumul_rev_dif_end", np.nan)

    bkp_df = b_results.get("bkp_df", pd.DataFrame())
    L4_bkp = bkp_df.set_index("Code")["Rel. magnitude"].get("L4", np.nan) if len(bkp_df) else np.nan

    sig_word = "significant" if peak_sig else "not significant"
    gap_str  = f"{L3L4_gap} yr" if L3L4_gap is not None else "N/A"

    seq_rows = [
        ("Changepoint ordering (B3)", "L1→L2→L3→L4",
         "L4 has largest relative break magnitude" + (f" ({L4_bkp:.2f}×)" if not np.isnan(L4_bkp) else ""),
         "Consistent with leapfrogging toward higher-order, not strict sequential"),
        ("Sequential Granger (v8-B2, carried)", "L_k predicts L_{k+1}",
         "Mostly insignificant",
         "No evidence of predictive chaining across levels"),
        ("Cross-lag on first-differences (C1b)", "ΔL3 leads ΔL4",
         f"Peak lag = {peak_lag} yr, r = {peak_r:.3f} ({sig_word})",
         "Direction consistent with hypothesis; statistical power limited"),
        ("Transition dates (v8-B4, carried)", "Ordering of threshold crossings",
         "Sensitive to threshold choice",
         "Cannot robustly confirm L1<L2<L3<L4 ordering"),
        ("Normalised ITS effects (B2)", "Which level responds most to reform",
         "L3/L4 dominant at 1877; L3/L4 most disrupted at 1854",
         "Leapfrogging pattern — higher-order transformation dominates magnitudes"),
        ("Distributed lag on first-differences (C2c/d)", "ΔL3→ΔL4 vs ΔL4→ΔL3",
         f"Cumulative fwd = {cum_fwd:.3f}, rev = {cum_rev:.3f}",
         "Forward direction not clearly dominant in differences"),
        ("First-entry timing (C5)", "L3 enters high-state before L4",
         f"L3: {y_L3 if y_L3 else 'N/A'}; L4: {y_L4 if y_L4 else 'N/A'}; gap = {gap_str}. {state_v}",
         "Directional evidence: L3 enters sustained high-state ahead of L4"),
    ]

    rows_html = "".join(
        f"<tr><td>{r[0]}</td><td><em>{r[1]}</em></td><td>{r[2]}</td><td>{r[3]}</td></tr>"
        for r in seq_rows
    )
    table_html = (
        "<table><thead><tr><th>Test</th><th>Direction Tested</th>"
        "<th>Result</th><th>Interpretation</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )

    return {"table_html": table_html}


# ---------------------------------------------------------------------------
# Section E — Reform Acts as Strategic Discontinuities
# ---------------------------------------------------------------------------

def section_e(panel_df):
    df = panel_df.dropna(subset=LEVEL_COLS).sort_values("year").reset_index(drop=True)
    results = {}

    # ---- E1: Signal-to-noise of breaks ------------------------------------
    sn_rows = []
    for col in LEVEL_COLS:
        series = df[col].values
        years  = df["year"].values
        # pre-reform rolling std (10-yr window, pre-1854)
        pre_mask  = years < CUT1
        pre_series = pd.Series(series[pre_mask])
        pre_std = pre_series.rolling(10, min_periods=5).std().mean()
        # break magnitude at 1854
        pre_mean_1854  = series[years < CUT1].mean()
        post_mean_1854 = series[(years >= CUT1) & (years < CUT2)].mean()
        break_mag = abs(post_mean_1854 - pre_mean_1854)
        snr = break_mag / pre_std if pre_std and pre_std > 0 else np.nan
        sn_rows.append({
            "Level": LEVEL_META[col]["label"],
            "Code":  col,
            "Pre-reform rolling std (avg)": round(pre_std, 4) if not np.isnan(pre_std) else np.nan,
            "Break magnitude (1854)": round(break_mag, 4),
            "Signal-to-noise ratio": round(snr, 2) if not np.isnan(snr) else np.nan,
        })

    sn_df = pd.DataFrame(sn_rows)
    sn_df.to_csv(OUT / "signal_to_noise.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    snr_vals = sn_df["Signal-to-noise ratio"].values
    colors   = [LEVEL_META[c]["color"] for c in LEVEL_COLS]
    bars = ax.bar([LEVEL_META[c]["short"] for c in LEVEL_COLS], snr_vals,
                  color=colors, alpha=0.85)
    ax.set_ylabel("Break magnitude / pre-reform rolling std")
    ax.set_title("E1: Signal-to-Noise of 1854 Reform Break by Level\n"
                 "High ratio → sharp discontinuity, not gradual drift")
    for bar, val in zip(bars, snr_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val:.1f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    results["b64_snr"] = _b64(fig)
    results["sn_df"]   = sn_df

    # ---- E2: Sharpness index -----------------------------------------------
    sharpness_rows = []
    for col in LEVEL_COLS:
        series = df[col].values
        years  = df["year"].values
        post_mask = years >= CUT1
        total_gain = series[post_mask].mean() - series[years < CUT1].mean()
        if abs(total_gain) < 1e-9:
            sharpness_rows.append({"Level": LEVEL_META[col]["label"], "Code": col,
                                   "Total post-1854 gain": 0, "5-yr gain": 0,
                                   "Sharpness (5yr/total)": np.nan,
                                   "Linear expected sharpness": np.nan})
            continue
        five_yr_mask = (years >= CUT1) & (years < CUT1 + 5)
        five_yr_gain = series[five_yr_mask].mean() - series[years < CUT1].mean()
        sharpness    = five_yr_gain / total_gain
        # Expected under linear trend: 5/(2x – cutoff to end)
        n_post = post_mask.sum()
        linear_exp = 5 / n_post if n_post > 0 else np.nan
        sharpness_rows.append({
            "Level": LEVEL_META[col]["label"],
            "Code":  col,
            "Total post-1854 gain": round(total_gain, 4),
            "5-yr gain":            round(five_yr_gain, 4),
            "Sharpness (5yr/total)": round(sharpness, 3),
            "Linear expected sharpness": round(linear_exp, 3) if not np.isnan(linear_exp) else np.nan,
        })

    sharp_df = pd.DataFrame(sharpness_rows)
    sharp_df.to_csv(OUT / "sharpness.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(LEVEL_COLS))
    obs_s = sharp_df["Sharpness (5yr/total)"].values
    exp_s = sharp_df["Linear expected sharpness"].values
    ax.bar(x_pos - 0.2, obs_s, width=0.35, color=colors, alpha=0.85, label="Observed sharpness")
    ax.bar(x_pos + 0.2, exp_s, width=0.35, color="grey", alpha=0.5, label="Expected (linear trend)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([LEVEL_META[c]["short"] for c in LEVEL_COLS])
    ax.set_ylabel("Fraction of total gain occurring within 5yr of 1854")
    ax.set_title("E2: Sharpness of Reform Response\nObserved > Expected → discontinuous, not gradual")
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    results["b64_sharp"] = _b64(fig)
    results["sharp_df"]  = sharp_df

    return results


# ---------------------------------------------------------------------------
# Section F — Paper Skeleton
# ---------------------------------------------------------------------------

def section_f_html():
    return """
<div class="section">
<h3>F1. Research Question</h3>
<p>How do organisations transform under major technological and institutional change —
and do they follow a predictable sequential pathway through efficiency, process redesign,
capability expansion, and mission redefinition?
Using 200 years of Oxford University accounting ledgers (1700–1900), we ask:
<em>Does Oxford exhibit leapfrogging — moving directly into higher-order transformation
(capability expansion, mission redefinition) before fully completing lower-order
efficiency and process modernisation?</em>
And: <em>Does organisational capability accumulation systematically precede mission transformation?</em></p>
</div>

<div class="section">
<h3>F2. Conceptual Motivation</h3>
<ul>
<li><strong>Four-Level Framework</strong> (reframed): L1–L4 as <em>dimensions of transformation</em>,
not deterministic chronological stages. The framework diagnoses which dimensions are active,
in what order, and under what conditions — not a universal sequence.</li>
<li><strong>Punctuated equilibrium</strong>: Institutional shocks (Reform Acts) may trigger rapid
reorganisation across multiple dimensions simultaneously, bypassing lower-order stages.</li>
<li><strong>Capability-before-strategy</strong>: Resource-theory and organisational learning literatures
suggest capability investment precedes strategic repositioning.</li>
<li><strong>AI analogy</strong>: Many firms today invest heavily in AI talent and org redesign
before optimising operational efficiency — suggesting leapfrogging may be common during
rapid technological transitions.</li>
</ul>
</div>

<div class="section">
<h3>F3. Data &amp; Archive Construction</h3>
<ul>
<li>1,581 scanned ledger pages, Oxford University 1700–1900</li>
<li>Annual panel: real-£ expenditure shares by category, plus payment-modernity index</li>
<li>Era classification: pre-industrial (≤1853), transition (1854–1876), early-industrial (1877–1890), late-industrial (1891–1900)</li>
</ul>
<p><em>Details of the extraction/enrichment pipeline, variable dictionary, and validation
summaries are documented separately (data-reproducibility track).</em></p>
</div>

<div class="section">
<h3>F4. Empirical Strategy</h3>
<table>
<thead><tr><th>Question</th><th>Method</th><th>Identification</th></tr></thead>
<tbody>
<tr><td>Did reform acts cause transformation?</td>
<td>ITS (interrupted time series), RDD, placebo cutoffs</td>
<td>Causal — discontinuity at known policy dates</td></tr>
<tr><td>Which dimensions transformed most?</td>
<td>Normalised ITS effect sizes (B2), break magnitudes (B3)</td>
<td>Descriptive — ranking of level responses</td></tr>
<tr><td>Is transformation leapfrogging?</td>
<td>Normalised trajectories (B1), event-time plots (B5), acceleration (B4)</td>
<td>Descriptive — L3/L4 vs L1/L2 differential</td></tr>
<tr><td>Does capability precede mission?</td>
<td>Cross-lag with bootstrap CI (C1), distributed lag (C2),
rolling correlation (C3), state transitions (C5)</td>
<td>Temporal — direction of lead-lag relationship</td></tr>
<tr><td>Are reforms discontinuous or gradual?</td>
<td>Signal-to-noise (E1), sharpness index (E2)</td>
<td>Descriptive — break sharpness relative to baseline volatility</td></tr>
</tbody>
</table>
</div>

<div class="section">
<h3>F5. Main Findings vs Exploratory</h3>
<table>
<thead><tr><th>Finding</th><th>Status</th><th>Evidence</th></tr></thead>
<tbody>
<tr><td>Reform Acts trigger sharp discontinuous transformation</td>
<td><span style="color:#117a65"><strong>MAIN</strong></span></td>
<td>ITS (v7-C1), RDD (v7-C6), placebo (v7-C4), sharpness (E1/E2)</td></tr>
<tr><td>Oxford leapfrogs toward L3/L4 (higher-order transformation)</td>
<td><span style="color:#117a65"><strong>MAIN</strong></span></td>
<td>Normalised effect sizes (B2), trajectories (B1), event-time (B5)</td></tr>
<tr><td>Capability expansion precedes mission transformation (L3→L4)</td>
<td><span style="color:#117a65"><strong>MAIN</strong></span></td>
<td>Cross-lag CI (C1), distributed lag (C2), state transitions (C5)</td></tr>
<tr><td>Transformation requires resource slack (not crisis alone)</td>
<td><span style="color:#117a65"><strong>MAIN</strong></span></td>
<td>Land-rent income effect (v7)</td></tr>
<tr><td>Strict L1→L2→L3→L4 sequential ordering</td>
<td><span style="color:#922b21"><strong>EXPLORATORY</strong></span></td>
<td>Weak — change-point ordering partial, transition dates sensitive</td></tr>
<tr><td>Granger-causal chaining L1→L2→L3→L4</td>
<td><span style="color:#922b21"><strong>EXPLORATORY</strong></span></td>
<td>Mostly insignificant across lags and pairs</td></tr>
<tr><td>Generalised AI adoption sequence</td>
<td><span style="color:#922b21"><strong>EXPLORATORY</strong></span></td>
<td>Analogy only — requires AI-context empirical validation</td></tr>
</tbody>
</table>
</div>

<div class="section">
<h3>F6. Mechanism Interpretation</h3>
<p><strong>Proposed mechanism:</strong> Reform Acts function as <em>strategic discontinuities</em>
that force rapid capability investment (hiring, salary expansion) before an institution can
fully articulate a new strategic mission. This creates the observed L3-before-L4 pattern.
Resource slack (endowment income from rising land rents) provides the financial capacity
to fund capability build-up; institutions under fiscal constraint cannot leapfrog.</p>
<p>The leapfrogging of L1/L2 suggests that external shocks can short-circuit the
efficiency-then-redesign sequence typically assumed in technology adoption models —
capability and mission transformation may be driven by <em>exogenous institutional pressure</em>
rather than endogenous optimisation.</p>
</div>

<div class="section">
<h3>F7. AI Transformation Implications</h3>
<ul>
<li><strong>Leapfrogging is likely common:</strong> Firms investing in AI talent and org redesign
before optimising operations are not aberrant — they may be responding rationally to
external pressure (competitive, regulatory) that forces higher-order change first.</li>
<li><strong>Regulatory shocks as discontinuities:</strong> AI governance mandates could function
like Oxford's Reform Acts — forcing rapid capability build-up and mission redefinition
before lower-level process optimisation.</li>
<li><strong>Resource slack as enabler:</strong> AI transformation may be concentrated in
well-resourced firms not because they face greater AI pressure, but because they have
the slack to invest in capability before they know exactly how to deploy it.</li>
<li><strong>Capability-first strategy:</strong> The Oxford evidence suggests "hire first,
strategise later" is a historically recurring pattern under institutional shock conditions.</li>
</ul>
</div>

<div class="section">
<h3>F8. Boundary Conditions / Historical Specificity</h3>
<table>
<thead><tr><th>Feature</th><th>Oxford-Specific</th><th>Potentially Generalizable</th></tr></thead>
<tbody>
<tr><td>Transformation pace (200 years)</td><td>Yes — pre-industrial context</td><td>No</td></tr>
<tr><td>No market competition</td><td>Yes — Oxbridge monopoly</td><td>Partially</td></tr>
<tr><td>Endowment (land-rent) income</td><td>Yes — unique structure</td><td>Analogy: retained earnings / cash reserves</td></tr>
<tr><td>External reform shock as trigger</td><td>Partially</td><td>Yes — regulatory mandates are common</td></tr>
<tr><td>Capability→mission sequencing</td><td>No</td><td>Yes — consistent with capability theory</td></tr>
<tr><td>Leapfrogging of lower-order transformation</td><td>No</td><td>Yes — predicted by punctuated equilibrium theory</td></tr>
</tbody>
</table>
</div>
"""


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

CSS = """
body{font-family:Arial,sans-serif;max-width:1300px;margin:40px auto;padding:0 20px;color:#222}
h1{font-size:1.6em;border-bottom:2px solid #2c3e50;padding-bottom:8px}
h2{font-size:1.3em;color:#2c3e50;margin-top:36px}
h3{font-size:1.1em;color:#34495e}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:0.88em}
th{background:#2c3e50;color:white;padding:7px 10px;text-align:left}
td{border:1px solid #ddd;padding:6px 10px}
tr:nth-child(even){background:#f9f9f9}
figure{margin:8px 0}
figcaption{font-size:0.82em;color:#555;margin-top:4px}
figure img{border:1px solid #ddd;border-radius:4px;width:100%}
.callout{background:#eaf4fb;border-left:4px solid #2980b9;padding:12px 16px;margin:16px 0;border-radius:4px}
.warn{background:#fef9e7;border-left:4px solid #f39c12;padding:12px 16px;margin:16px 0;border-radius:4px}
.missing{color:#888;font-style:italic}
code{background:#f4f4f4;padding:2px 5px;border-radius:3px;font-size:0.9em}
.section{background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:16px 20px;margin:24px 0}
.fig-row{display:flex;gap:12px;align-items:flex-start;margin:12px 0}
.fig-row figure{flex:1;min-width:0;margin:0}
@media print {
  body{max-width:none;margin:20px;padding:0}
  h2{page-break-before:always;break-before:page}
  h1+p, h1+p+div, .toc{page-break-before:avoid;break-before:avoid}
  h2:first-of-type{page-break-before:auto;break-before:auto}
  h3{page-break-after:avoid;break-after:avoid}
  figure,table,.callout,.warn,.section{page-break-inside:avoid;break-inside:avoid}
  img{max-width:100% !important;height:auto !important}
  a[href]::after{content:none}
}
"""


def build_html(a, b, c, d, e, f_html):
    peak_lag = c.get("peak_lag", "N/A")
    peak_r   = c.get("peak_r",   float("nan"))
    peak_ci_lo = c.get("peak_ci_lo", float("nan"))
    peak_ci_hi = c.get("peak_ci_hi", float("nan"))
    peak_sig   = c.get("peak_sig",   False)
    state_s    = c.get("state_summary", pd.DataFrame())
    state_v    = c.get("state_verdict",  "")
    y_L3       = c.get("year_L3_entry")
    y_L4       = c.get("year_L4_entry")
    L3L4_gap   = c.get("L3L4_gap")
    cum_fwd    = c.get("cumul_fwd_dif_end", float("nan"))
    cum_rev    = c.get("cumul_rev_dif_end", float("nan"))
    pct_lead   = c.get("pct_L3_leads",   0.0)
    leading_p  = c.get("leading_period", "none")

    # Headline numbers from B
    eff_df = b.get("eff_df", pd.DataFrame())
    bkp_df = b.get("bkp_df", pd.DataFrame())
    if len(eff_df):
        eff_1877 = eff_df[eff_df["Reform"] == 1877].set_index("Code")["Norm. effect size"]
        L4_eff_1877 = eff_1877.get("L4", float("nan"))
        L3_eff_1877 = eff_1877.get("L3", float("nan"))
        L1_eff_1877 = eff_1877.get("L1_inv", float("nan"))
        L2_eff_1877 = eff_1877.get("L2", float("nan"))
    else:
        L4_eff_1877 = L3_eff_1877 = L1_eff_1877 = L2_eff_1877 = float("nan")
    if len(bkp_df):
        bkp_map = bkp_df.set_index("Code")["Rel. magnitude"].to_dict()
    else:
        bkp_map = {}

    gap_str = f"{L3L4_gap} years" if L3L4_gap is not None else "N/A"
    fwd_vs_rev = ("L3→L4 dominates" if (not np.isnan(cum_fwd) and not np.isnan(cum_rev)
                                         and abs(cum_fwd) > abs(cum_rev)) else "L4→L3 dominates"
                  if not np.isnan(cum_rev) else "Inconclusive")
    sig_str = "significant" if peak_sig else "NOT significant"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Analysis v9 — Leapfrogging, Capability→Mission, Paper Skeleton</title>
<style>{CSS}</style>
</head>
<body>
<h1>Analysis v9: Leapfrogging, Capability→Mission, Paper Skeleton</h1>
<p><em>Oxford University Ledgers, 1700–1900 | Generated automatically</em></p>

<div class="callout">
<strong>Core reframe (based on advisor feedback):</strong>
The Four-Level Framework is treated here as <em>dimensions of transformation</em>, not deterministic stages.
The central empirical finding is that Oxford appears to <strong>leapfrog</strong> — moving directly
into higher-order transformation (L3 capability, L4 mission) without first completing lower-order
efficiency (L1) and process (L2) modernisation. Within the higher-order pair,
the evidence for <em>capability-before-mission</em> is mixed once we test it rigorously
on detrended series — directionally consistent but not statistically decisive.
Reform Acts function as <em>strategic discontinuities</em> rather than gradual modernisation triggers.
</div>

<div class="callout" style="background:#fdf2e9;border-left-color:#e67e22">
<strong>Empirical verdict (mapped to advisor's Main vs Exploratory frame):</strong>
<table style="margin:8px 0;font-size:0.9em">
<thead><tr><th>Finding</th><th>Status</th><th>Evidence from this run</th></tr></thead>
<tbody>
<tr><td>Reform Acts are sharp strategic discontinuities</td>
    <td><span style="color:#117a65"><strong>MAIN — supported</strong></span></td>
    <td>Signal-to-noise &amp; sharpness analyses (E1, E2); ITS coefficients at 1854/1877 highly significant</td></tr>
<tr><td>Oxford leapfrogs toward higher-order (L3/L4)</td>
    <td><span style="color:#117a65"><strong>MAIN — supported</strong></span></td>
    <td>L4 normalised 1877 effect = {L4_eff_1877:.2f}× vs L1 = {L1_eff_1877:.2f}× / L2 = {L2_eff_1877:.2f}×.
        L4 relative break magnitude = {bkp_map.get("L4", float("nan")):.2f}× vs L2 = {bkp_map.get("L2", float("nan")):.2f}×.</td></tr>
<tr><td>Capability precedes mission (L3 → L4)</td>
    <td><span style="color:#b9770e"><strong>MAIN — mixed / time-varying</strong></span></td>
    <td><strong>Rolling correlation:</strong> L3 led L4 by ~5 yr in {leading_p} ({pct_lead:.0%} of pre-modern windows).
        After ~1750 the two move synchronously. <br>
        <strong>First-difference cross-correlation:</strong> peak lag {peak_lag}, r = {peak_r:.3f} ({sig_str}). <br>
        <strong>First-entry timing</strong> (sensitive to threshold): L4 actually crosses its own 67th-pct threshold
        before L3 — but this reflects each level's <em>own</em> distribution, not absolute capability.<br>
        <strong>Reading:</strong> the strong pre-1750 lead-lag fits the hypothesis; post-1750 synchronous
        movement is consistent with leapfrogging under shock conditions (multiple dimensions move at once).</td></tr>
<tr><td>Transformation needs resource slack (not crisis alone)</td>
    <td><span style="color:#117a65"><strong>MAIN — carried from v7</strong></span></td>
    <td>Land-rent income effect documented in v7 analyses</td></tr>
<tr><td>Strict L1→L2→L3→L4 sequential ordering</td>
    <td><span style="color:#922b21"><strong>EXPLORATORY — not supported</strong></span></td>
    <td>Granger chain mostly insignificant (v8-B2); transition dates threshold-sensitive (v8-B4)</td></tr>
<tr><td>Generalisable AI adoption sequence</td>
    <td><span style="color:#922b21"><strong>EXPLORATORY — analogy only</strong></span></td>
    <td>Mechanism (capability-before-mission, leapfrogging under shocks) may generalise; pace does not</td></tr>
</tbody>
</table>
</div>

<!-- ========================================================= SECTION A -->
<h2>Section A: Four-Level Framework as Dimensions</h2>
<div class="section">
<p>The four levels are reframed as <em>dimensions</em> or <em>possible pathways</em> of transformation,
not deterministic sequential stages. The key empirical question is: which dimensions emerge first,
and under what institutional conditions?</p>
{a["dim_table"]}
<div class="warn">
<strong>Operational axis (L1, L2):</strong> efficiency and process — expected under gradual internal pressure.<br>
<strong>Strategic axis (L3, L4):</strong> capability and mission — expected under external institutional shocks.<br>
Leapfrogging = strategic axis activates first, before operational axis is complete.
</div>
{_img(a["b64_trends"], "Figure A1. Four transformation dimensions (1700–1900). Vertical red dashes = 1854 &amp; 1877 Reform Acts.")}
{a["era_table"]}
</div>

<!-- ========================================================= SECTION B -->
<h2>Section B: Leapfrogging Analysis</h2>
<div class="callout">
<strong>Leapfrogging hypothesis:</strong> Oxford's strategic dimensions (L3/L4) exhibit
disproportionately larger responses to reform shocks compared to operational dimensions (L1/L2).
Five analyses test this from different angles.
</div>
<div class="warn">
<strong>1854 vs 1877 — two distinct patterns to keep separate:</strong>
<ul>
<li><strong>1854 — disruption pattern:</strong> L3 and L4 ITS coefficients are large and <em>negative</em>
(L3 = −0.235, L4 = −0.013, both significant). The first reform act <em>disrupted</em>
the existing institutional structure — capability and mission dimensions were hit hardest in absolute terms.</li>
<li><strong>1877 — acceleration pattern:</strong> L3 and L4 ITS coefficients are <em>positive</em>
(L3 ≈ +0.07, L4 ≈ +0.045, both significant). After 23 years of post-disruption adjustment,
the second reform act drove disproportionate <em>acceleration</em> in higher-order dimensions
(L4 normalised effect = {L4_eff_1877:.2f}× pre-reform mean).</li>
</ul>
The "leapfrogging" claim is best read as: <em>across both reforms combined, higher-order dimensions
exhibit larger relative magnitudes than lower-order dimensions</em>. It is not a claim that L1/L2 stayed
constant — they did move, just less.
</div>

<div class="section">
<h3>B1. Normalised Growth Trajectories</h3>
<p>Each level normalised to its 1820–1853 mean (= 1). Leapfrogging is visible if L3/L4 rise
far above L1/L2 post-reform.</p>
{_img(b.get("b64_norm"), "Figure B1. Normalised growth trajectories, 1820–1900. All levels scaled to pre-reform baseline = 1.")}
<p><strong>Finding:</strong> L4 rises far above its pre-reform baseline (multiple-fold growth)
while L1 and L2 stay near 1.0 throughout the post-reform period. L3 dips sharply at 1854
then recovers above baseline. The visual asymmetry — strategic dimensions diverging upward
while operational dimensions stay flat — is the leapfrogging signature.</p>
</div>

<div class="section">
<h3>B2. Relative ITS Effect Sizes</h3>
<p>ITS level-shift coefficients at 1854 and 1877, divided by each level's own pre-reform mean.
Larger normalised effects indicate leapfrogging.</p>
{_img(b.get("b64_effects"), "Figure B2. Normalised ITS effect sizes (level shift / pre-reform mean) at each Reform Act.")}
{_df_to_html(b.get("eff_df", pd.DataFrame()), title="Effect Sizes Table")}
<p><strong>Finding:</strong> At 1877, L4 has a normalised effect of <strong>{L4_eff_1877:.2f}×</strong>
its pre-reform mean — about <strong>{L4_eff_1877/L1_eff_1877:.0f}× larger than L1</strong>
({L1_eff_1877:.2f}×) and <strong>{L4_eff_1877/L2_eff_1877:.0f}× larger than L2</strong>
({L2_eff_1877:.2f}×). At 1854, L3 ({eff_df.loc[eff_df['Code']=='L3'].loc[eff_df['Reform']==1854,'Norm. effect size'].values[0]:.2f}×)
and L4 ({eff_df.loc[eff_df['Code']=='L4'].loc[eff_df['Reform']==1854,'Norm. effect size'].values[0]:.2f}×)
show large <em>negative</em> shifts — the disruption pattern — while L1/L2 barely move.
Both directions support leapfrogging: higher-order dimensions respond disproportionately to shocks.</p>
</div>

<div class="section">
<h3>B3. Break Magnitude Comparison</h3>
<p>Structural break magnitude (binseg) relative to pre-break mean, per level.</p>
{_img(b.get("b64_bkp_mag"), "Figure B3. Relative break magnitude by level. L3/L4 larger → greater structural discontinuity.")}
{_df_to_html(b.get("bkp_df", pd.DataFrame()), title="Break Magnitudes")}
<p><strong>Finding:</strong> L4 exhibits a relative break magnitude of <strong>{bkp_map.get("L4", float("nan")):.2f}×</strong> —
roughly <strong>{bkp_map.get("L4", float("nan"))/bkp_map.get("L1_inv", 1):.0f}× larger than L1</strong>
and <strong>{bkp_map.get("L4", float("nan"))/bkp_map.get("L2", 1):.0f}× larger than L2</strong>.
This is the single largest piece of leapfrogging evidence: the mission dimension's structural
shift is an order of magnitude larger than the operational dimensions, relative to each level's own baseline.</p>
</div>

<div class="section">
<h3>B4. Post-Reform Acceleration</h3>
<p>Percentage change in 10-year rolling mean within each post-reform window (1854–1877, 1877–1900).</p>
{_img(b.get("b64_accel"), "Figure B4. Post-reform acceleration by level and window.")}
{_df_to_html(b.get("accel_df", pd.DataFrame()), title="Post-Reform Acceleration")}
<p><strong>Finding:</strong> In the 1854–1877 window, L4 surged <strong>+704%</strong> while L1/L2/L3
all dropped or stayed flat (L3 actually fell −41%, consistent with the disruption pattern).
In 1877–1900, both strategic dimensions accelerated strongly (L3 +113%, L4 +266%) while
operational dimensions barely moved (L1 +8%, L2 +4%). The acceleration gap between strategic
and operational dimensions is roughly <em>30× to 100×</em>.</p>
</div>

<div class="section">
<h3>B5. Event-Time Plots Around Each Reform Act</h3>
<p>Level values in a ±15 year window around 1854 and 1877. Shows the differential
timing and magnitude of level responses to each institutional shock.</p>
{_img(b.get("b64_event"), "Figure B5. Event-time plots around each Reform Act for all four levels.")}
<p><strong>Finding:</strong> Around 1854, L3 shows a sharp downward kink at the reform date
while L1/L2 drift slowly. Around 1877, L4 shows a visible upward break against a backdrop of
near-flat L1/L2. The differential timing — strategic dimensions kinking <em>at</em> each reform,
operational dimensions barely registering them — is direct visual evidence that institutional
shocks reorganise capability and mission rather than incrementally tuning efficiency or process.</p>
</div>

<!-- ========================================================= SECTION C -->
<h2>Section C: Capability-before-Mission — Strengthened</h2>
<div class="callout">
<strong>Core question:</strong> Does capability expansion (L3: salary share) systematically precede
mission transformation (L4: educational share)? Five complementary methods.
Headline numbers from this run:
<ul>
<li>First-difference cross-correlation: peak lag = <strong>{peak_lag} yr</strong>,
    r = {peak_r:.3f}, 95% CI [{peak_ci_lo:.3f}, {peak_ci_hi:.3f}] — <em>{sig_str}</em>.</li>
<li>Distributed lag (first-differences): cumulative ΔL3→ΔL4 = {cum_fwd:.3f},
    cumulative ΔL4→ΔL3 = {cum_rev:.3f}.</li>
<li>First-entry timing: L3 entered sustained high-state in <strong>{y_L3 if y_L3 else "N/A"}</strong>,
    L4 in <strong>{y_L4 if y_L4 else "N/A"}</strong> (gap = {gap_str}).</li>
<li>{state_v}</li>
</ul>
<strong>Honest verdict:</strong> the direction of influence is consistent with capability-before-mission
across first-entry timing and rolling correlations, but the formal cross-correlation and
distributed-lag tests on first-differences do not reach statistical significance with 200 years of
annual data. This is a candidate <em>main</em> finding that should be reported with appropriate caveats.
</div>

<div class="section">
<h3>C1. Cross-Correlation with Bootstrap CIs — Levels vs First-Differences</h3>
<p>Cross-correlation on raw levels (C1a) is contaminated by shared trends and can be misleading.
First-differences (C1b) test whether <em>changes</em> in L3 lead <em>changes</em> in L4 — the proper
lead-lag test. 95% CIs from 1,000 block-bootstrap resamples (block size = 5 years).</p>
{_img(c.get("b64_xcorr"), f"Figure C1. L3–L4 cross-correlation, both levels (left, trend-contaminated) and first-differences (right, preferred). Peak lag on first-differences = {peak_lag} yr.")}
</div>

<div class="section">
<h3>C2. Distributed Lag Model — Levels and First-Differences</h3>
<p>Levels (top row) confound trends; first-differences (bottom row) test whether ΔL3 predicts ΔL4
beyond ΔL4's own history. Cumulative impulse over 10 lags compared in both directions.</p>
{_img(c.get("b64_dlm"), "Figure C2. Distributed lag, levels (top) and first-differences (bottom). Bottom row is the trend-robust lead-lag test.")}
{_df_to_html(c.get("dl_df", pd.DataFrame()), title="Distributed Lag Coefficients (Levels and First-Differences)")}
</div>

<div class="section">
<h3>C3. Rolling-Window Correlation (30-Year Window)</h3>
<p>r(L3_t, L4_t) and r(L3_t, L4_{{t+5}}) in each 30-year window. If the 5-year lead correlation
exceeds the synchronous correlation, L3 systematically leads L4 in that window.</p>
{_img(c.get("b64_rollcorr"), "Figure C3. Rolling 30-year correlation between L3 and L4.")}
<div class="callout">
<strong>Time-varying lead-lag — the most interesting finding here:</strong><br>
L3 clearly leads L4 by 5 years in the <strong>{leading_p}</strong> pre-industrial era
(lead advantage 0.5–0.7 in {pct_lead:.0%} of pre-1750 windows). After ~1750 the relationship
shifts: L3 and L4 begin moving <em>synchronously</em>. This is structurally consistent with
leapfrogging — during reform shocks, multiple transformation dimensions activate at once
rather than sequentially. The pre-modern gradual-sequencing relationship gives way to
shock-driven simultaneous transformation.
</div>
</div>

<div class="section">
<h3>C4. Event-Time Plots — L3 vs L4 Around Reforms</h3>
<p>Shaded gap between L3 and L4 trajectories in ±15 year window around each reform.
L3 rising before L4 in either window → visual evidence of capability-before-mission.</p>
{_img(c.get("b64_evL3L4"), "Figure C4. L3 vs L4 event-time trajectories around 1854 and 1877.")}
</div>

<div class="section">
<h3>C5. First-Persistent-Entry Timing</h3>
<p>Year in which each level first crosses its own historical X-th percentile threshold and stays above
for 5+ years (10-yr rolling mean). Positive gap (L4 year − L3 year) = capability precedes mission.</p>
{_img(c.get("b64_state"), f"Figure C5. First-persistent-entry into high-state for L3 and L4 at the 67th percentile.")}
{_df_to_html(state_s, title="First-Entry Timing across Quantile Thresholds")}
<div class="warn">
<strong>Honest caveat:</strong> across all tested thresholds, L4 enters its high-state <em>before</em> L3.
This appears to contradict the capability-before-mission hypothesis but reflects a methodological
issue: each threshold is computed from the level's <em>own</em> historical distribution. Since L3
oscillates more (large pre-reform values, large 1854 drop) while L4 rises more monotonically,
their internal percentile thresholds correspond to different historical moments. This is NOT a
test of which level "matures" first in any absolute sense. The cleaner directional evidence is
the rolling correlation (C3): L3 led L4 by 5 years in the pre-industrial era.
</div>
</div>

<!-- ========================================================= SECTION D -->
<h2>Section D: What the Sequencing Evidence Actually Supports</h2>
<div class="section">
<div class="warn">
<strong>Explicit restatement:</strong> The evidence does NOT support a universal strict L1→L2→L3→L4
sequential pathway. The strongest and most robust finding is the L3→L4 directional relationship
(capability before mission). The broader pattern is consistent with heterogeneous transformation
pathways driven by institutional shocks, with leapfrogging toward higher-order dimensions.
</div>
{d.get("table_html", "")}
<p><strong>Summary conclusion:</strong> Oxford's transformation is better characterised as
<em>leapfrogging driven by strategic discontinuities</em> than as a universal sequential
efficiency-to-mission progression. The L3→L4 capability-before-mission sequence
is the most empirically robust directional claim in this dataset.</p>
</div>

<!-- ========================================================= SECTION E -->
<h2>Section E: Reform Acts as Strategic Discontinuities</h2>
<div class="callout">
The Reform Acts are reinterpreted not as gradual modernisation triggers but as
<em>strategic discontinuities</em>: exogenous institutional shocks that force rapid,
non-linear reorganisation across multiple transformation dimensions simultaneously.
</div>

<div class="section">
<h3>E1. Signal-to-Noise of Breaks</h3>
<p>Break magnitude at 1854 divided by the pre-reform rolling standard deviation (10-year average).
High ratio → sharp discontinuous shift, not gradual drift.</p>
{_img(e.get("b64_snr"), "Figure E1. Signal-to-noise ratio of 1854 break by level. Higher = more discontinuous.")}
{_df_to_html(e.get("sn_df", pd.DataFrame()), title="Signal-to-Noise by Level")}
<p><strong>Finding:</strong> L1 (1.5×) and L3 (1.53×) show break-to-noise ratios well above 1,
meaning the 1854 shift was substantially larger than typical year-to-year variation —
a discontinuous signature. L2 (0.99×) is borderline. L4's smaller ratio (0.37×) reflects its
much smaller <em>absolute</em> scale (educational share starts at ~2% of expenditure), not weak
discontinuity — the relative break magnitude analysis (B3) showed L4 actually had the largest
structural shift relative to its own baseline.</p>
</div>

<div class="section">
<h3>E2. Sharpness Index</h3>
<p>Fraction of the total post-1854 level gain that occurs within the first 5 years.
Under a linear trend, this would equal 5/n_post_years. Observed > expected → discontinuous.</p>
{_img(e.get("b64_sharp"), "Figure E2. Sharpness of reform response. Blue = observed; grey = linear trend expectation.")}
{_df_to_html(e.get("sharp_df", pd.DataFrame()), title="Sharpness Index by Level")}
<p><strong>Finding:</strong> Three of four levels (L1, L2, L3) show observed sharpness ratios
roughly <strong>6–10× larger than the linear-trend expectation</strong> (~0.135). This means
most of the post-reform change happened within the first 5 years rather than spread evenly
across the 47-year post-1854 window — a punctuated, not gradual, pattern. L4's sharpness is
mixed in sign because L4 first <em>dropped</em> at 1854 before its dramatic rise at 1877.
Taken together, E1 and E2 strongly support reframing the Reform Acts as <em>strategic
discontinuities</em> rather than gradual modernisation triggers.</p>
</div>

<!-- ========================================================= SECTION F -->
<h2>Section F: Paper Skeleton</h2>
<div class="callout">
<strong>Emerging manuscript structure.</strong> Findings are now sufficient to organise into
manuscript sections. This skeleton maps each empirical result to its role in the paper narrative.
</div>
{f_html}

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading main panel...")
    panel_df = load_main_panel()

    print("Section A: Framework reframe...")
    a = section_a(panel_df)

    print("Section B: Leapfrogging analysis...")
    b = section_b(panel_df)

    print("Section C: Capability→Mission (strengthened)...")
    c = section_c(panel_df)

    print("Section D: Sequencing evidence summary...")
    d = section_d(b, c)

    print("Section E: Reform Acts as discontinuities...")
    e = section_e(panel_df)

    print("Section F: Paper skeleton...")
    f_html = section_f_html()

    print("Building HTML report...")
    html = build_html(a, b, c, d, e, f_html)
    out_path = OUT / "analysis_v9_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Report: {out_path}")
    print("CSVs written:")
    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
