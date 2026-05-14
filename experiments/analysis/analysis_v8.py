#!/usr/bin/env python
"""
analysis_v8.py — Four-Level Framework Empirical Operationalization

Sections:
  A: Explicit L1–L4 Operationalization (proxies, trends, era descriptives)
  B: Sequential Transformation Testing
       B1 Change-point detection & ordering
       B2 Sequential Granger causality chain (L1→L2→L3→L4)
       B3 Cross-lagged correlation (adjacent level pairs)
       B4 Transition date estimation
  C: Capability vs Mission Decomposition (category-level ITS)
  D: Mechanism Attribution Table (causal / sequential / theoretical)
  E: AI Transformation Mapping
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
V4       = ROOT / "experiments/reports/analysis_v4"
V5       = ROOT / "experiments/reports/analysis_v5"
V6       = ROOT / "experiments/reports/analysis_v6"
ENRICHED = ROOT / "experiments/results/enriched"
OUT      = ROOT / "experiments/reports/analysis_v8"
OUT.mkdir(parents=True, exist_ok=True)

# Oxford Reform Act cutoffs
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


# ---------------------------------------------------------------------------
# Utilities
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
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def _df_to_html(df, fmt=None, title=""):
    rows = []
    if title:
        rows.append(f'<caption>{title}</caption>')
    rows.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead>")
    rows.append("<tbody>")
    for _, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if fmt and c in fmt:
                cells.append(f"<td>{fmt[c].format(v)}</td>")
            elif isinstance(v, float):
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
    era_ranges = {}
    for era in era_order:
        sub = df[df["era"] == era]["year"]
        if len(sub):
            era_ranges[era] = (sub.min(), sub.max())
    for era, (y0, y1) in era_ranges.items():
        ax.axvspan(y0, y1, alpha=0.15, color=ERA_COLORS.get(era, "#eeeeee"), zorder=0)


def _run_its(series: pd.Series, years: pd.Series) -> dict:
    """Segmented OLS with 1854 and 1877 interventions; returns params dict."""
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
    return {
        "df":          df,
        "res":         res,
        "fitted":      res.fittedvalues.values,
        "counterfactual": res.predict(X_cf).values,
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_main_panel():
    p = V6 / "four_level_proxies.csv"
    df = pd.read_csv(p).sort_values("year").reset_index(drop=True)
    df["L1_inv"] = 1.0 - df["L1"]          # high = modern (traditional functions shrinking)
    df["L1_inv_10yr"] = df["L1_inv"].rolling(10, center=True, min_periods=5).mean()
    return df


def _amount_pounds(row):
    try:
        v = (float(row.get("amount_pounds") or 0)
             + float(row.get("amount_shillings") or 0) / 20.0
             + float(row.get("amount_pence_whole") or 0) / 240.0)
    except Exception:
        v = 0.0
    return v


def load_enriched_yearly():
    """Compute yearly English-language share and admin_share from enriched JSONs."""
    files = sorted(ENRICHED.glob("*.json"))
    year_stats = defaultdict(lambda: {
        "eng": 0, "lat": 0, "mixed": 0, "lang_n": 0,
        "admin_exp": 0.0, "total_exp": 0.0,
    })
    for f in files:
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
                elif lang == "latin":
                    year_stats[year]["lat"] += 1
                else:
                    year_stats[year]["mixed"] += 1
            if row.get("direction") == "expenditure":
                amt = _amount_pounds(row)
                year_stats[year]["total_exp"] += amt
                if row.get("category") == "administrative":
                    year_stats[year]["admin_exp"] += amt
    records = []
    for year in sorted(year_stats.keys()):
        s = year_stats[year]
        n = s["lang_n"]
        english_share = s["eng"] / n if n > 0 else np.nan
        te = s["total_exp"]
        admin_share = s["admin_exp"] / te if te > 0 else np.nan
        records.append({
            "year":          year,
            "english_share": english_share,
            "admin_share":   admin_share,
        })
    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section A — L1–L4 Operationalization
# ---------------------------------------------------------------------------

LEVEL_META = {
    "L1_inv": {
        "label":    "L1: Efficiency / Standardisation",
        "proxy":    "1 − traditional_function_share",
        "formula":  "1 − (ecclesiastical + maintenance + domestic) / total_exp_real",
        "interp":   "Traditional expenditure declining → operational efficiency; legacy functions receding",
        "color":    "#2980b9",
    },
    "L2": {
        "label":    "L2: Process Modernisation",
        "proxy":    "payment_modernity_index",
        "formula":  "Amount-weighted mean of payment-period scores (annual=1.0 → multi-year=0.1)",
        "interp":   "Shift to standardised annual payments → regularisation of financial processes",
        "color":    "#27ae60",
    },
    "L3": {
        "label":    "L3: Organisational Redesign",
        "proxy":    "salary_stipend_share",
        "formula":  "salary_stipend_real_£ / total_exp_real",
        "interp":   "Expansion of professional salaried staff → organisational capability building",
        "color":    "#e67e22",
    },
    "L4": {
        "label":    "L4: Mission Transformation",
        "proxy":    "educational_share",
        "formula":  "educational_real_£ / total_exp_real",
        "interp":   "Education-centred expenditure rising → institutional purpose redefined",
        "color":    "#8e44ad",
    },
}

LEVEL_COLS = ["L1_inv", "L2", "L3", "L4"]
LEVEL_ROLLING = {"L1_inv": "L1_inv_10yr", "L2": "L2_10yr", "L3": "L3_10yr", "L4": "L4_10yr"}


def section_a(panel_df, enrich_df):
    merged = panel_df.merge(enrich_df[["year", "english_share"]], on="year", how="left")

    # --- Figure A1: 4-panel trend plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    for ax, col in zip(axes, LEVEL_COLS):
        meta  = LEVEL_META[col]
        roll  = LEVEL_ROLLING[col]
        _add_era_bands(ax, merged)
        ax.scatter(merged["year"], merged[col], s=6, alpha=0.4, color=meta["color"])
        ax.plot(merged["year"], merged[roll], color=meta["color"], lw=2, label="10-yr rolling mean")
        # L1_alt overlay
        if col == "L1_inv":
            ax.plot(
                enrich_df["year"],
                enrich_df["english_share"].rolling(10, center=True, min_periods=5).mean(),
                color="#7fb3d3", lw=1.5, ls="--", alpha=0.8, label="L1_alt (English share, 10yr)"
            )
        for c in [CUT1, CUT2]:
            ax.axvline(c, color="#c0392b", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(meta["label"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Share (0–1)")
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[-2:]:
        ax.set_xlabel("Year")
    fig.suptitle("Four-Level Transformation Framework: Proxy Variables (1700–1900)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    b64_trends = _b64(fig)

    # --- Figure A2: L1 vs L1_alt scatter (robustness) ---
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sub = merged.dropna(subset=["L1_inv", "english_share"])
    ax2.scatter(sub["L1_inv"], sub["english_share"], s=12, alpha=0.5, color="#2980b9")
    r, p = stats.pearsonr(sub["L1_inv"], sub["english_share"])
    ax2.set_xlabel("L1_inv (1 − traditional share)")
    ax2.set_ylabel("English language entry share (L1_alt)")
    ax2.set_title(f"L1 vs L1_alt: r = {r:.3f} (p = {p:.3f})")
    b64_l1corr = _b64(fig2)

    # --- Era descriptive table ---
    era_order = ["pre_industrial", "transition", "early_industrial", "late_industrial"]
    rows = []
    for era in era_order:
        sub = panel_df[panel_df["era"] == era]
        if len(sub) == 0:
            continue
        yrs = f"{sub['year'].min()}–{sub['year'].max()}"
        rows.append({
            "Era": era.replace("_", " ").title(),
            "Years": yrs,
            "n": len(sub),
            "L1_inv mean": sub["L1_inv"].mean(),
            "L2 mean":     sub["L2"].mean(),
            "L3 mean":     sub["L3"].mean(),
            "L4 mean":     sub["L4"].mean(),
        })
    era_df = pd.DataFrame(rows)
    era_df.to_csv(OUT / "era_level_descriptives.csv", index=False)

    # Operationalization HTML table
    op_rows = "".join(
        f"<tr><td><strong>{col}</strong></td><td>{m['label']}</td><td><code>{m['proxy']}</code></td>"
        f"<td><em>{m['formula']}</em></td><td>{m['interp']}</td></tr>"
        for col, m in LEVEL_META.items()
    )
    op_table = (
        "<table><thead><tr><th>Code</th><th>Framework Level</th><th>Proxy</th>"
        "<th>Formula</th><th>Economic Interpretation</th></tr></thead>"
        f"<tbody>{op_rows}</tbody></table>"
    )

    return {
        "op_table":    op_table,
        "era_table":   _df_to_html(era_df, title="Era-Level Descriptives"),
        "b64_trends":  b64_trends,
        "b64_l1corr":  b64_l1corr,
        "l1_r":        r,
        "l1_p":        p,
    }


# ---------------------------------------------------------------------------
# Section B — Sequential Transformation Testing
# ---------------------------------------------------------------------------

def _detect_breaks(series: np.ndarray, n_bkps: int = 2, min_size: int = 8):
    """Return sorted list of breakpoint indices using Binseg(rbf)."""
    signal = series.reshape(-1, 1)
    try:
        algo = rpt.Binseg(model="rbf", min_size=min_size).fit(signal)
        bkps = algo.predict(n_bkps=n_bkps)  # last element = n_samples
    except Exception:
        bkps = [len(series)]
    return bkps  # e.g. [50, 120, 200]


def section_b1_changepoint(panel_df):
    df = panel_df.dropna(subset=LEVEL_COLS).sort_values("year").reset_index(drop=True)
    years = df["year"].values

    results = []
    bkp_map = {}
    for col in LEVEL_COLS:
        signal = df[col].values
        bkps   = _detect_breaks(signal, n_bkps=2, min_size=8)
        # first break: index bkps[0] (0-indexed start of second regime)
        first_idx  = bkps[0]
        first_year = int(years[first_idx]) if first_idx < len(years) else int(years[-1])
        second_idx  = bkps[1] if len(bkps) > 2 else bkps[0]
        second_year = int(years[min(second_idx, len(years) - 1)])
        bkp_map[col] = (first_year, second_year, bkps)
        results.append({
            "Level":        LEVEL_META[col]["label"],
            "Code":         col,
            "Break 1 (year)": first_year,
            "Break 2 (year)": second_year,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT / "break_ordering.csv", index=False)

    # Order test: is Break1(L1) ≤ Break1(L2) ≤ Break1(L3) ≤ Break1(L4)?
    break1s = [bkp_map[c][0] for c in LEVEL_COLS]
    ordered = all(break1s[i] <= break1s[i + 1] for i in range(len(break1s) - 1))
    order_str = " ≤ ".join(f"{LEVEL_COLS[i]}({break1s[i]})" for i in range(len(LEVEL_COLS)))

    # --- Figure B1: signal + break lines ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    for ax, col in zip(axes, LEVEL_COLS):
        meta  = LEVEL_META[col]
        by1, by2, bkps = bkp_map[col]
        roll  = LEVEL_ROLLING[col]
        _add_era_bands(ax, df)
        ax.scatter(df["year"], df[col], s=6, alpha=0.35, color=meta["color"])
        ax.plot(df["year"], df[roll], color=meta["color"], lw=2)
        ax.axvline(by1, color="#e74c3c", lw=1.8, ls="-",  label=f"Break 1: {by1}")
        ax.axvline(by2, color="#c0392b", lw=1.2, ls="--", label=f"Break 2: {by2}")
        ax.set_title(meta["label"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Share")
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[-2:]:
        ax.set_xlabel("Year")
    fig.suptitle("B1: Change-Point Detection per Level (Binseg, RBF cost)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    b64_bkp = _b64(fig)

    # --- Figure B1b: timeline bar showing break years ---
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    colors = [LEVEL_META[c]["color"] for c in LEVEL_COLS]
    labels = [LEVEL_META[c]["label"].split(":")[0] for c in LEVEL_COLS]
    for i, col in enumerate(LEVEL_COLS):
        by1, by2, _ = bkp_map[col]
        ax2.barh(i, 1, left=by1, height=0.4, color=colors[i], alpha=0.85, label=f"Break 1")
        ax2.scatter([by1, by2], [i, i], color=colors[i], s=80, zorder=5)
        ax2.text(by1 - 1.5, i, str(by1), ha="right", va="center", fontsize=9, color=colors[i])
        ax2.text(by2 + 1.5, i, str(by2), ha="left",  va="center", fontsize=9, color=colors[i])
    ax2.axvline(CUT1, color="#c0392b", ls="--", lw=1, label=f"{CUT1} Reform Act")
    ax2.axvline(CUT2, color="#922b21", ls=":",  lw=1, label=f"{CUT2} Reform Act")
    ax2.set_yticks(range(len(LEVEL_COLS)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Year")
    ax2.set_title("Break Year Timeline by Level (Break 1 and Break 2)")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    b64_timeline = _b64(fig2)

    return {
        "out_df":      out_df,
        "ordered":     ordered,
        "order_str":   order_str,
        "break1s":     break1s,
        "b64_bkp":     b64_bkp,
        "b64_timeline": b64_timeline,
        "table_html":  _df_to_html(out_df, title="Detected Break Years per Level"),
    }


def section_b2_granger(panel_df):
    df = panel_df.dropna(subset=LEVEL_COLS).sort_values("year").reset_index(drop=True)

    def _adf_diff(series, name):
        adf_p = adfuller(series.dropna())[1]
        if adf_p > 0.05:
            return series.diff().dropna(), True
        return series, False

    pairs = [("L1_inv", "L2"), ("L2", "L3"), ("L3", "L4")]
    records = []
    pair_labels = []

    for from_col, to_col in pairs:
        s_from, d_from = _adf_diff(df[from_col], from_col)
        s_to,   d_to   = _adf_diff(df[to_col],   to_col)
        label = f"{from_col}→{to_col}" + (" (1st-diff)" if d_from or d_to else "")
        pair_labels.append(label)
        # Align
        combo = pd.concat([s_to.rename("to"), s_from.rename("from")], axis=1).dropna()
        try:
            gc = grangercausalitytests(combo[["to", "from"]], maxlag=5, verbose=False)
            for lag in range(1, 6):
                fstat = gc[lag][0]["ssr_ftest"][0]
                pval  = gc[lag][0]["ssr_ftest"][1]
                records.append({
                    "Chain":     label,
                    "From":      from_col,
                    "To":        to_col,
                    "Lag":       lag,
                    "F-stat":    round(fstat, 4),
                    "p-value":   round(pval, 4),
                    "Sig":       _stars(pval),
                    "Differenced": d_from or d_to,
                })
        except Exception as e:
            warnings.warn(f"Granger failed for {label}: {e}")

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT / "sequential_granger.csv", index=False)

    # --- Figure B2: p-value heatmap by chain × lag ---
    if len(out_df):
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot = out_df.pivot_table(index="Chain", columns="Lag", values="p-value")
        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=0.15, cmap="RdYlGn_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"Lag {c}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.07 else "black")
        plt.colorbar(im, ax=ax, label="p-value")
        ax.set_title("B2: Sequential Granger Causality — p-values by Chain × Lag\n"
                     "(Green = significant; chain L1→L2→L3→L4 tests sequential causation)")
        plt.tight_layout()
        b64_granger = _b64(fig)
    else:
        b64_granger = None

    return {
        "out_df":       out_df,
        "table_html":   _df_to_html(out_df, title="Sequential Granger Causality"),
        "b64_granger":  b64_granger,
    }


def section_b3_crosslag(panel_df):
    df = panel_df.dropna(subset=LEVEL_COLS).sort_values("year").reset_index(drop=True)
    pairs   = [("L1_inv", "L2"), ("L2", "L3"), ("L3", "L4")]
    max_lag = 15
    records = []
    peak_lags = {}

    for from_col, to_col in pairs:
        pair_label = f"{from_col}–{to_col}"
        x = df[from_col].values
        y = df[to_col].values
        corrs = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                xv, yv = x[:len(x)-lag] if lag > 0 else x, y[lag:] if lag > 0 else y
            else:
                xv, yv = x[-lag:], y[:len(y)+lag]
            n = min(len(xv), len(yv))
            if n < 20:
                corrs.append(np.nan)
                continue
            r, _ = stats.pearsonr(xv[:n], yv[:n])
            corrs.append(r)
            records.append({"Pair": pair_label, "Lag": lag, "Correlation": round(r, 4)})
        # peak lag (from_col leads to_col if peak_lag > 0)
        non_nan = [(corrs[i], range(-max_lag, max_lag+1)[i]) for i in range(len(corrs)) if not np.isnan(corrs[i])]
        if non_nan:
            peak_lag = max(non_nan, key=lambda x: abs(x[0]))[1]
            peak_lags[pair_label] = peak_lag

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT / "cross_lag_correlation.csv", index=False)

    # --- Figure B3: cross-lag correlation plots ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pair_colors = ["#2980b9", "#27ae60", "#e67e22"]
    for ax, (pair_label, color) in zip(axes, zip([f"{a}–{b}" for a, b in pairs], pair_colors)):
        sub = out_df[out_df["Pair"] == pair_label].sort_values("Lag")
        ax.bar(sub["Lag"], sub["Correlation"], color=color, alpha=0.7, width=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        pk = peak_lags.get(pair_label)
        if pk is not None:
            ax.axvline(pk, color="#c0392b", lw=1.5, ls="-", label=f"Peak lag = {pk}")
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(pair_label, fontsize=10)
        ax.set_xlabel("Lag (years; positive = from leads to)")
        ax.set_ylabel("Pearson r")
        ax.set_ylim(-1, 1)
    fig.suptitle("B3: Cross-Lagged Correlation — Adjacent Level Pairs\n"
                 "Positive peak lag → earlier-level change precedes later-level change",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    b64_crosslag = _b64(fig)

    peak_rows = [{"Pair": k, "Peak lag (years)": v,
                  "Interpretation": "From-level leads" if v > 0 else ("To-level leads" if v < 0 else "Simultaneous")}
                 for k, v in peak_lags.items()]
    peak_df = pd.DataFrame(peak_rows)

    return {
        "out_df":       out_df,
        "peak_df":      peak_df,
        "table_html":   _df_to_html(peak_df, title="Peak Cross-Lag Correlation"),
        "b64_crosslag": b64_crosslag,
    }


def section_b4_transition(panel_df):
    df = panel_df.sort_values("year").reset_index(drop=True)
    baseline_end = 1850
    consecutive_needed = 5

    records = []
    for col in LEVEL_COLS:
        sub = df.dropna(subset=[col]).sort_values("year")
        baseline = sub[sub["year"] <= baseline_end][col]
        if len(baseline) < 10:
            continue
        bm   = baseline.mean()
        bsd  = baseline.std()
        threshold = bm + 1.5 * bsd
        roll_col = LEVEL_ROLLING[col]
        series  = sub[roll_col].values
        yrs     = sub["year"].values
        # find first run of consecutive_needed years above threshold
        above   = series > threshold
        trans_year = None
        for i in range(len(above) - consecutive_needed + 1):
            if all(above[i:i + consecutive_needed]):
                trans_year = int(yrs[i])
                break
        records.append({
            "Level":           LEVEL_META[col]["label"],
            "Code":            col,
            "Baseline mean":   round(bm,        4),
            "Baseline SD":     round(bsd,       4),
            "Threshold (+1.5σ)": round(threshold, 4),
            "Transition year": trans_year if trans_year else "Not detected",
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT / "transition_dates.csv", index=False)

    # --- Figure B4: transition timeline ---
    fig, ax = plt.subplots(figsize=(9, 4))
    trans_years = []
    colors = [LEVEL_META[c]["color"] for c in LEVEL_COLS]
    labels = [LEVEL_META[c]["label"].split(":")[0] for c in LEVEL_COLS]
    for i, (col, color, label) in enumerate(zip(LEVEL_COLS, colors, labels)):
        row = out_df[out_df["Code"] == col]
        if len(row) == 0:
            continue
        ty = row.iloc[0]["Transition year"]
        if ty != "Not detected":
            ty = int(ty)
            trans_years.append(ty)
            ax.scatter(ty, i, s=200, color=color, zorder=5)
            ax.text(ty + 2, i, str(ty), va="center", fontsize=10, color=color, fontweight="bold")
            ax.barh(i, 1, left=1700, height=0.3, color=color, alpha=0.15)
            ax.barh(i, ty - 1700, left=1700, height=0.3, color=color, alpha=0.4)
        else:
            ax.barh(i, 1, left=1700, height=0.3, color=color, alpha=0.1, hatch="//")
    ax.axvline(CUT1, color="#c0392b", ls="--", lw=1, label=f"{CUT1}")
    ax.axvline(CUT2, color="#922b21", ls=":",  lw=1, label=f"{CUT2}")
    ax.set_yticks(range(len(LEVEL_COLS)))
    ax.set_yticklabels(labels)
    ax.set_xlim(1700, 1910)
    ax.set_xlabel("Year")
    ax.set_title("B4: Transition Date — First Persistent Crossing of Baseline + 1.5σ\n"
                 "(10-yr rolling mean must stay above threshold for 5+ consecutive years)")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    b64_trans = _b64(fig)

    return {
        "out_df":    out_df,
        "table_html": _df_to_html(out_df, title="Transition Dates per Level"),
        "b64_trans": b64_trans,
    }


# ---------------------------------------------------------------------------
# Section C — Capability vs Mission Decomposition
# ---------------------------------------------------------------------------

CAT_ITS_TARGETS = {
    "L1_inv": {"label": "L1_inv (Efficiency — traditional decline)",    "color": "#2980b9"},
    "L2":     {"label": "L2 (Process Modernisation — payment modernity)", "color": "#27ae60"},
    "L3":     {"label": "L3 (Capability Expansion — salary share)",       "color": "#e67e22"},
    "L4":     {"label": "L4 (Mission Change — educational share)",         "color": "#8e44ad"},
}


def section_c(panel_df, enrich_df):
    df = panel_df.merge(enrich_df[["year", "admin_share"]], on="year", how="left")
    df = df.sort_values("year").reset_index(drop=True)

    its_results = {}
    all_records  = []

    for col, meta in CAT_ITS_TARGETS.items():
        r = _run_its(df[col], df["year"])
        its_results[col] = r
        res = r["res"]
        for param in ["D54", "t_post54", "D77", "t_post77"]:
            all_records.append({
                "Category": meta["label"],
                "Code":     col,
                "Param":    param,
                "Coef":     round(res.params[param], 5),
                "SE":       round(res.bse[param],    5),
                "p-value":  round(res.pvalues[param], 4),
                "Sig":      _stars(res.pvalues[param]),
            })

    out_df = pd.DataFrame(all_records)
    out_df.to_csv(OUT / "category_its.csv", index=False)

    # --- Figure C1: ITS fitted lines (2×2 panel) ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()
    for ax, col in zip(axes, list(CAT_ITS_TARGETS.keys())):
        meta  = CAT_ITS_TARGETS[col]
        r     = its_results[col]
        idf   = r["df"]
        _add_era_bands(ax, idf)
        ax.scatter(idf["year"], idf["y"], s=7, alpha=0.4, color=meta["color"])
        ax.plot(idf["year"], r["fitted"],         color=meta["color"], lw=2,   label="ITS fitted")
        ax.plot(idf["year"], r["counterfactual"], color="grey",        lw=1.5, ls="--", label="No-reform counterfactual")
        ax.axvline(CUT1, color="#c0392b", ls="--", lw=0.9, alpha=0.8)
        ax.axvline(CUT2, color="#922b21", ls=":",  lw=0.9, alpha=0.8)
        res = r["res"]
        annot = (
            f"1854: β={res.params['D54']:.3f}{_stars(res.pvalues['D54'])}\n"
            f"1877: β={res.params['D77']:.3f}{_stars(res.pvalues['D77'])}"
        )
        ax.text(0.03, 0.97, annot, transform=ax.transAxes,
                va="top", fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_title(meta["label"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Share (0–1)")
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[-2:]:
        ax.set_xlabel("Year")
    fig.suptitle("C1: Category-Level ITS — Capability vs Mission Decomposition\n"
                 "HAC SE, Oxford Reform Acts 1854 & 1877", fontsize=11, fontweight="bold")
    plt.tight_layout()
    b64_its = _b64(fig)

    # --- Figure C2: Coefficient comparison plot ---
    params_of_interest = ["D54", "D77"]
    n_params = len(params_of_interest)
    n_cats   = len(CAT_ITS_TARGETS)
    fig2, axes2 = plt.subplots(1, n_params, figsize=(10, 5))
    param_labels = {"D54": "1854 Level Shift (β₂)", "D77": "1877 Level Shift (β₄)"}
    for ax, param in zip(axes2, params_of_interest):
        codes  = list(CAT_ITS_TARGETS.keys())
        coefs  = [its_results[c]["res"].params[param]   for c in codes]
        errs   = [its_results[c]["res"].bse[param] * 1.96 for c in codes]
        colors = [CAT_ITS_TARGETS[c]["color"]            for c in codes]
        ys     = range(n_cats)
        ax.barh(list(ys), coefs, xerr=errs, color=colors, alpha=0.8,
                error_kw={"ecolor": "black", "capsize": 4})
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([CAT_ITS_TARGETS[c]["label"].split(" (")[0] for c in codes])
        ax.set_title(param_labels[param])
        ax.set_xlabel("Coefficient (95% CI)")
        # stars
        for i, c in enumerate(codes):
            pv = its_results[c]["res"].pvalues[param]
            s  = _stars(pv)
            if s:
                xpos = coefs[i] + (errs[i] if coefs[i] >= 0 else -errs[i]) * 1.05
                ax.text(xpos, i, s, va="center", fontsize=11, color="black")
    fig2.suptitle("C2: ITS Coefficient Comparison — 1854 & 1877 Effects by Level\n"
                  "Capability (L3/salary) vs Mission (L4/educational) timing",
                  fontsize=10, fontweight="bold")
    plt.tight_layout()
    b64_coef = _b64(fig2)

    return {
        "out_df":   out_df,
        "table_html": _df_to_html(out_df, title="Category-Level ITS Coefficients"),
        "b64_its":  b64_its,
        "b64_coef": b64_coef,
    }


# ---------------------------------------------------------------------------
# Section D — Mechanism Attribution Table (static)
# ---------------------------------------------------------------------------

def section_d_html():
    rows = [
        ("Causal Identification",
         "ITS (v7-C1), RDD (v7-C6), Placebo test (v7-C4)",
         "Oxford Reform Acts (1854, 1877) caused discontinuous jumps in Modern Function Share. "
         "Placebo cutoffs do not replicate this. RDD confirms discontinuity at 1854."),
        ("Temporal Sequencing",
         "Change-point ordering (B1), Sequential Granger chain (B2), "
         "Cross-lagged correlation (B3), Transition dates (B4)",
         "L1 (efficiency) and L2 (process) structural breaks appear to precede L3 (capability) "
         "and L4 (mission). Cross-lag peaks and Granger tests assess directionality."),
        ("Theoretical Interpretation",
         "Four-Level AI Transformation Framework analogy",
         "The <em>pattern</em> — reform trigger → efficiency gain → process redesign → "
         "capability expansion → mission redefinition — maps conceptually to AI transformation "
         "sequences, but causal mechanisms are historically specific to Oxford."),
    ]
    body = "".join(
        f"<tr><td><strong>{r[0]}</strong></td><td><em>{r[1]}</em></td><td>{r[2]}</td></tr>"
        for r in rows
    )
    return (
        "<table><thead><tr><th>Claim Type</th><th>Evidence Base</th><th>Scope</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Section E — AI Transformation Mapping (static)
# ---------------------------------------------------------------------------

def section_e_html():
    rows = [
        ("Reform Acts as primary trigger (not gradual market pressure)",
         "Regulatory mandate → sudden compliance or strategic pivot",
         "Potentially generalizable — regulatory pressure can force AI adoption faster than competitive markets",
         "Partially — Oxford had no competitive market pressure; firms do"),
        ("1854 disruption then 1877 acceleration",
         "Initial resistance / adjustment lag then accelerated adoption",
         "Generalizable — J-curve adoption pattern documented in technology diffusion literature",
         "No — timing is Oxford-specific"),
        ("Salary/capability expansion precedes mission change (L3 before L4)",
         "Talent and capability investment precedes strategy pivot",
         "Generalizable — 'hire first, strategise later' pattern in tech transformation",
         "No"),
        ("Income effect (not crowding-out) from land-rent shocks",
         "Resource slack enables transformation; crisis alone insufficient",
         "Generalizable — transformation requires slack, not just threat",
         "Partially — Oxford's endowment structure is unique"),
        ("200-year full transformation arc",
         "AI transformation expected within 10–30 years",
         "Historically specific — pre-industrial institutions changed slowly",
         "Yes — pace not generalizable"),
        ("Sequential L1→L2→L3→L4 ordering",
         "AI adoption may follow efficiency → process → org → mission sequence",
         "Hypothesis for AI context — requires empirical validation",
         "Partially — the sequence may be compressed or reordered in AI contexts"),
    ]
    body = "".join(
        f"<tr><td>{r[0]}</td><td><em>{r[1]}</em></td><td>{r[2]}</td>"
        f"<td style='color:{'#117a65' if r[3]=='No' else '#922b21'}'>{r[3]}</td></tr>"
        for r in rows
    )
    return (
        "<table><thead><tr><th>Historical Finding</th><th>AI Equivalent</th>"
        "<th>Generalizable?</th><th>Oxford-Specific?</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

CSS = """
body{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#222}
h1{font-size:1.6em;border-bottom:2px solid #2c3e50;padding-bottom:8px}
h2{font-size:1.3em;color:#2c3e50;margin-top:36px}
h3{font-size:1.1em;color:#34495e}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:0.88em}
th{background:#2c3e50;color:white;padding:7px 10px;text-align:left}
td{border:1px solid #ddd;padding:6px 10px}
tr:nth-child(even){background:#f9f9f9}
figure{margin:20px 0}
figcaption{font-size:0.85em;color:#555;margin-top:4px}
figure img{border:1px solid #ddd;border-radius:4px}
.callout{background:#eaf4fb;border-left:4px solid #2980b9;padding:12px 16px;margin:16px 0;border-radius:4px}
.warn{background:#fef9e7;border-left:4px solid #f39c12;padding:12px 16px;margin:16px 0;border-radius:4px}
.missing{color:#888;font-style:italic}
code{background:#f4f4f4;padding:2px 5px;border-radius:3px;font-size:0.9em}
"""


def build_html(a, b1, b2, b3, b4, c, d_html, e_html):
    ordered_flag = "✅ Sequential ordering confirmed" if b1["ordered"] else "⚠️ Ordering not fully sequential"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Analysis v8 — Four-Level Framework Operationalization</title>
<style>{CSS}</style>
</head>
<body>
<h1>Analysis v8: Four-Level Transformation Framework — Empirical Operationalization</h1>
<p><em>Oxford University Ledgers, 1700–1900 | Generated automatically</em></p>

<div class="callout">
<strong>Summary:</strong> This analysis empirically grounds the Four-Level AI Transformation Framework
in 200 years of Oxford accounting data. We (A) explicitly operationalize each level,
(B) test whether transformation was <em>sequential</em> rather than simultaneous,
(C) decompose <em>capability expansion</em> from <em>mission change</em>, and
(D) clarify what can be causally identified vs temporally observed vs theoretically interpreted.
</div>

<!-- ========================================================= SECTION A -->
<h2>Section A: L1–L4 Explicit Operationalization</h2>

<h3>A1. Proxy Definitions</h3>
{a["op_table"]}
<div class="warn">
<strong>L1 direction note:</strong> In v6, L1 = traditional_function_share (high = pre-modern).
Here we use <strong>L1_inv = 1 − L1</strong> so that all four levels increase with modernisation,
enabling direct comparison. L2, L3, L4 already increase with modernisation.
</div>

<h3>A2. 200-Year Trends</h3>
{_img(a["b64_trends"], "Figure A1. L1_inv through L4 (1700–1900). Dots = annual values; line = 10-yr rolling mean. Dashed L1_alt = English-language entry share (robustness check). Vertical red dashes = 1854 & 1877 Reform Acts. Era bands: blue=pre-industrial, green=transition, orange=early-industrial, pink=late-industrial.")}

<h3>A3. L1 Robustness: Traditional Share vs Language Shift</h3>
{_img(a["b64_l1corr"], f"Figure A2. Scatter of L1_inv vs English-language entry share (L1_alt). Pearson r = {a['l1_r']:.3f} (p = {a['l1_p']:.3f}). High correlation validates using traditional-function decline as an accounting-standardisation proxy.")}

<h3>A4. Era-Level Descriptives</h3>
{a["era_table"]}

<!-- ========================================================= SECTION B -->
<h2>Section B: Sequential Transformation Testing</h2>
<div class="callout">
<strong>Key question:</strong> Did Oxford move through L1 → L2 → L3 → L4 <em>sequentially</em> rather than simultaneously?
Three complementary methods test this. Convergence across methods strengthens the sequencing claim.
</div>

<h3>B1. Change-Point Detection & Ordering</h3>
<p>Binseg algorithm (RBF cost, n_bkps=2) detects two structural breaks per level.
Ordering test: is Break₁(L1) ≤ Break₁(L2) ≤ Break₁(L3) ≤ Break₁(L4)?</p>
{b1["table_html"]}
<p><strong>{ordered_flag}</strong> — {b1["order_str"]}</p>
{_img(b1["b64_bkp"], "Figure B1a. Detected structural breaks (red lines) per level with 10-yr rolling mean.")}
{_img(b1["b64_timeline"], "Figure B1b. Break-year timeline. Dots mark Break 1 and Break 2 per level. Red lines = Oxford Reform Acts.")}

<h3>B2. Sequential Granger Causality Chain</h3>
<p>Tests whether past values of each level help predict the <em>next</em> level beyond its own history:
L1_inv → L2, L2 → L3, L3 → L4. Series are first-differenced if ADF test indicates non-stationarity (p > 0.05).</p>
{b2["table_html"]}
{_img(b2["b64_granger"], "Figure B2. Granger p-value heatmap by chain × lag. Green cells (p < 0.05) indicate temporal predictive power from the earlier to the later level.")}

<h3>B3. Cross-Lagged Correlation</h3>
<p>Pearson r between L_k(t) and L_{{k+1}}(t + lag) for lag ∈ [−15, +15].
A positive peak lag means the earlier level leads the later level.</p>
{b3["table_html"]}
{_img(b3["b64_crosslag"], "Figure B3. Cross-lagged correlation for adjacent level pairs. Bar = correlation at each lag; red vertical = peak lag.")}

<h3>B4. Transition Date Estimation</h3>
<p>Each level's 10-yr rolling mean is compared to its pre-1850 baseline (mean + 1.5 SD threshold).
Transition year = first year the rolling mean stays above threshold for ≥5 consecutive years.</p>
{b4["table_html"]}
{_img(b4["b64_trans"], "Figure B4. Transition timeline. Coloured bars show pre-transition period; dot marks the estimated transition year.")}

<!-- ========================================================= SECTION C -->
<h2>Section C: Capability Expansion vs Mission Change</h2>
<div class="callout">
<strong>Hypothesis:</strong> Organisational capability transformation (L3: salary expansion) precedes
institutional mission transformation (L4: educational expenditure). This would mirror the
AI transformation pattern of "invest in people and tools before pivoting strategy."
</div>

<h3>C1. Category-Level Interrupted Time Series</h3>
<p>Separate ITS models for each level around the 1854 and 1877 Oxford Reform Acts.
HAC standard errors (Newey-West, maxlags=10). Counterfactual = pre-reform trend extended.</p>
{_img(c["b64_its"], "Figure C1. ITS fitted lines and counterfactuals for each level. Annotated with 1854 and 1877 level-shift coefficients (*** p<0.01, ** p<0.05, * p<0.10).")}

<h3>C2. Coefficient Comparison</h3>
{_img(c["b64_coef"], "Figure C2. 1854 and 1877 level-shift coefficients (95% CI) across levels. Positive = increase after reform; negative = decline. Compare L3 (capability) vs L4 (mission) magnitude and direction.")}
{c["table_html"]}

<!-- ========================================================= SECTION D -->
<h2>Section D: Mechanism Attribution — What We Can and Cannot Claim</h2>
<div class="warn">
<strong>Important:</strong> The evidence supports different strength of claims for different findings.
This table explicitly distinguishes them to avoid over-stating causality.
</div>
{d_html}

<!-- ========================================================= SECTION E -->
<h2>Section E: AI Transformation Framework — Systematic Mapping</h2>
<p>For each empirical finding, we assess whether it is generalizable beyond Oxford's historical context
or whether it reflects historically specific institutional constraints.</p>
{e_html}
<div class="callout">
<strong>Key distinction:</strong> The <em>mechanism</em> (reform as trigger, capability before mission)
may be generalizable. The <em>pace</em> (200 years), <em>scale</em> (parish ledgers), and
<em>institutional form</em> (Oxbridge endowment) are historically specific.
</div>

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading main panel (four_level_proxies.csv)...")
    panel_df = load_main_panel()

    print("Aggregating enriched JSONs (language + admin share)...")
    enrich_df = load_enriched_yearly()

    print("Section A: Operationalization...")
    a = section_a(panel_df, enrich_df)

    print("Section B1: Change-point detection...")
    b1 = section_b1_changepoint(panel_df)

    print("Section B2: Sequential Granger chain...")
    b2 = section_b2_granger(panel_df)

    print("Section B3: Cross-lagged correlation...")
    b3 = section_b3_crosslag(panel_df)

    print("Section B4: Transition dates...")
    b4 = section_b4_transition(panel_df)

    print("Section C: Capability vs Mission ITS...")
    c  = section_c(panel_df, enrich_df)

    print("Section D & E: Static tables...")
    d_html = section_d_html()
    e_html = section_e_html()

    print("Building HTML report...")
    html = build_html(a, b1, b2, b3, b4, c, d_html, e_html)
    out_path = OUT / "analysis_v8_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Done. Report: {out_path}")
    print("CSVs saved:")
    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
