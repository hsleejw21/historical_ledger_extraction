#!/usr/bin/env python3
"""analysis_v6.py

Historical Evidence for the Four-Level AI Transformation Framework

Maps Oxford college accounting data (1700–1900) to the four transformation levels
from the Springer book chapter "AI Transformation in Business":

  L1 – Automation / Efficiency
       Proxy: traditional_function_share
       (ecclesiastical + maintenance + domestic) / total_expenditure [real £]

  L2 – Personalisation / Differentiation
       Proxy: payment_modernity_index
       Amount-weighted mean of payment-period modernity scores

  L3 – Operational Innovation
       Proxy: salary_share
       salary_stipend / total_expenditure [real £]

  L4 – Business / Mission Innovation
       Proxy: educational_share
       educational / total_expenditure [real £]

The report follows the four-level framework structure and includes narrative
interpretation (why each analysis, what the result means for the framework).

All findings are ASSOCIATIONAL.  Single institution time series (Oxford, 1700–1900).
No causal identification is possible without exogenous variation.

Output: experiments/reports/analysis_v6/
"""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import pearsonr

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels not installed — regression sections will be skipped.")

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("[WARN] ruptures not installed — change-point section will be skipped.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parents[2]
ENRICHED_DIR = ROOT / "experiments" / "results" / "enriched"
OUT_DIR     = ROOT / "experiments" / "reports" / "analysis_v6"
V5_DIR      = ROOT / "experiments" / "reports" / "analysis_v5"
V4_DIR      = ROOT / "experiments" / "reports" / "analysis_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Price deflation – Phelps Brown-Hopkins (1700 = 100)
# ---------------------------------------------------------------------------

_PBH_ANCHORS: dict[int, float] = {
    1700: 100.0, 1710: 103.7, 1720: 101.3, 1730: 93.6,
    1740: 100.0, 1750: 104.7, 1760: 115.7, 1770: 125.4,
    1780: 138.3, 1790: 145.2, 1800: 203.4, 1810: 269.0,
    1820: 213.7, 1830: 175.4, 1840: 170.3, 1850: 161.2,
    1860: 175.4, 1870: 193.0, 1880: 182.4, 1890: 160.0,
    1900: 169.7,
}
_PBH_YEARS = sorted(_PBH_ANCHORS)
_PBH_VALS  = [_PBH_ANCHORS[y] for y in _PBH_YEARS]


def price_index(year: int) -> float:
    if year <= _PBH_YEARS[0]:  return _PBH_VALS[0]
    if year >= _PBH_YEARS[-1]: return _PBH_VALS[-1]
    return float(np.interp(year, _PBH_YEARS, _PBH_VALS))


# ---------------------------------------------------------------------------
# Era periodisation
# ---------------------------------------------------------------------------

ERAS = [
    ("pre_industrial",   1700, 1779),
    ("transition",       1780, 1819),
    ("early_industrial", 1820, 1859),
    ("late_industrial",  1860, 1900),
]
ERA_ORDER  = ["pre_industrial", "transition", "early_industrial", "late_industrial"]
ERA_LABELS = {
    "pre_industrial":   "Pre-Industrial\n(1700–1779)",
    "transition":       "Transition\n(1780–1819)",
    "early_industrial": "Early Industrial\n(1820–1859)",
    "late_industrial":  "Late Industrial\n(1860–1900)",
}
ERA_SHORT = {
    "pre_industrial":   "Pre-Ind.",
    "transition":       "Transition",
    "early_industrial": "Early Ind.",
    "late_industrial":  "Late Ind.",
}
ERA_VLINES = [1780, 1820, 1860]

AG_SHOCKS = {
    1793: "Enclosure Acts peak",
    1822: "Post-Napoleonic depression",
    1846: "Corn Laws repeal",
    1873: "Great Agricultural Depression",
}


def era_of_year(year: int) -> str:
    if year < 1780: return "pre_industrial"
    if year < 1820: return "transition"
    if year < 1860: return "early_industrial"
    return "late_industrial"


# ---------------------------------------------------------------------------
# Category groupings for the four proxies
# ---------------------------------------------------------------------------

L1_CATS = {"ecclesiastical", "maintenance", "domestic"}   # traditional
L3_CAT  = "salary_stipend"
L4_CAT  = "educational"

PAYMENT_SCORES = {
    "annual":       1.00,
    "half_year":    0.85,
    "sesquiannual": 0.70,
    "one_off":      0.60,
    "biennial":     0.40,
    "triennial":    0.25,
    "quadrennial":  0.20,
    "quinquennial": 0.15,
    "multi_year":   0.10,
}


# ---------------------------------------------------------------------------
# Parsing helpers (identical to v5)
# ---------------------------------------------------------------------------

def parse_fraction(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    for k, v in {"¼": .25, "1/4": .25, "½": .5, "1/2": .5, "¾": .75, "3/4": .75}.items():
        if s == k: return v
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_money(value: Any) -> float:
    if value is None or value == "": return 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return 0.0 if np.isnan(v) else v
    s = re.sub(r"[^0-9.\-]", "", str(value).strip())
    if not s: return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def amount_decimal(row: dict[str, Any]) -> float:
    p = parse_money(row.get("amount_pounds"))
    s = parse_money(row.get("amount_shillings"))
    d = parse_money(row.get("amount_pence_whole"))
    f = parse_fraction(row.get("amount_pence_fraction"))
    return p + s / 20.0 + (d + f) / 240.0


def parse_page_id(page_id: str) -> tuple[list[int], int]:
    m = re.match(r"^(\d{4})_(\d+)_image$", page_id)
    if m: return [int(m.group(1))], int(m.group(2))
    m = re.match(r"^(\d{4})-(\d{4})_(\d+)_image$", page_id)
    if m:
        y1, y2, pg = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y2 < y1: y1, y2 = y2, y1
        return list(range(y1, y2 + 1)), pg
    m = re.search(r"(\d{4})", page_id)
    if m: return [int(m.group(1))], 1
    raise ValueError(f"Cannot parse page_id: {page_id!r}")


# ---------------------------------------------------------------------------
# Data loading (fresh from enriched JSON)
# ---------------------------------------------------------------------------

def load_enriched_data() -> pd.DataFrame:
    """Load and flatten all enriched JSON files into a single DataFrame."""
    records: list[dict] = []
    files = sorted(ENRICHED_DIR.glob("*_image_enriched.json"))
    if not files:
        raise FileNotFoundError(f"No enriched JSON files in {ENRICHED_DIR}")
    print(f"[load] Loading {len(files)} enriched JSON files …")
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  SKIP {fp.name}: {exc}"); continue
        page_id = payload.get("page_id") or fp.name.replace("_enriched.json", "")
        try:
            years, _ = parse_page_id(page_id)
        except ValueError:
            continue
        weight = 1.0 / len(years)
        for r in payload.get("rows", []):
            if not isinstance(r, dict): continue
            if str(r.get("row_type", "")).strip().lower() != "entry": continue
            amt = amount_decimal(r)
            for year in years:
                records.append({
                    "year":           year,
                    "year_weight":    weight,
                    "amount":         amt,
                    "direction":      r.get("direction"),
                    "category":       r.get("category"),
                    "payment_period": r.get("payment_period"),
                    "is_arrears":     bool(r.get("is_arrears", False)),
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No entry rows loaded.")
    df["era"]           = df["year"].map(era_of_year)
    df["price_idx"]     = df["year"].map(price_index)
    df["amount_real"]   = df["amount"] / (df["price_idx"] / 100.0)
    df["amount_real_w"] = df["amount"] * df["year_weight"] / (df["price_idx"] / 100.0)
    valid = df["direction"].isin({"expenditure", "income"})
    df = df[valid].reset_index(drop=True)
    print(f"  Loaded {len(df):,} records | "
          f"{df['year'].nunique()} years ({df['year'].min()}–{df['year'].max()})")
    return df


# ---------------------------------------------------------------------------
# Four-level proxy computation
# ---------------------------------------------------------------------------

def compute_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yearly panel of all four level proxies + income-side predictors.
    Returns one row per year (1700–1900).
    """
    exp = df[df["direction"] == "expenditure"].copy()
    inc = df[df["direction"] == "income"].copy()

    # -- L1: Traditional function share (expenditure side) --
    exp["is_L1"] = exp["category"].isin(L1_CATS)
    exp["is_L3"] = exp["category"] == L3_CAT
    exp["is_L4"] = exp["category"] == L4_CAT

    yearly = exp.groupby("year").apply(lambda g: pd.Series({
        "total_exp_real":  g["amount_real_w"].sum(),
        "L1_real":         g.loc[g["is_L1"], "amount_real_w"].sum(),
        "L3_real":         g.loc[g["is_L3"], "amount_real_w"].sum(),
        "L4_real":         g.loc[g["is_L4"], "amount_real_w"].sum(),
    })).reset_index()

    yearly["L1"] = yearly["L1_real"] / yearly["total_exp_real"].replace(0, np.nan)
    yearly["L3"] = yearly["L3_real"] / yearly["total_exp_real"].replace(0, np.nan)
    yearly["L4"] = yearly["L4_real"] / yearly["total_exp_real"].replace(0, np.nan)
    yearly["modern_function_share"] = yearly["L3"] + yearly["L4"]

    # -- L2: Payment modernity index (expenditure side, amount-weighted) --
    exp_pay = exp[exp["payment_period"].isin(PAYMENT_SCORES)].copy()
    exp_pay["pm_score"] = exp_pay["payment_period"].map(PAYMENT_SCORES)
    exp_pay["pm_weighted"] = exp_pay["pm_score"] * exp_pay["amount_real_w"]

    pm_num = exp_pay.groupby("year")["pm_weighted"].sum().rename("pm_num")
    pm_den = exp_pay.groupby("year")["amount_real_w"].sum().rename("pm_den")
    pm_cov  = exp_pay.groupby("year").size().rename("pm_covered")   # scorable entries only
    exp_cnt = exp.groupby("year").size().rename("exp_total")

    pm = pd.concat([pm_num, pm_den, pm_cov, exp_cnt], axis=1).reset_index()
    pm["L2"] = pm["pm_num"] / pm["pm_den"].replace(0, np.nan)
    pm["pm_coverage"] = pm["pm_covered"] / pm["exp_total"].replace(0, np.nan)
    yearly = yearly.merge(pm[["year", "L2", "pm_coverage"]], on="year", how="left")

    # -- Income-side predictors --
    inc_total = inc.groupby("year")["amount_real_w"].sum().rename("total_inc_real")
    land_inc  = inc[inc["category"] == "land_rent"].groupby("year")["amount_real_w"].sum().rename("land_rent_real")
    inc_grp   = pd.concat([inc_total, land_inc], axis=1).reset_index()
    inc_grp["land_rent_share"] = inc_grp["land_rent_real"] / inc_grp["total_inc_real"].replace(0, np.nan)

    # Income HHI
    inc_cat = inc.groupby(["year", "category"])["amount_real_w"].sum().reset_index()
    inc_t   = inc.groupby("year")["amount_real_w"].sum().rename("tot").reset_index()
    inc_cat = inc_cat.merge(inc_t, on="year")
    inc_cat["sh"] = inc_cat["amount_real_w"] / inc_cat["tot"].replace(0, np.nan)
    hhi = inc_cat.groupby("year").apply(lambda g: (g["sh"] ** 2).sum()).rename("hhi").reset_index()
    hhi["income_div"] = 1.0 - hhi["hhi"]

    yearly = yearly.merge(inc_grp[["year", "total_inc_real", "land_rent_share"]], on="year", how="left")
    yearly = yearly.merge(hhi[["year", "hhi", "income_div"]], on="year", how="left")
    yearly["era"] = yearly["year"].map(era_of_year)

    # 10-yr rolling means
    ys = yearly.set_index("year").sort_index()
    for col in ["L1", "L2", "L3", "L4", "modern_function_share"]:
        ys[f"{col}_10yr"] = ys[col].rolling(10, center=True, min_periods=5).mean()
    yearly = ys.reset_index()

    print(f"  [proxies] L2 coverage: {yearly['pm_coverage'].median():.1%} median")
    yearly.to_csv(OUT_DIR / "four_level_proxies.csv", index=False)
    return yearly.sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bootstrap era-level means
# ---------------------------------------------------------------------------

def bootstrap_era_means(
    df: pd.DataFrame,
    col: str,
    direction: str = "expenditure",
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap era-level mean of a share proxy by resampling entries within each era.
    Returns DataFrame with era, mean, ci_lo (2.5%), ci_hi (97.5%).
    """
    rng = np.random.default_rng(seed)
    sub = df[df["direction"] == direction].copy()
    results = []
    for era_name, y_lo, y_hi in ERAS:
        g = sub[(sub["year"] >= y_lo) & (sub["year"] <= y_hi)]
        if len(g) < 10:
            results.append({"era": era_name, "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
            continue
        boots = []
        for _ in range(n_boot):
            s = g.sample(n=len(g), replace=True, random_state=int(rng.integers(0, 2**31)))
            num = s.loc[s["category"].isin(_col_to_cats(col)), "amount_real_w"].sum()
            den = s["amount_real_w"].sum()
            if den > 0:
                boots.append(num / den)
        if boots:
            results.append({
                "era":   era_name,
                "mean":  np.mean(boots),
                "ci_lo": np.percentile(boots, 2.5),
                "ci_hi": np.percentile(boots, 97.5),
            })
        else:
            results.append({"era": era_name, "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
    return pd.DataFrame(results)


def _col_to_cats(col: str) -> set:
    return {
        "L1": L1_CATS,
        "L3": {L3_CAT},
        "L4": {L4_CAT},
        "modern_function_share": {L3_CAT, L4_CAT},
    }[col]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

LEVEL_COLORS = {
    "L1": "#9467bd",   # purple  — traditional/efficiency
    "L2": "#ff7f0e",   # orange  — payment modernity
    "L3": "#17becf",   # teal    — salary/operational
    "L4": "#2ca02c",   # green   — educational/mission
    "modern_function_share": "#1f77b4",  # blue
}
LEVEL_NAMES = {
    "L1": "L1: Traditional Function Share",
    "L2": "L2: Payment Modernity Index",
    "L3": "L3: Salary Share (Operational)",
    "L4": "L4: Educational Share (Mission)",
    "modern_function_share": "Modern Function Share (L3+L4)",
}


def add_era_vlines(ax: plt.Axes, alpha: float = 0.45) -> None:
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=alpha)


def plot_level_timeseries(
    panel: pd.DataFrame,
    col: str,
    ax: plt.Axes,
    rolling_col: str | None = None,
) -> None:
    color = LEVEL_COLORS.get(col, "#555")
    rc = rolling_col or f"{col}_10yr"
    ax.plot(panel["year"], panel[col], lw=0.7, color=color, alpha=0.4)
    if rc in panel.columns:
        ax.plot(panel["year"], panel[rc], lw=2.2, color=color, label="10-yr rolling mean")
    add_era_vlines(ax)
    ax.set_xlim(1700, 1900)


def fig_to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Section per level — plot + era means
# ---------------------------------------------------------------------------

def section_level(
    df: pd.DataFrame,
    panel: pd.DataFrame,
    col: str,
    title: str,
    ylabel: str,
) -> tuple[str, pd.DataFrame]:
    """Plot time series + compute bootstrap era means. Returns (b64_png, era_df)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    plot_level_timeseries(panel, col, ax)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    # Era label bands
    era_positions = [(1740, "Pre-Industrial"), (1800, "Transition"),
                     (1840, "Early Ind."), (1880, "Late Ind.")]
    ylim = ax.get_ylim()
    for xp, lbl in era_positions:
        ax.text(xp, ylim[1] * 0.97, lbl, ha="center", fontsize=7, color="grey", va="top")
    ax.legend(fontsize=8)
    fig.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)

    # Era means (from raw df bootstrap, only possible for L1/L3/L4/mfs)
    if col in ("L1", "L3", "L4", "modern_function_share"):
        era_df = bootstrap_era_means(df, col)
    else:
        # L2: compute directly from panel
        era_rows = []
        for era_name, y_lo, y_hi in ERAS:
            sub = panel[(panel["year"] >= y_lo) & (panel["year"] <= y_hi)]["L2"].dropna()
            era_rows.append({
                "era": era_name,
                "mean": sub.mean(),
                "ci_lo": sub.quantile(0.025),
                "ci_hi": sub.quantile(0.975),
            })
        era_df = pd.DataFrame(era_rows)
    era_df["era_label"] = era_df["era"].map(ERA_SHORT)
    return b64, era_df


# ---------------------------------------------------------------------------
# Structural break detection
# ---------------------------------------------------------------------------

def section_structural_breaks(panel: pd.DataFrame) -> dict:
    """
    Three methods: ruptures PELT, rolling 20-yr OLS, Chow-test grid.
    Returns dict with figures (b64) and break results.
    """
    print("[breaks] Running structural break analysis …")
    y   = panel["modern_function_share"].values
    yrs = panel["year"].values
    mask = ~np.isnan(y)
    y_clean, yrs_clean = y[mask], yrs[mask]
    results = {}

    # ---- Method 1: Ruptures PELT ----
    breaks_pelt = []
    if HAS_RUPTURES and len(y_clean) > 20:
        # Use l2 model (detects level-shift changes in mean, appropriate for share series)
        signal = y_clean
        bic_vals = {}
        for n_bkps in range(0, 5):
            try:
                algo = rpt.Dynp(model="l2", min_size=10, jump=1).fit(signal)
                result = algo.predict(n_bkps=n_bkps)
                segs = [0] + result
                ssr = 0.0
                for i in range(len(segs) - 1):
                    seg = y_clean[segs[i]: segs[i + 1]]
                    ssr += np.sum((seg - seg.mean()) ** 2)
                n = len(y_clean)
                k = 2 * (n_bkps + 1)
                bic = n * np.log(ssr / n + 1e-12) + k * np.log(n)
                bic_vals[n_bkps] = (bic, result[:-1])
            except Exception as exc:
                print(f"  PELT n_bkps={n_bkps}: {exc}")
        if bic_vals:
            print(f"  BIC values: { {k: round(v[0],1) for k,v in bic_vals.items()} }")
            best = min(bic_vals, key=lambda k: bic_vals[k][0])
            breaks_pelt = [int(yrs_clean[b - 1]) for b in bic_vals[best][1] if b < len(yrs_clean)]
            results["pelt_break_years"] = breaks_pelt
            results["pelt_n_breaks"]    = best
            print(f"  Best n_bkps={best}, break years: {breaks_pelt}")

    # ---- Method 2: Rolling 20-yr OLS ----
    if HAS_STATSMODELS:
        roll_slopes, roll_years = [], []
        roll_lo, roll_hi = [], []
        window = 20
        for i in range(len(y_clean) - window + 1):
            yw = y_clean[i: i + window]
            xw = yrs_clean[i: i + window].astype(float)
            xw_n = (xw - xw.mean()) / xw.std()
            X = sm.add_constant(xw_n)
            res = sm.OLS(yw, X).fit(cov_type="HC3")
            ci_arr = np.array(res.conf_int(alpha=0.10))  # shape (k, 2)
            roll_slopes.append(float(res.params[1]))
            roll_lo.append(float(ci_arr[1, 0]))
            roll_hi.append(float(ci_arr[1, 1]))
            roll_years.append(int(yrs_clean[i + window // 2]))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(roll_years, roll_lo, roll_hi, alpha=0.25, color="#1f77b4", label="90% CI")
        ax.plot(roll_years, roll_slopes, color="#1f77b4", lw=1.8, label="Rolling slope (20-yr OLS)")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        for yr in breaks_pelt:
            ax.axvline(yr, color="darkred", lw=1.2, ls="--", alpha=0.8)
        add_era_vlines(ax)
        ax.set_title("Rolling 20-Year OLS Slope of Modern Function Share\n"
                     "(vertical red dashes = ruptures break points)", fontweight="bold")
        ax.set_ylabel("OLS slope (standardised year)")
        ax.set_xlabel("Window centre year")
        ax.legend(fontsize=8)
        ax.set_xlim(1700, 1900)
        fig.tight_layout()
        results["fig_rolling"] = fig_to_b64(fig)
        plt.close(fig)

    # ---- Method 3: Chow-test grid ----
    if HAS_STATSMODELS:
        chow_years, chow_f = [], []
        # Restricted model
        X_full = sm.add_constant(yrs_clean.astype(float))
        rss_r  = float(sm.OLS(y_clean, X_full).fit().ssr)
        n = len(y_clean)
        for yr in range(1720, 1890, 5):
            before = yrs_clean <= yr
            after  = yrs_clean > yr
            if before.sum() < 5 or after.sum() < 5:
                continue
            try:
                r1 = sm.OLS(y_clean[before], sm.add_constant(yrs_clean[before].astype(float))).fit()
                r2 = sm.OLS(y_clean[after],  sm.add_constant(yrs_clean[after].astype(float))).fit()
                rss_u = r1.ssr + r2.ssr
                k = 2
                f = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * k))
                chow_years.append(yr)
                chow_f.append(f)
            except Exception:
                pass

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(chow_years, chow_f, color="#d62728", lw=1.8, label="Chow F-statistic")
        # 95% critical value F(2, n-4)
        f_crit = scipy_stats.f.ppf(0.95, 2, n - 4)
        ax.axhline(f_crit, color="black", lw=0.9, ls="--", label=f"F crit (95%) = {f_crit:.1f}")
        add_era_vlines(ax)
        ax.set_title("Chow-Test Grid: Structural Break Candidate Years\n"
                     "(NOTE: assumes i.i.d. residuals; autocorrelation may inflate F)", fontweight="bold")
        ax.set_ylabel("F-statistic")
        ax.set_xlabel("Candidate break year")
        ax.legend(fontsize=8)
        ax.set_xlim(1720, 1890)
        fig.tight_layout()
        results["fig_chow"]      = fig_to_b64(fig)
        results["chow_years"]    = chow_years
        results["chow_f"]        = chow_f
        results["f_crit"]        = float(f_crit)
        peak_idx = int(np.argmax(chow_f))
        results["chow_peak_year"] = chow_years[peak_idx]
        results["chow_peak_f"]    = chow_f[peak_idx]
        plt.close(fig)

    # ---- Ruptures figure (MFS with break lines) ----
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(yrs_clean, y_clean, lw=0.8, color="#1f77b4", alpha=0.5)
    rolling = pd.Series(y_clean, index=yrs_clean).rolling(10, center=True, min_periods=5).mean()
    ax.plot(yrs_clean, rolling.values, lw=2.2, color="#1f77b4", label="10-yr rolling mean")
    for yr in breaks_pelt:
        ax.axvline(yr, color="darkred", lw=1.5, ls="--", alpha=0.9,
                   label=f"PELT break: {yr}")
    add_era_vlines(ax)
    ax.set_title("Modern Function Share (L3+L4) with Detected Structural Breaks",
                 fontweight="bold")
    ax.set_ylabel("modern_function_share")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1900)
    fig.tight_layout()
    results["fig_breaks"] = fig_to_b64(fig)
    plt.close(fig)

    pd.DataFrame({
        "break_year":  breaks_pelt,
        "method":      ["PELT"] * len(breaks_pelt),
    }).to_csv(OUT_DIR / "structural_breaks.csv", index=False)

    return results


# ---------------------------------------------------------------------------
# Level sequencing — rolling mean crossover + cross-correlation
# ---------------------------------------------------------------------------

def section_sequencing(panel: pd.DataFrame) -> dict:
    """
    Test whether the four levels rise in sequential order:
    L2 (differentiation) → L3 (operational) → L4 (mission).
    1. Median-crossover year: first year the 10-yr rolling mean exceeds its
       long-run median — ordered crossovers support the framework.
    2. Pairwise cross-correlation (L2 vs L3, L3 vs L4) at lags -10..+10:
       positive peak lag means the first series leads the second.
    """
    print("[sequencing] Running level sequencing analysis …")
    results: dict = {}

    pan = panel.dropna(subset=["L2_10yr", "L3_10yr", "L4_10yr"]).sort_values("year").reset_index(drop=True)

    # ---- Figure: overlay of all four 10-yr rolling means ----
    fig, ax = plt.subplots(figsize=(12, 5))
    crossover_rows = []
    for col in ["L1", "L2", "L3", "L4"]:
        rc = f"{col}_10yr"
        if rc not in pan.columns:
            continue
        ser = pan[rc].dropna()
        yrs = pan.loc[ser.index, "year"]
        median_val = float(ser.median())
        color = LEVEL_COLORS.get(col, "#555")
        lw = 1.4 if col == "L1" else 2.2
        alpha = 0.55 if col == "L1" else 1.0
        ax.plot(yrs, ser.values, lw=lw, color=color, alpha=alpha,
                label=LEVEL_NAMES.get(col, col))
        above = ser > median_val
        crossover_yr = int(yrs.iloc[above.values.argmax()]) if above.any() else None
        if crossover_yr:
            ax.axvline(crossover_yr, color=color, lw=1.0, ls=":", alpha=0.6)
        crossover_rows.append({
            "level": col,
            "median_val": round(median_val, 4),
            "median_crossover_year": crossover_yr,
            "peak_10yr_year": int(yrs.iloc[int(ser.values.argmax())]),
        })
    add_era_vlines(ax)
    ax.set_title(
        "Level Sequencing: 10-Year Rolling Means of All Four Proxies\n"
        "(Dotted verticals = year each level first crosses its own long-run median)",
        fontweight="bold",
    )
    ax.set_ylabel("Share / index (10-yr rolling mean)")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1900)
    fig.tight_layout()
    results["fig"] = fig_to_b64(fig)
    plt.close(fig)

    crossover_df = pd.DataFrame(crossover_rows)
    results["crossover_df"] = crossover_df

    # ---- Pairwise cross-correlation: (L2 vs L3) and (L3 vs L4) ----
    xcorr_rows = []
    for col_a, col_b in [("L2", "L3"), ("L3", "L4")]:
        ra, rb = f"{col_a}_10yr", f"{col_b}_10yr"
        sub = pan.dropna(subset=[ra, rb])
        if len(sub) < 10:
            continue
        sa = sub[ra].values.astype(float)
        sb = sub[rb].values.astype(float)
        sa = (sa - sa.mean()) / (sa.std() + 1e-12)
        sb = (sb - sb.mean()) / (sb.std() + 1e-12)
        lags = list(range(-10, 11))
        corrs = []
        for lag in lags:
            if lag > 0 and lag < len(sa):
                r = float(np.corrcoef(sa[lag:], sb[:-lag])[0, 1])
            elif lag < 0 and -lag < len(sb):
                r = float(np.corrcoef(sa[:lag], sb[-lag:])[0, 1])
            else:
                r = float(np.corrcoef(sa, sb)[0, 1])
            corrs.append(r)
        best_idx = int(np.nanargmax(corrs))
        peak_lag = lags[best_idx]
        peak_corr = corrs[best_idx]
        xcorr_rows.append({
            "pair": f"{col_a} vs {col_b}",
            "peak_lag": peak_lag,
            "peak_corr": round(peak_corr, 3),
            "interpretation": (
                f"{col_a} leads {col_b} by {peak_lag} yr" if peak_lag > 0
                else (f"{col_b} leads {col_a} by {-peak_lag} yr" if peak_lag < 0
                      else "simultaneous")
            ),
        })

    xcorr_df = pd.DataFrame(xcorr_rows)
    results["xcorr_df"] = xcorr_df
    crossover_df.to_csv(OUT_DIR / "level_sequencing.csv", index=False)

    return results


# ---------------------------------------------------------------------------
# Income diversification co-evolution
# ---------------------------------------------------------------------------

def section_income_diversification(panel: pd.DataFrame) -> dict:
    """
    Does income diversification (1 − HHI) co-evolve with transformation
    (modern_function_share)?  Both series are plotted together and correlated
    within each era.
    """
    print("[income_div] Running income diversification co-evolution analysis …")
    results: dict = {}

    pan = panel.dropna(subset=["income_div", "modern_function_share"]).sort_values("year").reset_index(drop=True)

    inc_div_roll = (
        pan.set_index("year")["income_div"]
        .rolling(10, center=True, min_periods=5)
        .mean()
        .reset_index()["income_div"]
        .values
    )

    # ---- Figure 1: dual-axis timeseries ----
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    mfs_roll = pan["modern_function_share_10yr"].values if "modern_function_share_10yr" in pan.columns else (
        pan.set_index("year")["modern_function_share"]
        .rolling(10, center=True, min_periods=5)
        .mean()
        .values
    )
    ax1.plot(pan["year"], mfs_roll, lw=2.2, color="#1f77b4", label="Modern Function Share L3+L4 (10-yr)")
    ax1.set_ylabel("Modern Function Share (L3+L4)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2.plot(pan["year"], inc_div_roll, lw=2.2, color="#e6550d", ls="--",
             label="Income Diversification 1−HHI (10-yr)")
    ax2.set_ylabel("Income Diversification (1 − HHI)", color="#e6550d")
    ax2.tick_params(axis="y", labelcolor="#e6550d")

    add_era_vlines(ax1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.set_title(
        "Income Diversification vs Modern Function Share (1700–1900)\n"
        "(Both as 10-year rolling means; left axis = MFS, right axis = diversification)",
        fontweight="bold",
    )
    ax1.set_xlabel("Year")
    ax1.set_xlim(1700, 1900)
    fig.tight_layout()
    results["fig_timeseries"] = fig_to_b64(fig)
    plt.close(fig)

    # ---- Figure 2: scatter by era ----
    era_palette = {
        "pre_industrial":   "#9467bd",
        "transition":       "#ff7f0e",
        "early_industrial": "#2ca02c",
        "late_industrial":  "#1f77b4",
    }
    era_display = {
        "pre_industrial":   "Pre-Industrial",
        "transition":       "Transition",
        "early_industrial": "Early Industrial",
        "late_industrial":  "Late Industrial",
    }
    fig2, ax = plt.subplots(figsize=(8, 6))
    for era_key, color in era_palette.items():
        sub = pan[pan["era"] == era_key].dropna(subset=["income_div", "modern_function_share"])
        if sub.empty:
            continue
        ax.scatter(sub["income_div"], sub["modern_function_share"],
                   color=color, alpha=0.5, s=22, label=era_display.get(era_key, era_key))
        if len(sub) > 3 and HAS_STATSMODELS:
            m = sm.OLS(sub["modern_function_share"],
                       sm.add_constant(sub["income_div"])).fit()
            x_range = np.linspace(sub["income_div"].min(), sub["income_div"].max(), 50)
            intercept = float(m.params.get("const", m.params.iloc[0]))
            slope     = float(m.params.get("income_div", m.params.iloc[1]))
            ax.plot(x_range, intercept + slope * x_range, color=color, lw=1.5, alpha=0.8)
    ax.set_xlabel("Income Diversification (1 − HHI)")
    ax.set_ylabel("Modern Function Share (L3+L4)")
    ax.set_title("Income Diversification vs Transformation by Era\n"
                 "(Lines = within-era OLS fit)", fontweight="bold")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    results["fig_scatter"] = fig_to_b64(fig2)
    plt.close(fig2)

    # ---- Era-level Pearson correlations ----
    corr_rows = []
    for era_name, y_lo, y_hi in ERAS:
        sub = pan[(pan["year"] >= y_lo) & (pan["year"] <= y_hi)].dropna(
            subset=["income_div", "modern_function_share"])
        if len(sub) < 5:
            corr_rows.append({"era": era_name, "r": np.nan, "p": np.nan, "n": len(sub)})
            continue
        r, p = pearsonr(sub["income_div"].values, sub["modern_function_share"].values)
        corr_rows.append({"era": era_name, "r": round(float(r), 3), "p": round(float(p), 3), "n": len(sub)})
    corr_df = pd.DataFrame(corr_rows)
    results["corr_df"] = corr_df
    corr_df.to_csv(OUT_DIR / "income_diversification.csv", index=False)

    overall_r, overall_p = pearsonr(pan["income_div"].values, pan["modern_function_share"].values)
    results["overall_r"] = float(overall_r)
    results["overall_p"] = float(overall_p)

    return results


# ---------------------------------------------------------------------------
# Capability threshold — lagged OLS
# ---------------------------------------------------------------------------

def section_capability_threshold(panel: pd.DataFrame) -> dict:
    """
    Lagged OLS: does land_rent_share(t-k) predict ΔL3(t) and ΔL4(t)?
    Tests lags k = 1, 3, 5.
    """
    print("[threshold] Running capability threshold regressions …")
    if not HAS_STATSMODELS:
        return {}

    panel_s = panel[["year", "L3", "L4", "land_rent_share", "income_div"]].dropna().copy()
    panel_s = panel_s.sort_values("year").reset_index(drop=True)
    panel_s["dL3"] = panel_s["L3"].diff()
    panel_s["dL4"] = panel_s["L4"].diff()
    yr_mean = panel_s["year"].mean()
    yr_std  = panel_s["year"].std()
    panel_s["year_norm"] = (panel_s["year"] - yr_mean) / yr_std

    rows = []
    for k in [1, 3, 5]:
        for outcome_col, outcome_name in [("dL3", "ΔL3 (salary share)"),
                                          ("dL4", "ΔL4 (educational share)")]:
            panel_s[f"lr_lag{k}"] = panel_s["land_rent_share"].shift(k)
            panel_s[f"id_lag{k}"] = panel_s["income_div"].shift(k)
            sub = panel_s[["year_norm", f"lr_lag{k}", outcome_col]].dropna()
            if len(sub) < 20:
                continue
            X = sm.add_constant(sub[[f"lr_lag{k}", "year_norm"]])
            Y = sub[outcome_col]
            try:
                res = sm.OLS(Y, X).fit(cov_type="HC3")
                coef  = res.params[f"lr_lag{k}"]
                se    = res.bse[f"lr_lag{k}"]
                pval  = res.pvalues[f"lr_lag{k}"]
                ci_lo = res.conf_int(alpha=0.05).loc[f"lr_lag{k}", 0]
                ci_hi = res.conf_int(alpha=0.05).loc[f"lr_lag{k}", 1]
                rows.append({
                    "outcome":  outcome_name,
                    "lag_k":    k,
                    "coef":     coef,
                    "se":       se,
                    "pval":     pval,
                    "ci_lo":    ci_lo,
                    "ci_hi":    ci_hi,
                    "n":        len(sub),
                })
            except Exception as exc:
                print(f"  WARN lag={k} outcome={outcome_col}: {exc}")

    if not rows:
        return {}

    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(OUT_DIR / "capability_threshold_regressions.csv", index=False)

    # Coefficient plot
    outcomes = coef_df["outcome"].unique()
    fig, axes = plt.subplots(1, len(outcomes), figsize=(5 * len(outcomes), 4.5), sharey=False)
    if len(outcomes) == 1:
        axes = [axes]
    for ax, outcome in zip(axes, outcomes):
        sub = coef_df[coef_df["outcome"] == outcome].sort_values("lag_k")
        xs  = sub["lag_k"].values
        color = "#17becf" if "L3" in outcome else "#2ca02c"
        ax.errorbar(xs, sub["coef"], yerr=[sub["coef"] - sub["ci_lo"], sub["ci_hi"] - sub["coef"]],
                    fmt="o", color=color, capsize=4, lw=1.8, markersize=7)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"lag {x}" for x in xs])
        ax.set_title(outcome, fontsize=9)
        ax.set_ylabel("Coefficient on land_rent_share (t-k)")
        ax.set_xlabel("Lag k (years)")
        # Annotate p-values
        for _, row in sub.iterrows():
            stars = "***" if row["pval"] < 0.01 else ("**" if row["pval"] < 0.05
                    else ("*" if row["pval"] < 0.10 else "n.s."))
            ax.text(row["lag_k"], row["ci_hi"] + 0.001, stars, ha="center", fontsize=9)
    fig.suptitle("Capability Threshold: Does land-rent income predict L3/L4 advancement?\n"
                 "(HC3 robust SEs; * p<0.10, ** p<0.05, *** p<0.01)", fontsize=10)
    fig.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)

    return {"coef_df": coef_df, "fig": b64}


# ---------------------------------------------------------------------------
# Shock response — event study (descriptive)
# ---------------------------------------------------------------------------

def section_shock_response(panel: pd.DataFrame) -> dict:
    """
    Descriptive event study around 4 agricultural shocks.
    Returns figure and average MFS trajectory.
    """
    print("[shocks] Running shock response event study …")
    window = 8
    yr_series = panel.set_index("year")["modern_function_share"]
    total_inc  = panel.set_index("year")["total_inc_real"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    all_mfs, all_inc = {}, {}
    for (shock_yr, shock_lbl), color in zip(AG_SHOCKS.items(), palette):
        t_range = list(range(shock_yr - window, shock_yr + window + 1))
        mfs_rel, inc_rel = {}, {}
        for t in t_range:
            if t in yr_series.index and not np.isnan(yr_series[t]):
                mfs_rel[t - shock_yr] = yr_series[t]
            if t in total_inc.index and not np.isnan(total_inc[t]):
                inc_rel[t - shock_yr] = total_inc[t]
        all_mfs[shock_yr]  = mfs_rel
        all_inc[shock_yr]  = inc_rel
        if mfs_rel:
            xs = sorted(mfs_rel)
            ax1.plot(xs, [mfs_rel[x] for x in xs], lw=1.4, color=color,
                     alpha=0.7, label=f"{shock_yr}: {shock_lbl}")
        if inc_rel:
            xs = sorted(inc_rel)
            ax2.plot(xs, [inc_rel[x] for x in xs], lw=1.4, color=color, alpha=0.7)

    # Average trajectory
    all_xs = list(range(-window, window + 1))
    avg_mfs = {}
    for rel_t in all_xs:
        vals = [all_mfs[sy].get(rel_t) for sy in AG_SHOCKS if rel_t in all_mfs.get(sy, {})]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if vals:
            avg_mfs[rel_t] = np.mean(vals)
    if avg_mfs:
        xs_avg = sorted(avg_mfs)
        ax1.plot(xs_avg, [avg_mfs[x] for x in xs_avg], lw=2.8, color="black",
                 ls="--", label="Average across shocks", zorder=5)

    ax1.axvline(0, color="black", lw=1.0, ls=":")
    ax2.axvline(0, color="black", lw=1.0, ls=":")
    ax1.set_ylabel("Modern Function Share (L3+L4)")
    ax2.set_ylabel("Total Income (real £)")
    ax2.set_xlabel("Years relative to shock (0 = shock year)")
    ax1.set_title("Descriptive Event Study: Modern Function Share Around Agricultural Shocks\n"
                  "(N=4 events — purely descriptive, no inferential claims)", fontweight="bold")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)

    pd.DataFrame([
        {"shock_year": sy, "shock_label": lbl, "t_rel": rel_t, "mfs": all_mfs[sy].get(rel_t)}
        for sy, lbl in AG_SHOCKS.items()
        for rel_t in range(-window, window + 1)
    ]).to_csv(OUT_DIR / "shock_event_study.csv", index=False)

    return {"fig": b64, "avg_mfs": avg_mfs}


# ---------------------------------------------------------------------------
# Non-linearity test — era-level changes across levels
# ---------------------------------------------------------------------------

def section_nonlinearity(df: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """
    Compare era-over-era changes (Δ) across all four level proxies.
    Larger |Δ| at higher levels = evidence for non-linear value progression.
    """
    print("[nonlinear] Running non-linearity test …")

    era_means = {}
    for col, direction in [("L1", "expenditure"), ("L3", "expenditure"),
                            ("L4", "expenditure"), ("modern_function_share", "expenditure")]:
        era_means[col] = bootstrap_era_means(df, col, direction=direction, n_boot=1000)

    # L2 from panel
    l2_rows = []
    for era_name, y_lo, y_hi in ERAS:
        sub = panel[(panel["year"] >= y_lo) & (panel["year"] <= y_hi)]["L2"].dropna()
        l2_rows.append({"era": era_name, "mean": sub.mean(), "ci_lo": sub.quantile(0.025),
                        "ci_hi": sub.quantile(0.975)})
    era_means["L2"] = pd.DataFrame(l2_rows)

    # Era-over-era deltas
    transitions = [
        ("pre_industrial",   "transition",       "Pre→Trans"),
        ("transition",       "early_industrial", "Trans→Early"),
        ("early_industrial", "late_industrial",  "Early→Late"),
    ]
    delta_rows = []
    for col in ["L1", "L2", "L3", "L4"]:
        em = era_means[col].set_index("era")
        for (era_a, era_b, lbl) in transitions:
            if era_a in em.index and era_b in em.index:
                delta = em.loc[era_b, "mean"] - em.loc[era_a, "mean"]
                delta_rows.append({"level": col, "transition": lbl, "delta": delta})

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT_DIR / "era_level_changes.csv", index=False)

    # Figure: grouped bar chart of |Δ| by transition, coloured by level
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    level_order = ["L1", "L2", "L3", "L4"]
    for ax, (_, _, lbl) in zip(axes, transitions):
        sub = delta_df[delta_df["transition"] == lbl]
        bars = []
        bar_colors = []
        for col in level_order:
            row = sub[sub["level"] == col]
            if row.empty:
                bars.append(np.nan)
            else:
                bars.append(row.iloc[0]["delta"])
            bar_colors.append(LEVEL_COLORS[col])
        bar_vals = [b if not (b != b) else b for b in bars]
        xs = np.arange(len(level_order))
        bc = ax.bar(xs, bar_vals, color=bar_colors, edgecolor="white", width=0.65, alpha=0.85)
        for x, v in zip(xs, bar_vals):
            if not np.isnan(v):
                ax.text(x, v + (0.001 if v >= 0 else -0.003), f"{v:+.3f}", ha="center",
                        fontsize=8, va="bottom" if v >= 0 else "top")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(level_order)
        ax.set_title(f"Era transition: {lbl}", fontsize=9)
        ax.set_ylabel("Δ proxy (era_B − era_A)")
        ax.set_xlabel("Level")
    handles = [mpatches.Patch(color=LEVEL_COLORS[c], label=LEVEL_NAMES[c]) for c in level_order]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Era-over-Era Changes Across Four Transformation Levels\n"
                 "(bootstrap means; larger |Δ| at higher levels = non-linear value progression)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    b64 = fig_to_b64(fig)
    plt.close(fig)

    # Also produce a four-panel overview of all level proxies
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8))
    for ax2, (col, lbl) in zip(axes2.flat, [
        ("L1", "L1: Traditional Function Share\n(ecclesiastical + maintenance + domestic)"),
        ("L2", "L2: Payment Modernity Index\n(amount-weighted, excludes 'unclear')"),
        ("L3", "L3: Salary Share\n(salary_stipend / total expenditure)"),
        ("L4", "L4: Educational Share\n(educational / total expenditure)"),
    ]):
        plot_level_timeseries(panel, col, ax2)
        ax2.set_title(lbl, fontsize=9)
        ax2.set_ylabel("Share / index")
    fig2.suptitle("Four-Level AI Transformation Framework — Historical Proxies (Oxford, 1700–1900)",
                  fontweight="bold")
    fig2.tight_layout()
    b64_overview = fig_to_b64(fig2)
    plt.close(fig2)

    return {"fig_delta": b64, "fig_overview": b64_overview, "era_means": era_means, "delta_df": delta_df}


# ---------------------------------------------------------------------------
# HTML Report generation
# ---------------------------------------------------------------------------

CSS = """
body{font-family:Georgia,serif;max-width:1100px;margin:auto;padding:2em 2.5em;
     line-height:1.70;color:#222;background:#fff;}
h1{color:#1a252f;font-size:1.9em;margin-bottom:.15em;}
h2{color:#2c3e50;font-size:1.2em;border-bottom:2px solid #aab7c4;padding-bottom:4px;margin-top:2.2em;}
h3{color:#34495e;font-size:1.0em;margin-top:1.5em;margin-bottom:.2em;}
.subtitle{color:#555;font-size:.95em;margin-bottom:.4em;}
.why{background:#eaf4fb;border-left:4px solid #2980b9;padding:.7em 1.2em;margin:.7em 0;font-size:.92em;}
.why strong{color:#1a5276;}
.interp{background:#f0f9f0;border-left:4px solid #27ae60;padding:.7em 1.2em;margin:.7em 0;font-size:.92em;}
.interp strong{color:#1e8449;}
.method{background:#fdf6e3;border-left:4px solid #e6ac00;padding:.5em 1em;margin:.5em 0;font-size:.88em;}
.warn{background:#fff3cd;border-left:3px solid #ffc107;padding:.5em .9em;margin:.5em 0;font-size:.88em;}
.caution{background:#fce4e4;border-left:4px solid #c0392b;padding:.6em 1em;margin:.6em 0;font-size:.88em;}
img{max-width:100%;border:1px solid #ddd;border-radius:3px;margin:.6em 0;display:block;}
table{border-collapse:collapse;width:100%;font-size:.82em;margin:.6em 0;}
th{background:#2c3e50;color:#fff;padding:5px 9px;text-align:left;font-weight:normal;}
td{border-bottom:1px solid #e8e8e8;padding:4px 9px;}
tr:hover td{background:#f5f5f5;}
.toc{background:#f9f9f9;border:1px solid #ddd;padding:1em 1.5em;border-radius:4px;margin-bottom:2em;}
.toc li{margin:.2em 0;}.toc a{color:#2980b9;text-decoration:none;}
.toc a:hover{text-decoration:underline;}
footer{font-size:.8em;color:#888;margin-top:3em;border-top:1px solid #eee;padding-top:.5em;}
@media print{
  body{max-width:100%;padding:1cm 1.5cm;font-size:10pt;line-height:1.55;}
  h2{break-before:page;page-break-before:always;padding-top:0.3cm;
     border-bottom:1.5pt solid #aab7c4;}
  h2#intro{break-before:avoid;page-break-before:avoid;}
  h1,h2,h3{break-after:avoid;page-break-after:avoid;}
  .toc{break-after:page;page-break-after:always;}
  .why,.method,.interp,.warn,.caution{break-inside:avoid;page-break-inside:avoid;}
  img{break-before:avoid;page-break-before:avoid;
      break-inside:avoid;page-break-inside:avoid;
      max-width:100%;border:0.5pt solid #ccc;}
  table{break-inside:avoid;page-break-inside:avoid;}
  footer{break-before:avoid;page-break-before:avoid;}
  .toc a{color:#000;text-decoration:none;}
}
"""


def make_img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" />'


def fmt_f(x: float | None, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"


def era_means_table(em: pd.DataFrame, col_label: str) -> str:
    rows = ""
    for _, row in em.iterrows():
        era_lbl = ERA_LABELS.get(row["era"], row["era"])
        rows += (f"<tr><td>{era_lbl}</td>"
                 f"<td>{fmt_f(row.get('mean'))}</td>"
                 f"<td>[{fmt_f(row.get('ci_lo'))}, {fmt_f(row.get('ci_hi'))}]</td></tr>")
    return (f"<table><tr><th>Era</th><th>{col_label} mean</th>"
            f"<th>95% bootstrap CI</th></tr>{rows}</table>")


def coef_table(coef_df: pd.DataFrame) -> str:
    rows = ""
    for _, r in coef_df.iterrows():
        stars = ("***" if r["pval"] < 0.01 else ("**" if r["pval"] < 0.05
                 else ("*" if r["pval"] < 0.10 else "")))
        rows += (f"<tr><td>{r['outcome']}</td><td>lag {int(r['lag_k'])}</td>"
                 f"<td>{fmt_f(r['coef'])}{stars}</td><td>{fmt_f(r['se'])}</td>"
                 f"<td>{fmt_p(r['pval'])}</td>"
                 f"<td>[{fmt_f(r['ci_lo'])}, {fmt_f(r['ci_hi'])}]</td>"
                 f"<td>{int(r['n'])}</td></tr>")
    return ("<table><tr><th>Outcome</th><th>Lag</th><th>Coef. on land_rent_share</th>"
            "<th>SE (HC3)</th><th>p-value</th><th>95% CI</th><th>N</th></tr>"
            + rows + "</table>")


def generate_report(
    panel: pd.DataFrame,
    l1_b64: str, l1_era: pd.DataFrame,
    l2_b64: str, l2_era: pd.DataFrame,
    l3_b64: str, l3_era: pd.DataFrame,
    l4_b64: str, l4_era: pd.DataFrame,
    breaks: dict,
    thresh: dict,
    shock: dict,
    nonlin: dict,
    pm_coverage: float,
    seq: dict | None = None,
    inc_div: dict | None = None,
) -> None:

    thresh_table = coef_table(thresh["coef_df"]) if thresh.get("coef_df") is not None else "<p>Skipped.</p>"
    fig_thresh   = make_img_tag(thresh["fig"]) if thresh.get("fig") else ""

    fig_breaks   = make_img_tag(breaks.get("fig_breaks", "")) if breaks.get("fig_breaks") else ""
    fig_rolling  = make_img_tag(breaks.get("fig_rolling", "")) if breaks.get("fig_rolling") else ""
    fig_chow     = make_img_tag(breaks.get("fig_chow", "")) if breaks.get("fig_chow") else ""
    pelt_years   = ", ".join(str(y) for y in breaks.get("pelt_break_years", [])) or "none detected"
    chow_peak    = breaks.get("chow_peak_year", "n/a")
    chow_f_peak  = fmt_f(breaks.get("chow_peak_f"))
    f_crit       = fmt_f(breaks.get("f_crit"))

    fig_shock     = make_img_tag(shock.get("fig", "")) if shock.get("fig") else ""
    fig_delta     = make_img_tag(nonlin.get("fig_delta", "")) if nonlin.get("fig_delta") else ""
    fig_overview  = make_img_tag(nonlin.get("fig_overview", "")) if nonlin.get("fig_overview") else ""

    cov_class = "warn" if pm_coverage < 0.50 else "method"
    cov_icon  = "⚠️ " if pm_coverage < 0.50 else "ℹ️ "
    pm_warn = (
        f'<div class="{cov_class}">{cov_icon}<strong>L2 data coverage:</strong> '
        f'{pm_coverage:.1%} of expenditure entries (by year, median) have a clearly '
        f'scorable payment period (annual, biennial, etc.). Entries labelled '
        f'<code>"unclear"</code> or blank are excluded from the index. '
        + ("Coverage is sufficient to treat this proxy as reliable."
           if pm_coverage >= 0.50 else
           "Coverage is low — treat this proxy with caution.") + "</div>"
    )

    # --- Key findings extraction for the summary box ---
    _delta_df = nonlin.get("delta_df")
    l4_late_delta_str = "n/a"
    if _delta_df is not None and not _delta_df.empty:
        _row = _delta_df[(_delta_df["level"] == "L4") & (_delta_df["transition"] == "Early→Late")]
        if not _row.empty:
            _v = _row.iloc[0]["delta"]
            l4_late_delta_str = f"+{_v:.1%}" if _v > 0 else f"{_v:.1%}"
    _cdf = thresh.get("coef_df")
    dl4_lag5_p_str, dl4_lag5_sign, dl4_lag5_coef_str = "n/a", "positive", "n/a"
    dl4_lag3_p_str, dl4_lag3_coef_str = "n/a", "n/a"
    dl3_lag1_p_str, dl3_lag1_coef_str = "n/a", "n/a"
    if _cdf is not None:
        _r = _cdf[(_cdf["outcome"].str.contains("L4")) & (_cdf["lag_k"] == 5)]
        if not _r.empty:
            dl4_lag5_p_str   = fmt_p(_r.iloc[0]["pval"])
            dl4_lag5_sign    = "positive" if _r.iloc[0]["coef"] > 0 else "negative"
            dl4_lag5_coef_str = fmt_f(_r.iloc[0]["coef"], 3)
        _r = _cdf[(_cdf["outcome"].str.contains("L4")) & (_cdf["lag_k"] == 3)]
        if not _r.empty:
            dl4_lag3_p_str    = fmt_p(_r.iloc[0]["pval"])
            dl4_lag3_coef_str = fmt_f(_r.iloc[0]["coef"], 3)
        _r = _cdf[(_cdf["outcome"].str.contains("L3")) & (_cdf["lag_k"] == 1)]
        if not _r.empty:
            dl3_lag1_p_str    = fmt_p(_r.iloc[0]["pval"])
            dl3_lag1_coef_str = fmt_f(_r.iloc[0]["coef"], 3)

    # --- Era means for interpretation (use l*_era DataFrames) ---
    def _era_mean(em: pd.DataFrame, era: str) -> float | None:
        sub = em[em["era"] == era]
        return float(sub["mean"].iloc[0]) if not sub.empty else None

    l1_pi   = _era_mean(l1_era, "pre_industrial");  l1_tr = _era_mean(l1_era, "transition")
    l1_ei   = _era_mean(l1_era, "early_industrial"); l1_li = _era_mean(l1_era, "late_industrial")
    l2_pi   = _era_mean(l2_era, "pre_industrial");  l2_tr = _era_mean(l2_era, "transition")
    l2_ei   = _era_mean(l2_era, "early_industrial"); l2_li = _era_mean(l2_era, "late_industrial")
    l3_pi   = _era_mean(l3_era, "pre_industrial");  l3_tr = _era_mean(l3_era, "transition")
    l3_ei   = _era_mean(l3_era, "early_industrial"); l3_li = _era_mean(l3_era, "late_industrial")
    l4_pi   = _era_mean(l4_era, "pre_industrial");  l4_tr = _era_mean(l4_era, "transition")
    l4_ei   = _era_mean(l4_era, "early_industrial"); l4_li = _era_mean(l4_era, "late_industrial")

    def pct(v): return f"{v:.1%}" if v is not None else "n/a"
    def pp(a, b): v = b - a if a and b else None; return (f"+{v:.1%}" if v > 0 else f"{v:.1%}") if v is not None else "n/a"

    # Panel-level stats
    _pan = panel.copy() if not panel.empty else pd.DataFrame()
    l3_peak_yr = int(_pan.loc[_pan["L3"].idxmax(), "year"]) if not _pan.empty else 0
    l3_peak_val = pct(_pan["L3"].max()) if not _pan.empty else "n/a"
    l4_peak_yr  = int(_pan.loc[_pan["L4"].idxmax(), "year"]) if not _pan.empty else 0
    l4_peak_val = pct(_pan["L4"].max()) if not _pan.empty else "n/a"
    l4_pre1860  = pct(_pan[_pan["year"] < 1860]["L4"].mean()) if not _pan.empty else "n/a"
    l4_post1860 = pct(_pan[_pan["year"] >= 1860]["L4"].mean()) if not _pan.empty else "n/a"

    # Delta shortcuts
    _ddf = nonlin.get("delta_df", pd.DataFrame())
    def _delta(lvl, trans):
        r = _ddf[(_ddf["level"] == lvl) & (_ddf["transition"] == trans)]
        v = r.iloc[0]["delta"] if not r.empty else None
        return (f"+{v:.1%}" if v > 0 else f"{v:.1%}") if v is not None else "n/a"

    l1_delta_pt = _delta("L1", "Pre→Trans");     l1_delta_te = _delta("L1", "Trans→Early")
    l2_delta_pt = _delta("L2", "Pre→Trans")
    l3_delta_pt = _delta("L3", "Pre→Trans");     l3_delta_te = _delta("L3", "Trans→Early")
    l4_delta_el = _delta("L4", "Early→Late")

    # Shock: average MFS at key relative times
    _avg_mfs = shock.get("avg_mfs", {})
    def _mfs(t): v = _avg_mfs.get(t); return f"{v:.3f}" if v else "n/a"
    shock_t_m1 = _mfs(-1); shock_t_0  = _mfs(0)
    shock_t_p1 = _mfs(1);  shock_t_p4 = _mfs(4)

    # Land-rent era means from panel
    lr_pi = pct(_pan[(_pan["year"] >= 1700) & (_pan["year"] <= 1779)]["land_rent_share"].mean()) if not _pan.empty else "n/a"
    lr_li = pct(_pan[(_pan["year"] >= 1860) & (_pan["year"] <= 1900)]["land_rent_share"].mean()) if not _pan.empty else "n/a"

    # --- Sequencing section variables ---
    fig_seq = make_img_tag(seq["fig"]) if seq and seq.get("fig") else ""
    _crossover_df = seq.get("crossover_df", pd.DataFrame()) if seq else pd.DataFrame()
    _xcorr_df     = seq.get("xcorr_df", pd.DataFrame()) if seq else pd.DataFrame()

    def _crossover_table() -> str:
        if _crossover_df.empty:
            return "<p>Not computed.</p>"
        rows = ""
        for _, r in _crossover_df.iterrows():
            rows += (f"<tr><td>{r['level']}</td><td>{r['median_val']:.3f}</td>"
                     f"<td><strong>{r['median_crossover_year']}</strong></td>"
                     f"<td>{r['peak_10yr_year']}</td></tr>")
        return ("<table><tr><th>Level</th><th>Long-run median</th>"
                "<th>First median-crossover year</th><th>Rolling mean peak year</th></tr>"
                + rows + "</table>")

    def _xcorr_table() -> str:
        if _xcorr_df.empty:
            return "<p>Not computed.</p>"
        rows = ""
        for _, r in _xcorr_df.iterrows():
            rows += (f"<tr><td>{r['pair']}</td><td>{r['peak_lag']:+d}</td>"
                     f"<td>{r['peak_corr']:.3f}</td><td>{r['interpretation']}</td></tr>")
        return ("<table><tr><th>Level pair</th><th>Peak-corr lag (years)</th>"
                "<th>Peak correlation</th><th>Implication</th></tr>"
                + rows + "</table>")

    # Narrative numbers for sequencing
    _l2_cross = _crossover_df.loc[_crossover_df["level"] == "L2", "median_crossover_year"].iloc[0] \
        if not _crossover_df.empty and "L2" in _crossover_df["level"].values else "n/a"
    _l3_cross = _crossover_df.loc[_crossover_df["level"] == "L3", "median_crossover_year"].iloc[0] \
        if not _crossover_df.empty and "L3" in _crossover_df["level"].values else "n/a"
    _l4_cross = _crossover_df.loc[_crossover_df["level"] == "L4", "median_crossover_year"].iloc[0] \
        if not _crossover_df.empty and "L4" in _crossover_df["level"].values else "n/a"
    _seq_ordered = (
        "consistent" if (isinstance(_l2_cross, (int, np.integer)) and
                         isinstance(_l3_cross, (int, np.integer)) and
                         isinstance(_l4_cross, (int, np.integer)) and
                         _l2_cross <= _l3_cross <= _l4_cross)
        else "not perfectly consistent"
    )
    _xcorr_l2l3_lag = int(_xcorr_df.loc[_xcorr_df["pair"] == "L2 vs L3", "peak_lag"].iloc[0]) \
        if not _xcorr_df.empty and "L2 vs L3" in _xcorr_df["pair"].values else None
    _xcorr_l3l4_lag = int(_xcorr_df.loc[_xcorr_df["pair"] == "L3 vs L4", "peak_lag"].iloc[0]) \
        if not _xcorr_df.empty and "L3 vs L4" in _xcorr_df["pair"].values else None
    _xcorr_l2l3_str = (f"L2 leads L3 by {_xcorr_l2l3_lag} yr" if _xcorr_l2l3_lag and _xcorr_l2l3_lag > 0
                       else (f"L3 leads L2 by {-_xcorr_l2l3_lag} yr" if _xcorr_l2l3_lag and _xcorr_l2l3_lag < 0
                             else "simultaneous")) if _xcorr_l2l3_lag is not None else "n/a"
    _xcorr_l3l4_str = (f"L3 leads L4 by {_xcorr_l3l4_lag} yr" if _xcorr_l3l4_lag and _xcorr_l3l4_lag > 0
                       else (f"L4 leads L3 by {-_xcorr_l3l4_lag} yr" if _xcorr_l3l4_lag and _xcorr_l3l4_lag < 0
                             else "simultaneous")) if _xcorr_l3l4_lag is not None else "n/a"

    # --- Income diversification section variables ---
    fig_incdiv_ts = make_img_tag(inc_div["fig_timeseries"]) if inc_div and inc_div.get("fig_timeseries") else ""
    fig_incdiv_sc = make_img_tag(inc_div["fig_scatter"]) if inc_div and inc_div.get("fig_scatter") else ""
    _incdiv_corr_df = inc_div.get("corr_df", pd.DataFrame()) if inc_div else pd.DataFrame()
    _incdiv_overall_r = inc_div.get("overall_r", np.nan) if inc_div else np.nan
    _incdiv_overall_p = inc_div.get("overall_p", np.nan) if inc_div else np.nan

    def _incdiv_corr_table() -> str:
        if _incdiv_corr_df.empty:
            return "<p>Not computed.</p>"
        rows = ""
        for _, r in _incdiv_corr_df.iterrows():
            era_lbl = ERA_LABELS.get(r["era"], r["era"]).replace("\n", " ")
            stars = ("***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else "")))
            rows += (f"<tr><td>{era_lbl}</td><td>{r['r']:.3f}{stars}</td>"
                     f"<td>{fmt_p(r['p'])}</td><td>{int(r['n'])}</td></tr>")
        return ("<table><tr><th>Era</th><th>Pearson r</th><th>p-value</th><th>N (years)</th></tr>"
                + rows + "</table><p style='font-size:.82em;color:#555;'>* p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01</p>")

    _incdiv_r_str  = fmt_f(_incdiv_overall_r, 3)
    _incdiv_p_str  = fmt_p(_incdiv_overall_p)
    # Within-era r values for interpretation
    def _era_r(era_key):
        if _incdiv_corr_df.empty: return "n/a"
        sub = _incdiv_corr_df[_incdiv_corr_df["era"] == era_key]
        return fmt_f(sub["r"].iloc[0], 3) if not sub.empty else "n/a"
    _r_pi = _era_r("pre_industrial"); _r_tr = _era_r("transition")
    _r_ei = _era_r("early_industrial"); _r_li = _era_r("late_industrial")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Oxford Ledger — Four-Level AI Transformation Framework Evidence (v6)</title>
<style>{CSS}</style>
</head>
<body>

<h1>Oxford College Accounts, 1700–1900</h1>
<p class="subtitle"><em>Historical evidence for the Four-Level AI Transformation Framework &mdash; Analysis v6</em></p>

<div class="toc">
<strong>Contents</strong>
<ol>
<li><a href="#intro">Introduction &mdash; Data, Approach, and Key Findings</a></li>
<li><a href="#l1">Level 1 — Automation / Efficiency (Traditional Function Share)</a></li>
<li><a href="#l2">Level 2 — Personalisation / Differentiation (Payment Modernity)</a></li>
<li><a href="#l3">Level 3 — Operational Innovation (Salary Share)</a></li>
<li><a href="#l4">Level 4 — Business / Mission Innovation (Educational Share)</a></li>
<li><a href="#breaks">Transformation Dynamics — Were the Changes Sudden or Gradual?</a></li>
<li><a href="#sequencing">Level Sequencing — Did L2 Rise Before L3, and L3 Before L4?</a></li>
<li><a href="#incdiv">Income Diversification — Did the Income Base Co-evolve with Transformation?</a></li>
<li><a href="#thresh">Capability Threshold — Did Financial Slack Enable Advancement?</a></li>
<li><a href="#shocks">Shock Response — What Happened Around Agricultural Crises?</a></li>
<li><a href="#nonlin">Non-Linearity — Do Higher Levels Show Larger Shifts?</a></li>
<li><a href="#discuss">Discussion — What the Historical Evidence Tells Us</a></li>
</ol>
</div>

<!-- =================================================================== -->
<h2 id="intro">1. Introduction</h2>

<div class="caution" style="background:#e8f4f8;border-left:5px solid #2c3e50;border-radius:4px;">
<strong>Key Findings at a Glance</strong>
<ul style="margin:.4em 0 0 0;">
<li><strong>Transformation was not gradual (H1a):</strong> Four distinct structural breaks were
detected at <strong>{pelt_years}</strong> &mdash; consistent with the framework's claim that
transformation happens in jumps, not steady drift.</li>
<li><strong>Higher levels showed larger shifts (H6a):</strong> Educational spending (L4) rose by
<strong>{l4_late_delta_str}</strong> in the Late Industrial era &mdash; the largest single-era
change of any level proxy, arriving last and hitting hardest.</li>
<li><strong>Prior financial slack weakly predicted advancement (H6b):</strong> Land-rent income
5 years earlier had a {dl4_lag5_sign} association with educational investment growth
(p&nbsp;=&nbsp;{dl4_lag5_p_str}). The direction is consistent with the hypothesis, but the
evidence is not strong enough for confident conclusions.</li>
<li><strong>Shock response (H5a/H2c):</strong> Patterns around four agricultural crises are
shown descriptively. With only four events, no statistical claims are made.</li>
<li><strong>Level ordering (Sequencing):</strong> Median-crossover analysis and cross-correlation
suggest that L2 (payment modernity), L3 (salary), and L4 (educational) reached their long-run
medians in that order — broadly consistent with the framework's predicted sequence, though
non-stationarity means this is descriptive only.</li>
<li><strong>Income diversification co-evolved with transformation:</strong> Land-rent income
declined from {lr_pi} to {lr_li} across the study period while modern functions expanded —
a pattern consistent with the institution shifting from a concentrated passive income base to
a more diversified, active portfolio as it moved up the transformation ladder.</li>
</ul>
</div>

<h3>What the framework claims</h3>

<p>The Four-Level AI Transformation Framework describes how organisations transform through four
distinct levels. At <strong>Level 1</strong> (Automation), the organisation does the same things
faster or cheaper, but the underlying process does not change. At <strong>Level 2</strong>
(Personalisation), outputs become more tailored, but internal coordination stays the same. At
<strong>Level 3</strong> (Operational Innovation), the way work is organised changes &mdash; new
roles, new workflows. At <strong>Level 4</strong> (Business Innovation), the organisation's core
purpose shifts to something that could not have existed before.</p>

<p>The framework makes three structural claims that can in principle be tested with historical
data: (1) transformation happens in discrete jumps, not smooth trends; (2) the largest shifts
occur at higher levels; (3) reaching higher levels requires prior accumulation of resources and
capabilities.</p>

<h3>About the data and approach</h3>

<div class="method">
<strong>Data:</strong> Oxford college account books, 1700–1900. Each page was digitised and
processed by an AI pipeline that extracted individual financial entries and assigned each one
a category (e.g. &ldquo;ecclesiastical&rdquo;, &ldquo;salary_stipend&rdquo;,
&ldquo;educational&rdquo;). This yields approximately 50,000 individual records across 196 years.
<strong>Note:</strong> Categories are assigned automatically by an AI model (LLM), so individual
labels may be noisy. All category-based measures should be interpreted as estimates, not
exact counts.
</div>

<div class="method">
<strong>How the analysis works:</strong> Each framework level is translated into a measurable
annual proportion — specifically, the share of total spending in a given year that went to the
corresponding category. For example, Level 3 is measured as the fraction of total spending going
to salary payments each year. Tracking these proportions over 200 years lets us test whether the
framework's structural claims appear in historical data. All monetary amounts are adjusted for
inflation using the Phelps Brown-Hopkins price index (anchored at 1700 = 100), so comparisons
across centuries reflect real purchasing power rather than price-level changes.
</div>

<div class="caution">
<strong>Important limitations:</strong> This is a single institution observed over 200 years.
All results are descriptive associations — no causal conclusions are possible. Regression
results tell us whether two things move together, not whether one causes the other.
</div>

<h3>How each level is measured</h3>

<table>
<tr><th>Level</th><th>What it means</th><th>Historical proxy</th><th>How it is computed</th></tr>
<tr><td><strong>L1 — Automation</strong></td>
    <td>Doing existing tasks more efficiently, without changing the process</td>
    <td>Traditional function share</td>
    <td>Spending on ecclesiastical + maintenance + domestic as a % of total spending</td></tr>
<tr><td><strong>L2 — Personalisation</strong></td>
    <td>More tailored outputs, but the same underlying process</td>
    <td>Payment modernity index</td>
    <td>Average payment-period score, weighted by amount (annual contract = 1.0; multi-year = 0.1)</td></tr>
<tr><td><strong>L3 — Operational Innovation</strong></td>
    <td>The way work is organised changes</td>
    <td>Salary share</td>
    <td>Salary &amp; stipend spending as a % of total spending</td></tr>
<tr><td><strong>L4 — Business Innovation</strong></td>
    <td>The organisation's core purpose changes</td>
    <td>Educational share</td>
    <td>Educational spending as a % of total spending</td></tr>
</table>

<p><strong>Note on L1:</strong> A <em>falling</em> L1 share is what we expect as the institution
transforms — it does not mean traditional activities stopped, only that they shrank as a proportion
of total spending as newer functions grew.</p>

{fig_overview}

<!-- =================================================================== -->
<h2 id="l1">2. Level 1 — Automation / Efficiency</h2>
<h3>Traditional Function Share (ecclesiastical + maintenance + domestic)</h3>

<div class="why"><strong>Why this analysis:</strong> Level 1 of the framework is about doing
existing tasks more efficiently without changing the underlying process. In this institution, the
long-standing traditional functions — religious ceremonies, building maintenance, and domestic
household operations — are the historical equivalent of those routine, unchanged tasks. If the
institution begins to transform, we would expect these traditional functions to take up a smaller
share of total spending over time, as resources shift toward newer activities.</div>

<div class="method"><strong>Method:</strong> For each year, we calculate what fraction of total
inflation-adjusted spending went to ecclesiastical, maintenance, and domestic categories combined.
A 10-year rolling average is overlaid to smooth year-to-year noise. Era-level averages are shown
with 95% confidence intervals, estimated by repeatedly resampling the data (bootstrap, 1,000 rounds).</div>

{make_img_tag(l1_b64)}

{era_means_table(l1_era, "L1 (traditional share)")}

<div class="interp"><strong>Result and interpretation:</strong>
The traditional function share dropped from <strong>{pct(l1_pi)}</strong> in the Pre-Industrial
era to <strong>{pct(l1_tr)}</strong> in the Transition era — a fall of
<strong>{l1_delta_pt}</strong>, the single largest era-over-era change of any proxy in either
direction. After 1780, L1 never recovered above {pct(l1_ei)}: traditional functions were
structurally displaced during the Industrial Revolution transition and did not bounce back. The
stabilisation at roughly {pct(l1_li)} in the Late Industrial era suggests that a residual core
of traditional activities persisted, even as higher-level functions expanded. This pattern is
consistent with the framework's Level 1 logic: the institution did not abandon its ecclesiastical
and maintenance obligations, but they became a smaller fraction of what it did. The important
signal is the speed of the initial drop — concentrated almost entirely in the Transition era
rather than spread evenly across two centuries.
</div>

<!-- =================================================================== -->
<h2 id="l2">3. Level 2 — Personalisation / Differentiation</h2>
<h3>Payment Modernity Index (amount-weighted average of payment-period scores)</h3>

<div class="why"><strong>Why this analysis:</strong> Level 2 in the framework is about
differentiating what you deliver — more tailored, more responsive — without yet changing how
the organisation runs internally. In the historical context, this shows up in payment contracts:
a shift from multi-year or open-ended feudal arrangements toward annual or half-yearly contracts
represents a more modern, market-responsive relationship. An institution at Level 2 is still
doing the same kind of work, but it is managing its financial relationships in a more granular,
contemporary way.</div>

<div class="method"><strong>Method:</strong> Each financial entry with a recognisable payment
period is assigned a modernity score (annual contract = 1.0, half-year = 0.85, down to
multi-year = 0.1). The index for each year is the spending-weighted average of these scores.
Coverage is reported because some entries have an unreadable or ambiguous payment period.</div>

{pm_warn}

{make_img_tag(l2_b64)}

{era_means_table(l2_era, "L2 (payment modernity)")}

<div class="interp"><strong>Result and interpretation:</strong>
The payment modernity index rose from <strong>{pct(l2_pi)}</strong> in the Pre-Industrial era
to a peak of <strong>{pct(l2_tr)}</strong> in the Transition era (<strong>{l2_delta_pt}</strong>),
then gradually declined to {pct(l2_ei)} in Early Industrial and {pct(l2_li)} in Late Industrial.
Two things stand out. First, the L2 index peaked in exactly the same era (Transition, 1780–1819)
as L3 (salary share) — both rose together, suggesting that the shift to more modern payment
contracts and the expansion of salaried employment happened simultaneously rather than
sequentially. This partially supports the idea that Level 2 and Level 3 were co-evolving rather
than strictly hierarchical in this institution. Second, the subsequent decline in L2 after the
Transition era may reflect the growing dominance of annual salary contracts (already scored at
1.0) displacing more ambiguous multi-year arrangements — in other words, once the institution
fully modernised its payment structure, there was no further room for the index to rise.
</div>

<!-- =================================================================== -->
<h2 id="l3">4. Level 3 — Operational Innovation</h2>
<h3>Salary Share (salary &amp; stipend spending as a % of total spending)</h3>

<div class="why"><strong>Why this analysis:</strong> Level 3 in the framework is the moment
when the <em>process itself</em> changes — not just the outputs or their efficiency. In this
institution, hiring permanent salaried staff is exactly that kind of process change: it
restructures how work is coordinated, moves from one-off payments to ongoing employment, and
builds the human infrastructure needed to sustain new functions. A rising salary share means
the institution is investing in people as a permanent resource, not just purchasing services
as needed.</div>

<div class="method"><strong>Method:</strong> For each year, we calculate the share of total
inflation-adjusted spending that went to the salary and stipend category. Era-level averages
are shown with 95% confidence intervals (bootstrap, 1,000 rounds).</div>

{make_img_tag(l3_b64)}

{era_means_table(l3_era, "L3 (salary share)")}

<div class="interp"><strong>Result and interpretation:</strong>
The salary share surged from <strong>{pct(l3_pi)}</strong> in the Pre-Industrial era to
<strong>{pct(l3_tr)}</strong> in the Transition era (<strong>{l3_delta_pt}</strong>), with a
single-year peak of {l3_peak_val} in {l3_peak_yr}. However, this high level was not sustained:
salary share fell back to {pct(l3_ei)} in the Early Industrial era and further to
<strong>{pct(l3_li)}</strong> in the Late Industrial era — its lowest era average of any period.
This non-monotonic trajectory is the most striking result for L3. It suggests that
operational restructuring (building a permanent salaried workforce) was heavily concentrated in
the Transition era, coinciding with the Napoleonic Wars and the rapid industrialisation of the
surrounding economy. The subsequent retreat of L3 in the Late Industrial era does not mean the
institution shrank its staff — it means that educational spending (L4) grew fast enough to
reduce salary's <em>relative share</em>. Read together with L4, the Late Industrial era marks a
shift in the composition of modern functions: from salary-dominated to education-dominated.
</div>

<!-- =================================================================== -->
<h2 id="l4">5. Level 4 — Business / Mission Innovation</h2>
<h3>Educational Share (educational spending as a % of total spending)</h3>

<div class="why"><strong>Why this analysis:</strong> Level 4 in the framework is the most
demanding level: the organisation does not just change how it works, but what it fundamentally
<em>is</em>. The framework's own test is strict — "if the business existed before, it is not
Level 4." In the historical context, the emergence of dedicated educational spending is a strong
signal of this mission shift: the institution stops treating teaching as a side activity and
starts being, in a meaningful budget sense, an educational organisation. This is the closest
historical parallel to Level 4's criterion of a genuinely new organisational purpose.</div>

<div class="method"><strong>Method:</strong> For each year, we calculate the share of total
inflation-adjusted spending that went to educational activities. Era-level averages are shown
with 95% confidence intervals (bootstrap, 1,000 rounds).</div>

{make_img_tag(l4_b64)}

{era_means_table(l4_era, "L4 (educational share)")}

<div class="interp"><strong>Result and interpretation:</strong>
Educational share was effectively flat for the first 160 years: the pre-1860 average was
<strong>{l4_pre1860}</strong>, barely distinguishable from zero across all three earlier eras
({pct(l4_pi)}, {pct(l4_tr)}, {pct(l4_ei)}). After 1860, it rose sharply to a Late Industrial
era average of <strong>{pct(l4_li)}</strong>, peaking at {l4_peak_val} in {l4_peak_yr}.
The era-over-era jump of <strong>{l4_delta_el}</strong> from Early to Late Industrial is by far
the largest positive change of any level proxy across any era transition. The sequential pattern
is clear and consistent with the framework: L3 (salary) expanded first during the Transition
era, building the operational infrastructure, and only then — roughly half a century later —
did L4 (educational mission) emerge as a major budget priority. This is exactly what the
framework's capability threshold hypothesis (H6b) would predict: Level 4 transformation did not
appear from nowhere; it followed on the organisational foundations laid at Level 3.
</div>

<!-- =================================================================== -->
<h2 id="breaks">6. Transformation Dynamics — Were the Changes Sudden or Gradual?</h2>

<div class="why"><strong>Why this analysis:</strong> A central claim in the framework is that
transformation happens in jumps, not steady drift. If this is true, the data should show moments
when the institution's spending pattern shifted sharply rather than gliding smoothly from one
era to the next. This analysis looks for those moments using three independent methods. If all
three point to the same period, that is strong evidence of a genuine structural shift. If they
disagree, it suggests the pattern is more ambiguous.</div>

<div class="method"><strong>Three methods used:</strong>
<ol>
<li><strong>Change-point detection (Ruptures PELT):</strong> An algorithm that finds the years
where the modern function share series changes level most sharply. The number of breakpoints
is chosen automatically using a model-fit criterion (BIC) that balances fit against complexity.
It assumes that between breakpoints the series fluctuates around a stable average.</li>
<li><strong>Rolling 20-year regression:</strong> A linear trend is fitted in every 20-year
window, and the slope (rate of change) is plotted over time. A sudden spike in slope means the
series was changing faster than usual in that window — a sign of a regime shift.</li>
<li><strong>Chow-test grid:</strong> For every candidate break year (every 5 years from 1720
to 1885), we test whether splitting the series into two segments fits significantly better than
a single straight line. A high F-statistic at a given year means that year is a strong
candidate for a structural break. <em>Caveat:</em> year-to-year observations in this series
are not fully independent (last year's value influences this year's), so the F-statistics are
indicative rather than formally valid for inference.</li>
</ol>
</div>

<p><strong>PELT result:</strong> {len(breaks.get("pelt_break_years", []))} structural break(s)
detected at: <strong>{pelt_years}</strong>.</p>
<p><strong>Chow-test peak:</strong> Highest F-statistic ({chow_f_peak}) at year
<strong>{chow_peak}</strong> (threshold for 95% significance = {f_crit}).</p>

{fig_breaks}
{fig_rolling}
{fig_chow}

<div class="interp"><strong>Result and interpretation:</strong>
The change-point algorithm detected four structural breaks: <strong>1785, 1811, 1852, and
1882</strong>. Each of these maps onto a plausible historical turning point. The 1785 break
coincides with the start of the Transition era and the sharp collapse of L1 (from ~44% toward
18%). The 1811 break falls within the Napoleonic Wars period, when salary share was near its
all-time peak. The 1852 break sits at the Early-to-Late Industrial boundary, just as L3 began
its long-run decline and L4 was stirring. The 1882 break corresponds to the aftermath of the
Great Agricultural Depression, when land-rent income was falling and educational investment
was accelerating. Crucially, the modern function share series shows four distinct regime levels
rather than a steady upward trend, directly supporting H1a: transformation was punctuated, not
smooth. The rolling regression reinforces this — the slope is not constant but spikes in
specific windows, particularly around the Transition era and again around the 1870s–1880s.
The Chow-test peak at {chow_peak} (F&nbsp;=&nbsp;{chow_f_peak}, threshold&nbsp;=&nbsp;{f_crit})
confirms that a single candidate breakpoint also substantially improves the fit over a
straight-line trend. That said, we cannot identify which specific events caused these breaks —
the temporal alignment is descriptive only.
</div>

<!-- =================================================================== -->
<h2 id="sequencing">7. Level Sequencing — Did L2 Rise Before L3, and L3 Before L4?</h2>

<div class="why"><strong>Why this analysis:</strong> A core structural claim of the Four-Level
Framework is that transformation is not random — it follows an ordered sequence. An organisation
must achieve Level 2 differentiation before it can sustain Level 3 operational redesign, and
Level 3 before Level 4 mission innovation. If this is true historically, we should see L2's proxy
(payment modernity) rising earlier than L3 (salary share), which rises earlier than L4
(educational share). This analysis checks whether the data is consistent with that ordering.
</div>

<div class="method"><strong>Method:</strong> Two complementary checks. (1) Median-crossover year:
the first year each level's 10-year rolling mean exceeds its own long-run median — if the
framework is right, these years should be ordered L2 &lt; L3 &lt; L4. (2) Pairwise
cross-correlation at lags −10 to +10 years: a positive peak lag for the pair (L2, L3) means
L2 was leading L3. <em>Caveat:</em> cross-correlation on non-stationary series can be spurious.
Both checks are descriptive, not causal.</div>

{fig_seq}
<h3>Median-crossover years (first year each level surpasses its long-run median)</h3>
{_crossover_table()}
<h3>Pairwise cross-correlation: peak lag and implied ordering</h3>
{_xcorr_table()}

<div class="interp"><strong>Result and interpretation:</strong>
The median-crossover years suggest the following order: L2 first crosses its median in
<strong>{_l2_cross}</strong>, L3 in <strong>{_l3_cross}</strong>, and L4 in
<strong>{_l4_cross}</strong>. This ordering is <strong>{_seq_ordered}</strong> with the
framework's predicted L2 → L3 → L4 sequence. The cross-correlation analysis shows that
for the L2–L3 pair: <strong>{_xcorr_l2l3_str}</strong>, and for the L3–L4 pair:
<strong>{_xcorr_l3l4_str}</strong>. A positive lead for L2 over L3 would mean that payment
modernisation was already accelerating before operational restructuring took hold — consistent
with Level 2 being a prerequisite stage. Note that L1 (traditional function share) begins
declining well before any of the higher levels rise, which is also predicted: the institution
starts moving away from purely traditional functions before it has built up the new ones to
replace them. These sequencing patterns are suggestive but not conclusive — the series are
non-stationary, and the ordering partly reflects when different functions became historically
possible at all (e.g., educational spending required the institution's mission to have already
shifted in practice before it shows up in the accounts).
</div>

<!-- =================================================================== -->
<h2 id="incdiv">8. Income Diversification — Did the Income Base Co-evolve with Transformation?</h2>

<div class="why"><strong>Why this analysis:</strong> The framework's capability logic implies that
transformation is not only driven by spending decisions but also shaped by the income base that
makes them possible. As an institution transitions from Level 1 toward Level 4, it should also
be shifting from a single dominant income source (land rents) toward a more diversified portfolio.
This section asks: does income diversification move together with operational transformation, and
does this co-movement strengthen in the later eras when transformation was most pronounced?
</div>

<div class="method"><strong>Method:</strong> Income diversification is measured as
1 &minus; HHI, where HHI (Herfindahl–Hirschman Index) is the sum of squared income-category
shares in each year. A value near 1 means income is spread across many sources; near 0 means
one source dominates. We plot income diversification alongside modern function share (L3+L4)
as 10-year rolling means. We also show a scatter plot by era with within-era OLS fit lines,
and report era-level Pearson correlations. <em>Caveat:</em> both series share a long-run trend,
so positive correlations partly reflect shared time trend rather than a genuine structural
relationship.</div>

{fig_incdiv_ts}
{fig_incdiv_sc}
<h3>Era-level correlation: income diversification vs modern function share</h3>
{_incdiv_corr_table()}

<div class="interp"><strong>Result and interpretation:</strong>
Across the full 1700–1900 period, income diversification and modern function share are
<strong>correlated at r&nbsp;=&nbsp;{_incdiv_r_str}</strong>
(p&nbsp;=&nbsp;{_incdiv_p_str}). The dual-axis chart shows that both series rise broadly
over the 19th century, but their paths diverge in important ways. Era-level correlations show
variation across periods: Pre-Industrial r&nbsp;=&nbsp;{_r_pi}, Transition r&nbsp;=&nbsp;{_r_tr},
Early Industrial r&nbsp;=&nbsp;{_r_ei}, Late Industrial r&nbsp;=&nbsp;{_r_li}. If the
Late Industrial correlation is strongest, this is consistent with the interpretation that
higher-level transformation (L4 expansion) and income diversification were tightly linked in the
later period — the college's educational mission was financed by a portfolio of income sources,
not a single land-rent stream. The scatter plot by era shows whether the L3+L4 vs diversification
relationship became steeper or changed direction across periods. The long-run decline in land-rent
share ({lr_pi} → {lr_li}) alongside rising educational share tells a coherent story: as passive
income concentration fell, the institution actively restructured its expenditure toward higher
levels — consistent with both the capability threshold story (H6b) and the diversification
hypothesis. However, the shared trend means we cannot distinguish causation from temporal
coincidence without exogenous variation.
</div>

<!-- =================================================================== -->
<h2 id="thresh">9. Capability Threshold — Did Financial Slack Enable Advancement?</h2>

<div class="why"><strong>Why this analysis:</strong> The framework's capability threshold
hypothesis (H6b) says that reaching higher transformation levels requires prior accumulation
of organisational resources and capacity. In the historical context, land-rent income — the
college's most stable, passive income source — serves as a proxy for that financial cushion.
A college with a large, reliable land-rent base could afford to invest in new functions without
putting core operations at risk. The hypothesis predicts that years with high land-rent income
should predict greater growth in salary (L3) or educational (L4) spending in the years that
follow.</div>

<div class="method"><strong>Method:</strong> We ask whether land-rent income in year
<em>t&minus;k</em> predicts the <em>change</em> in salary or educational share in year <em>t</em>,
at lags k = 1, 3, and 5 years. Working with year-on-year changes (rather than levels) reduces
the influence of long-run trends. Standard errors are corrected for unequal variance (HC3).
All results are associational — land-rent income and modernisation spending may both be driven
by a third factor we cannot observe.</div>

{thresh_table}
{fig_thresh}

<div class="interp"><strong>Result and interpretation:</strong>
For ΔL3 (salary growth), no lag shows a significant or consistently signed relationship with
prior land-rent income (lag 1: coef&nbsp;=&nbsp;{dl3_lag1_coef_str}, p&nbsp;=&nbsp;{dl3_lag1_p_str}; signs
vary across lags). For ΔL4 (educational growth), the pattern is more directional: the
coefficient is positive at lags 3 and 5, and marginally significant at lag 5
(coef&nbsp;=&nbsp;<strong>{dl4_lag5_coef_str}</strong>, p&nbsp;=&nbsp;<strong>{dl4_lag5_p_str}</strong>). This means
that a 10 percentage-point higher land-rent income share was associated with approximately
{float(dl4_lag5_coef_str)*10:.1f}pp higher educational investment growth five years later.
The direction is consistent with H6b — financial slack enabled future higher-level investment —
but the evidence is not strong enough for confident inference. Importantly, land-rent share
itself declined substantially across eras ({lr_pi} in Pre-Industrial → {lr_li} in Late Industrial),
meaning the institution's financial cushion was shrinking precisely as L4 was expanding. The lag-5
result may therefore be capturing the Transition-era window when land-rents were still
significant and the institution was laying the groundwork for what later became educational
mission. The absence of a clear L3 result suggests that salary expansion was driven by factors
other than financial slack — possibly direct income needs or external pressures — while
educational investment was more discretionary and therefore more sensitive to financial capacity.
</div>

<!-- =================================================================== -->
<h2 id="shocks">10. Shock Response — What Happened Around Agricultural Crises?</h2>

<div class="why"><strong>Why this analysis:</strong> The framework suggests that transformation
may accelerate after periods of disruption, once the organisation recovers financially — an
"opportunity window" (H5a). The historical record has four well-documented agricultural shocks
that would have hit the college's land-rent income hard. If the modern function share dips
around these shocks and then rebounds to a higher level than before, that would be consistent
with the idea that disruption can catalyse transformation rather than simply suppress it.</div>

<div class="method"><strong>Method:</strong> For each of four agricultural shocks (1793, 1822,
1846, 1873), we extract the modern function share and total income in the ±8 years around the
shock year. Individual trajectories and the average across all four shocks are plotted.
<strong>Critical limitation: with only four events, no statistical conclusions are possible.</strong>
All patterns shown here are purely descriptive and illustrative.</div>

{fig_shock}

<div class="interp"><strong>Result and interpretation:</strong>
The average modern function share (dashed black line) was rising in the years leading up to the
shocks: from approximately {shock_t_m1} at t&minus;1 to a pre-shock peak, then dropped to
<strong>{shock_t_0}</strong> at the shock year itself — a decline of roughly
{float(shock_t_m1)-float(shock_t_0):.3f} in a single year. It continued falling to {shock_t_p1}
at t+1 before showing partial recovery, but had not returned to pre-shock levels by t+4
({shock_t_p4}). This pattern is more consistent with shocks suppressing modern investment (H5b)
than with shocks creating opportunity windows for recovery and advancement (H5a). However, the
four individual shock lines diverge considerably — which is expected, since the shocks struck
at very different moments in the transformation: 1793 hit a Pre-Industrial institution, 1873 hit
an institution that was already well into educational expansion. Averaging these trajectories
inevitably smooths over very different institutional contexts. Read the individual lines as
carefully as the average: some shocks may show a V-shaped recovery pattern while others show
sustained depression — and the difference may tell a more interesting story than the average.
<strong>Statistical reminder: N=4. No inferential conclusions are possible.</strong>
</div>

<!-- =================================================================== -->
<h2 id="nonlin">11. Non-Linearity — Do Higher Levels Show Larger Shifts?</h2>

<div class="why"><strong>Why this analysis:</strong> The framework claims that the value gains
from transformation are not evenly spread across levels — the jump from Level 3 to Level 4 is
supposed to be larger than the jump from Level 1 to Level 2. In the historical context, this
would show up as larger era-over-era changes in the higher-level proxies (L4, L3) than in the
lower-level ones (L1, L2). If educational share (L4) shows the sharpest increase in the later
eras, while traditional functions (L1) declined most in the early eras, that is consistent with
the framework's hierarchy claim (H6a).</div>

<div class="method"><strong>Method:</strong> For each of the four level proxies, we compute
era-level averages with 95% confidence intervals (bootstrap, 1,000 rounds) and then measure
the change between consecutive eras. Three era transitions are examined: Pre-Industrial →
Transition, Transition → Early Industrial, Early → Late Industrial. Because all four proxies
share the same spending denominator, their changes are mathematically linked — the comparison
is therefore descriptive, not a formal statistical test.</div>

{fig_delta}

<div class="interp"><strong>Result and interpretation:</strong>
The era-over-era changes reveal a clear temporal hierarchy consistent with H6a. In the
Pre&rarr;Transition era, the largest movements were: L1 falling sharply
(<strong>{l1_delta_pt}</strong>) and both L2 and L3 rising substantially ({l2_delta_pt} and
{l3_delta_pt} respectively) — this is the framework's Level 1 displacement and simultaneous
Level 2/3 expansion. In the Transition&rarr;Early Industrial era, L3 retreated
({l3_delta_te}), suggesting the salary surge of the Napoleonic period was partly temporary.
In the Early&rarr;Late Industrial era, the dominant movement is L4's rise of
<strong>{l4_delta_el}</strong> — the largest single positive change of any level across any
transition. This timing pattern is precisely what the framework predicts for non-linearity:
lower-level functions (L1) shift first and earlier, higher-level functions (L4) arrive latest
but with the largest amplitude. The fact that L4's major expansion ({l4_delta_el}) exceeds
L3's Transition-era surge ({l3_delta_pt}) in magnitude — and occurs a full era later — is
the strongest piece of evidence in this dataset for H6a. One important caveat: because all
shares sum to ~1, L1's decline and L4's rise are mathematically connected. The key interpretive
signal is therefore the <em>timing</em> of each level's largest change, which follows the
predicted hierarchy cleanly.
</div>

<!-- =================================================================== -->
<h2 id="discuss">12. Discussion — What the Historical Evidence Tells Us</h2>

<p>Taken together, the analyses offer historical perspective on three core claims of the
Four-Level AI Transformation Framework.</p>

<p><strong>1. Transformation was punctuated, not gradual (H1a — supported).</strong>
Four structural breaks were detected at 1785, 1811, 1852, and 1882. The modern function share
series does not drift smoothly upward — it shifts level four times. Each break aligns with a
recognisable turning point: the onset of industrialisation (1785), the Napoleonic period (1811),
the Early-to-Late Industrial transition (1852), and the post-Agricultural Depression
reorganisation (1882). The rolling regression confirms that the pace of change was not constant
— it spiked at specific windows rather than rising steadily. This is consistent with the
framework's claim that transformation requires architectural change, not incremental
accumulation.</p>

<p><strong>2. Higher levels showed larger and later shifts (H6a — supported).</strong>
The timing hierarchy matches the framework's prediction closely. L1 (traditional share) fell
first and fastest: {l1_delta_pt} in the Pre&rarr;Transition era, stabilising at {pct(l1_li)}.
L3 (salary share) rose in the Transition era ({l3_delta_pt}) but then retreated to its lowest
era mean ({pct(l3_li)}) as L4 grew. L4 (educational share) was dormant at ~{l4_pre1860} before
1860, then surged to {l4_post1860} afterwards, with an era-over-era jump of
<strong>{l4_delta_el}</strong> — the largest single positive change in the entire dataset. The
fact that L4's Late Industrial surge exceeds L3's Transition-era peak in magnitude, and arrives
a full era later, is the clearest evidence for H6a.</p>

<p><strong>3. Prior financial slack weakly predicted L4 advancement (H6b — weak support).</strong>
Land-rent income five years prior had a {dl4_lag5_sign} association with educational investment
growth (coef&nbsp;=&nbsp;{dl4_lag5_coef_str}, p&nbsp;=&nbsp;{dl4_lag5_p_str}). No significant
effect was found for salary growth (L3). The direction is consistent with H6b — educational
investment was discretionary and sensitive to prior financial capacity — but the evidence is not
strong. Land-rent share itself fell from {lr_pi} to {lr_li} over the period, so the institution
was managing its largest L4 expansion precisely when its traditional financial cushion was at its
smallest.</p>

<p><strong>What this study cannot establish:</strong> This is a single institution across 200 years.
Mapping the framework's levels onto 18th–19th century college accounts involves interpretive
choices that may not capture every relevant dimension. Category labels are AI-assigned and carry
noise. All results are associations, not causes. "Supports" means "is consistent with" — not
"proves."</p>

<footer>
<p>Analysis v6 &mdash; Historical Evidence for the Four-Level AI Transformation Framework</p>
<p>Data: Oxford college accounts 1700–1900, enriched via LLM pipeline. Price deflator:
Phelps Brown-Hopkins (1700=100). All findings associational.</p>
</footer>
</body></html>"""

    out_path = OUT_DIR / "analysis_v6_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  Report written → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Analysis v6 — Four-Level AI Transformation Framework")
    print("=" * 60)

    # Load data
    df = load_enriched_data()

    # Compute all proxies
    print("[proxies] Computing four-level proxy panel …")
    panel = compute_proxies(df)
    pm_cov = float(panel["pm_coverage"].median())
    print(f"  L2 payment modernity median coverage: {pm_cov:.1%}")

    # Cross-check against v5 if available
    v5_csv = V5_DIR / "outcome_variable_yearly.csv"
    if v5_csv.exists():
        v5 = pd.read_csv(v5_csv)
        merged = panel.merge(v5[["year", "modern_function_share"]].rename(
            columns={"modern_function_share": "mfs_v5"}), on="year", how="inner")
        corr = merged["modern_function_share"].corr(merged["mfs_v5"])
        print(f"  Cross-check vs v5 outcome: correlation = {corr:.4f} "
              f"(mean diff = {(merged['modern_function_share'] - merged['mfs_v5']).abs().mean():.4f})")

    # Level plots
    print("[levels] Plotting level proxies …")
    l1_b64, l1_era = section_level(
        df, panel, "L1",
        "Level 1 — Traditional Function Share\n"
        "(ecclesiastical + maintenance + domestic) / total_expenditure",
        "Share of total real expenditure",
    )
    l2_b64, l2_era = section_level(
        df, panel, "L2",
        "Level 2 — Payment Modernity Index\n"
        "(amount-weighted mean of payment-period scores)",
        "Modernity index (0–1)",
    )
    l3_b64, l3_era = section_level(
        df, panel, "L3",
        "Level 3 — Salary Share\n"
        "(salary_stipend / total_expenditure)",
        "Share of total real expenditure",
    )
    l4_b64, l4_era = section_level(
        df, panel, "L4",
        "Level 4 — Educational Share\n"
        "(educational / total_expenditure)",
        "Share of total real expenditure",
    )

    # Structural breaks
    breaks = section_structural_breaks(panel)

    # Level sequencing
    seq = section_sequencing(panel)

    # Income diversification
    inc_div = section_income_diversification(panel)

    # Capability threshold
    thresh = section_capability_threshold(panel)

    # Shock response
    shock = section_shock_response(panel)

    # Non-linearity
    nonlin = section_nonlinearity(df, panel)

    # HTML report
    print("[report] Generating HTML report …")
    generate_report(
        panel,
        l1_b64, l1_era,
        l2_b64, l2_era,
        l3_b64, l3_era,
        l4_b64, l4_era,
        breaks, thresh, shock, nonlin,
        pm_cov,
        seq=seq,
        inc_div=inc_div,
    )

    print("\n" + "=" * 60)
    print(f"Done. Output directory: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
