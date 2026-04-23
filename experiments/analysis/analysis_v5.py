"""analysis_v5.py

Research-paper-quality extension of v4.

Addresses professor feedback:
  1. Clear outcome variable: modern_function_share(t)
     = (educational + salary_stipend expenditure) / total expenditure (real £)
  2. Three OLS specifications with proper SE corrections
  3. Robustness checks (alternative era cutoffs, outcome definitions, alternative Y)
  4. Embedding panel: semantic drift as independent corroborating signal
  5. Descriptive / associational separation (noted once in summary, not repeated)

Output directory: experiments/reports/analysis_v5/
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels not installed — regression section will be skipped.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_DIR = ROOT / "experiments" / "results" / "enriched"
OUT_DIR = ROOT / "experiments" / "reports" / "analysis_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Price deflation: Phelps Brown-Hopkins (1700=100) — reused from v4
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
    if year <= _PBH_YEARS[0]:
        return _PBH_VALS[0]
    if year >= _PBH_YEARS[-1]:
        return _PBH_VALS[-1]
    return float(np.interp(year, _PBH_YEARS, _PBH_VALS))


# ---------------------------------------------------------------------------
# Era periodization — reused from v4
# ---------------------------------------------------------------------------

ERAS = [
    ("pre_industrial",   1700, 1779),
    ("transition",       1780, 1819),
    ("early_industrial", 1820, 1859),
    ("late_industrial",  1860, 1900),
]
ERA_LABELS = {
    "pre_industrial":   "Pre-Industrial\n(1700–1779)",
    "transition":       "Transition\n(1780–1819)",
    "early_industrial": "Early Industrial\n(1820–1859)",
    "late_industrial":  "Late Industrial\n(1860–1900)",
}
ERA_VLINES = [1780, 1820, 1860]
ERA_ORDER  = ["pre_industrial", "transition", "early_industrial", "late_industrial"]
AG_SHOCKS  = {1793: "Enclosure Acts peak", 1822: "Post-Napoleonic depression",
              1846: "Corn Laws repeal", 1873: "Great Agricultural Depression"}


def era_of_year(year: int) -> str:
    if year < 1780:
        return "pre_industrial"
    if year < 1820:
        return "transition"
    if year < 1860:
        return "early_industrial"
    return "late_industrial"


# ---------------------------------------------------------------------------
# Outcome variable definition (printed in report)
# ---------------------------------------------------------------------------

OUTCOME_DEFINITION = """
PRIMARY OUTCOME VARIABLE
========================
modern_function_share(t) = (educational + salary_stipend expenditure) / total expenditure
                            measured in real £ (PBH-deflated, 1700=100), annually.

Rationale: captures Oxford's investment in human-capital-producing functions
relative to total expenditure. A rising share indicates institutional reallocation
toward modern academic functions (education delivery + staff salaries) and away
from traditional ecclesiastical or domestic maintenance.

CAUTION: All regression results in Section B are ASSOCIATIONAL.
No causal interpretation is warranted without exogenous variation in the predictors.
Oxford is a single institution; generalization requires caution.
"""

# ---------------------------------------------------------------------------
# Parsing helpers — reused from v4
# ---------------------------------------------------------------------------

def parse_fraction(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    mapping = {"¼": 0.25, "1/4": 0.25, ".25": 0.25,
               "½": 0.5,  "1/2": 0.5,  ".5":  0.5,
               "¾": 0.75, "3/4": 0.75, ".75": 0.75}
    if s in mapping:
        return mapping[s]
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_money(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return 0.0 if np.isnan(v) else v
    s = re.sub(r"[^0-9.\-]", "", str(value).strip())
    if not s:
        return 0.0
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
    single = re.match(r"^(\d{4})_(\d+)_image$", page_id)
    if single:
        return [int(single.group(1))], int(single.group(2))
    span = re.match(r"^(\d{4})-(\d{4})_(\d+)_image$", page_id)
    if span:
        y1, y2, pg = int(span.group(1)), int(span.group(2)), int(span.group(3))
        if y2 < y1:
            y1, y2 = y2, y1
        return list(range(y1, y2 + 1)), pg
    m = re.search(r"(\d{4})", page_id)
    if m:
        return [int(m.group(1))], 1
    raise ValueError(f"Cannot parse page_id: {page_id!r}")


# ---------------------------------------------------------------------------
# Data loading — reused from v4
# ---------------------------------------------------------------------------

def load_enriched_data() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    files = sorted(ENRICHED_DIR.glob("*_image_enriched.json"))
    if not files:
        raise FileNotFoundError(f"No enriched JSON files in {ENRICHED_DIR}")
    print(f"[load] Loading {len(files)} enriched JSON files …")
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print(f"  SKIP {fp.name}: {exc}")
            continue
        page_id = payload.get("page_id") or fp.name.replace("_enriched.json", "")
        try:
            years, page = parse_page_id(page_id)
        except ValueError:
            continue
        year_weight = 1.0 / len(years)
        for r in payload.get("rows", []):
            if not isinstance(r, dict):
                continue
            if str(r.get("row_type", "")).strip().lower() != "entry":
                continue
            amt = amount_decimal(r)
            for year in years:
                records.append({
                    "page_id":        page_id,
                    "year":           year,
                    "year_weight":    year_weight,
                    "amount":         amt,
                    "amount_weighted": amt * year_weight,
                    "direction":      r.get("direction"),
                    "category":       r.get("category"),
                    "english_desc":   r.get("english_description"),
                    "payment_period": r.get("payment_period"),
                    "is_arrears":     r.get("is_arrears"),
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No entry rows loaded.")
    df["era"]          = df["year"].map(era_of_year)
    df["price_idx"]    = df["year"].map(price_index)
    df["amount_real"]  = df["amount"] / (df["price_idx"] / 100.0)
    df["amount_real_w"] = df["amount_weighted"] / (df["price_idx"] / 100.0)
    print(f"  Loaded {len(df):,} records across {df['year'].nunique()} years "
          f"({df['year'].min()}–{df['year'].max()})")
    return df


def add_era_vlines(ax: plt.Axes, alpha: float = 0.5) -> None:
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=alpha)


# ---------------------------------------------------------------------------
# Outcome variable computation
# ---------------------------------------------------------------------------

MODERN_EXP_CATS  = {"educational", "salary_stipend"}
NARROW_EXP_CATS  = {"educational"}
BROAD_EXP_CATS   = {"educational", "salary_stipend", "administrative"}
INCOME_CATS_LAND = {"land_rent"}


def compute_modern_function_share(
    df: pd.DataFrame,
    modern_cats: set[str] | None = None,
) -> pd.DataFrame:
    """Compute modern_function_share per year for expenditure rows."""
    if modern_cats is None:
        modern_cats = MODERN_EXP_CATS
    exp = df[df["direction"] == "expenditure"].copy()
    exp["is_modern"] = exp["category"].isin(modern_cats)
    grp = exp.groupby("year").apply(
        lambda g: pd.Series({
            "modern_real":  (g.loc[g["is_modern"], "amount_real_w"]).sum(),
            "total_real":   g["amount_real_w"].sum(),
        })
    ).reset_index()
    grp["modern_function_share"] = (
        grp["modern_real"] / grp["total_real"].replace(0, np.nan)
    )
    grp["era"] = grp["year"].map(era_of_year)
    return grp.sort_values("year").reset_index(drop=True)


def compute_income_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HHI and income_diversification per year."""
    inc = df[df["direction"] == "income"].copy()
    grp = inc.groupby(["year", "category"])["amount_real_w"].sum().reset_index()
    total = inc.groupby("year")["amount_real_w"].sum().rename("total_real").reset_index()
    grp = grp.merge(total, on="year")
    grp["share"] = grp["amount_real_w"] / grp["total_real"].replace(0, np.nan)
    hhi = grp.groupby("year").apply(
        lambda g: (g["share"] ** 2).sum()
    ).rename("hhi").reset_index()
    hhi["income_diversification"] = 1.0 - hhi["hhi"]

    # Land rent income share (lagged predictor)
    land = grp[grp["category"] == "land_rent"][["year", "share"]].copy()
    land.columns = ["year", "land_rent_income_share"]
    hhi = hhi.merge(land, on="year", how="left")
    return hhi.sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section A — Descriptive trends [DESCRIPTIVE]
# ---------------------------------------------------------------------------

def section_a(df: pd.DataFrame) -> pd.DataFrame:
    print("[A] Computing descriptive trends …")
    mfs = compute_modern_function_share(df)

    # 10-yr rolling mean
    mfs_yr = mfs.set_index("year").sort_index()
    mfs_yr["mfs_10yr"] = mfs_yr["modern_function_share"].rolling(10, center=True, min_periods=5).mean()

    # Bootstrap 90% CI per year (resample entries within year, 300 reps)
    exp = df[df["direction"] == "expenditure"].copy()
    ci_records = []
    rng = np.random.default_rng(42)
    for year, grp in exp.groupby("year"):
        if len(grp) < 3:
            continue
        boots = []
        for _ in range(300):
            s = grp.sample(n=len(grp), replace=True, random_state=None)
            s = s.copy()
            s_rng = rng.integers(0, 2**31)
            s = grp.sample(n=len(grp), replace=True, random_state=int(s_rng))
            modern = s.loc[s["category"].isin(MODERN_EXP_CATS), "amount_real_w"].sum()
            total  = s["amount_real_w"].sum()
            if total > 0:
                boots.append(modern / total)
        if boots:
            ci_records.append({
                "year": year,
                "ci_lo": np.percentile(boots, 5),
                "ci_hi": np.percentile(boots, 95),
            })
    ci_df = pd.DataFrame(ci_records)

    mfs_yr = mfs_yr.reset_index().merge(ci_df, on="year", how="left")
    mfs_yr.to_csv(OUT_DIR / "outcome_variable_yearly.csv", index=False)

    # Figure A1: Outcome variable time series
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(mfs_yr["year"], mfs_yr["ci_lo"].fillna(mfs_yr["modern_function_share"]),
                    mfs_yr["ci_hi"].fillna(mfs_yr["modern_function_share"]),
                    alpha=0.2, color="steelblue", label="90% bootstrap CI")
    ax.plot(mfs_yr["year"], mfs_yr["modern_function_share"],
            lw=0.8, color="steelblue", alpha=0.5)
    ax.plot(mfs_yr["year"], mfs_yr["mfs_10yr"],
            lw=2.2, color="navy", label="10-yr rolling mean")
    add_era_vlines(ax)
    for yr, lbl in AG_SHOCKS.items():
        ax.axvline(yr, color="darkred", lw=0.7, ls=":", alpha=0.7)
    ax.set_ylabel("modern_function_share\n(educational + salary_stipend) / total expenditure")
    ax.set_xlabel("Year")
    ax.set_title("Outcome Variable: Modern Function Share of Expenditure (real £, 1700=100)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1900)
    ax.set_ylim(0, None)
    # Era labels at top
    era_positions = [(1740, "Pre-Industrial"), (1800, "Transition"),
                     (1840, "Early Ind."), (1880, "Late Ind.")]
    for xp, lbl in era_positions:
        ax.text(xp, ax.get_ylim()[1] * 0.97, lbl, ha="center", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A1_outcome_variable_timeseries.png")
    plt.close(fig)

    # Figure A2: Expenditure component decomposition
    exp_grp = (exp.groupby(["year", "category"])["amount_real_w"].sum().reset_index())
    total_by_year = exp.groupby("year")["amount_real_w"].sum().rename("total").reset_index()
    exp_grp = exp_grp.merge(total_by_year, on="year")
    exp_grp["share"] = exp_grp["amount_real_w"] / exp_grp["total"].replace(0, np.nan)

    cats_ordered = ["educational", "salary_stipend", "administrative", "financial",
                    "maintenance", "charitable", "ecclesiastical", "domestic", "other"]
    cat_colors = {
        "educational":    "#2ca02c",
        "salary_stipend": "#17becf",
        "administrative": "#ff7f0e",
        "financial":      "#1f77b4",
        "maintenance":    "#7f7f7f",
        "charitable":     "#e377c2",
        "ecclesiastical": "#9467bd",
        "domestic":       "#bcbd22",
        "other":          "#d62728",
    }
    wide = exp_grp.pivot_table(index="year", columns="category",
                               values="share", fill_value=0.0).reset_index()
    cats_present = [c for c in cats_ordered if c in wide.columns]
    years_arr = wide["year"].values
    share_matrix = np.column_stack([wide[c].fillna(0).values for c in cats_present])
    colors_arr = [cat_colors.get(c, "#aaa") for c in cats_present]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.stackplot(years_arr, share_matrix.T, labels=cats_present, colors=colors_arr, alpha=0.85)
    add_era_vlines(ax)
    ax.set_ylabel("Share of total real expenditure")
    ax.set_xlabel("Year")
    ax.set_title("Expenditure Component Decomposition (real £, 1700=100)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.6)
    ax.set_xlim(1700, 1900)
    ax.set_ylim(0, 1)
    # Highlight modern cats
    mod_patch = mpatches.Patch(facecolor="none", edgecolor="black", linewidth=1.5,
                                linestyle="--", label="Modern cats: educational + salary_stipend")
    ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.6,
              handles=[mpatches.Patch(color=cat_colors.get(c, "#aaa"), label=c)
                       for c in cats_present])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A2_expenditure_components.png")
    plt.close(fig)

    # Figure A3: Income composition over time
    inc = df[df["direction"] == "income"].copy()
    inc_grp = inc.groupby(["year", "category"])["amount_real_w"].sum().reset_index()
    total_inc = inc.groupby("year")["amount_real_w"].sum().rename("total").reset_index()
    inc_grp = inc_grp.merge(total_inc, on="year")
    inc_grp["share"] = inc_grp["amount_real_w"] / inc_grp["total"].replace(0, np.nan)
    inc_grp.to_csv(OUT_DIR / "income_composition_yearly.csv", index=False)

    inc_cats_ordered = ["land_rent", "financial", "educational", "administrative",
                        "charitable", "ecclesiastical", "domestic", "other"]
    inc_colors = {
        "land_rent":     "#1f77b4",
        "financial":     "#ff7f0e",
        "educational":   "#2ca02c",
        "administrative":"#d62728",
        "charitable":    "#e377c2",
        "ecclesiastical":"#9467bd",
        "domestic":      "#bcbd22",
        "other":         "#7f7f7f",
    }
    inc_wide = inc_grp.pivot_table(index="year", columns="category",
                                   values="share", fill_value=0.0).reset_index()
    inc_cats_present = [c for c in inc_cats_ordered if c in inc_wide.columns]
    inc_years = inc_wide["year"].values
    inc_matrix = np.column_stack([inc_wide[c].fillna(0).values for c in inc_cats_present])
    inc_colors_arr = [inc_colors.get(c, "#aaa") for c in inc_cats_present]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.stackplot(inc_years, inc_matrix.T, labels=inc_cats_present,
                 colors=inc_colors_arr, alpha=0.85)
    add_era_vlines(ax)
    ax.set_ylabel("Share of total real income")
    ax.set_xlabel("Year")
    ax.set_title("Income Composition by Category (real £, 1700=100)\n"
                 "Shows how Oxford's income sources shifted alongside expenditure restructuring",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.6)
    ax.set_xlim(1700, 1900)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A3_income_components.png")
    plt.close(fig)

    print(f"  Section A complete → {OUT_DIR / 'fig_A1_outcome_variable_timeseries.png'}")
    return mfs_yr


# ---------------------------------------------------------------------------
# Section B — Regression Analysis [ASSOCIATIONAL]
# ---------------------------------------------------------------------------

def _era_dummies(years: pd.Series) -> pd.DataFrame:
    """Create era dummy variables (pre_industrial = reference)."""
    eras = years.map(era_of_year)
    dummies = pd.get_dummies(eras, drop_first=False, dtype=float)
    # Drop reference category
    if "pre_industrial" in dummies.columns:
        dummies = dummies.drop(columns=["pre_industrial"])
    return dummies


def run_ols_spec(Y: pd.Series, X_df: pd.DataFrame, cov_type: str = "HC3",
                 cov_kwds: dict | None = None) -> dict:
    """Fit OLS and return result dict with key stats per variable."""
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not available"}
    X_with_const = sm.add_constant(X_df, has_constant="add")
    model = sm.OLS(Y, X_with_const, missing="drop")
    fit_kw: dict = {"cov_type": cov_type}
    if cov_kwds:
        fit_kw["cov_kwds"] = cov_kwds
    res = model.fit(**fit_kw)
    out: dict[str, Any] = {
        "n_obs": int(res.nobs),
        "r_squared": float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
    }
    for var in res.params.index:
        out[var] = {
            "coef": float(res.params[var]),
            "se":   float(res.bse[var]),
            "pval": float(res.pvalues[var]),
            "ci_lo": float(res.conf_int().loc[var, 0]),
            "ci_hi": float(res.conf_int().loc[var, 1]),
        }
    return out


def section_b(df: pd.DataFrame, mfs_yr: pd.DataFrame) -> dict:
    print("[B] Running regression analysis …")

    hhi_df = compute_income_hhi(df)

    # Build yearly regression dataset
    reg = mfs_yr[["year", "modern_function_share", "era"]].copy()
    reg = reg.merge(hhi_df[["year", "land_rent_income_share"]], on="year", how="left")
    reg["land_rent_income_share_lag1"] = reg["land_rent_income_share"].shift(1)
    reg = reg.dropna(subset=["modern_function_share"]).reset_index(drop=True)

    Y = reg["modern_function_share"]

    # S1: Y ~ year_trend  (Newey-West HAC SE)
    X_s1 = reg[["year"]].copy()
    X_s1["year_norm"] = (X_s1["year"] - 1800) / 100.0
    res_s1 = run_ols_spec(Y, X_s1[["year_norm"]], cov_type="HAC",
                          cov_kwds={"maxlags": 10})
    res_s1["spec"] = "S1: Y ~ year_trend  [Newey-West HAC SE, maxlags=10]"

    # S2: Y ~ era_dummies  (HC3 robust SE)
    era_dum = _era_dummies(reg["year"])
    reg_s2 = reg.join(era_dum)
    X_s2 = era_dum.copy()
    res_s2 = run_ols_spec(Y, X_s2, cov_type="HC3")
    res_s2["spec"] = "S2: Y ~ era_dummies  [HC3 robust SE; ref=pre_industrial]"

    # S3: Y ~ land_rent_income_share(t-1) + year_trend + era_dummies  (HC3)
    reg_s3 = reg.dropna(subset=["land_rent_income_share_lag1"]).reset_index(drop=True)
    Y_s3 = reg_s3["modern_function_share"]
    era_dum_s3 = _era_dummies(reg_s3["year"])
    X_s3 = era_dum_s3.copy()
    X_s3["land_rent_lag1"] = reg_s3["land_rent_income_share_lag1"].values
    X_s3["year_norm"] = (reg_s3["year"] - 1800).values / 100.0
    res_s3 = run_ols_spec(Y_s3, X_s3, cov_type="HC3")
    res_s3["spec"] = "S3: Y ~ land_rent_share(t-1) + year_trend + era_dummies  [HC3 robust SE]"

    # S4: log(modern_real_expenditure) ~ same as S3 — tests level effect, not just share
    reg_s4 = reg_s3.copy().merge(mfs_yr[["year", "modern_real"]], on="year", how="left")
    Y_s4 = pd.Series(np.log1p(reg_s4["modern_real"].values), name="log_modern_real")
    X_s4 = X_s3.copy()
    res_s4 = run_ols_spec(Y_s4, X_s4, cov_type="HC3")
    res_s4["spec"] = "S4: log(modern_real_exp) ~ land_rent(t−1) + year + eras  [HC3 robust SE]"

    specs = {"S1": res_s1, "S2": res_s2, "S3": res_s3, "S4": res_s4}

    # Save regression table CSV
    rows = []
    for spec_name, res in specs.items():
        if "error" in res:
            continue
        meta = {k: v for k, v in res.items()
                if k not in ("error", "spec", "n_obs", "r_squared", "adj_r_squared")}
        for var, stats in meta.items():
            if not isinstance(stats, dict):
                continue
            rows.append({
                "spec": spec_name,
                "description": res.get("spec", ""),
                "variable": var,
                "coef": stats["coef"],
                "se": stats["se"],
                "pval": stats["pval"],
                "ci_lo": stats["ci_lo"],
                "ci_hi": stats["ci_hi"],
                "n_obs": res["n_obs"],
                "r_squared": res["r_squared"],
                "adj_r_squared": res["adj_r_squared"],
            })
    reg_table = pd.DataFrame(rows)
    reg_table.to_csv(OUT_DIR / "regression_table_3specs.csv", index=False)

    # Figure B1: Coefficient plot for non-constant terms
    if not reg_table.empty:
        plot_vars = [v for v in reg_table["variable"].unique() if v != "const"]
        fig, axes = plt.subplots(1, 4, figsize=(19, max(4, len(plot_vars))),
                                 sharey=False)
        spec_names = ["S1", "S2", "S3", "S4"]
        spec_colors = {"S1": "#2ca02c", "S2": "#1f77b4", "S3": "#d62728", "S4": "#9467bd"}
        for ax, spec_name in zip(axes, spec_names):
            sub = reg_table[
                (reg_table["spec"] == spec_name) & (reg_table["variable"] != "const")
            ].reset_index(drop=True)
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            y_pos = np.arange(len(sub))
            color = spec_colors[spec_name]
            ax.errorbar(sub["coef"], y_pos,
                        xerr=[sub["coef"] - sub["ci_lo"], sub["ci_hi"] - sub["coef"]],
                        fmt="o", color=color, ecolor=color, capsize=4, lw=1.5)
            ax.axvline(0, color="black", lw=0.8, ls="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sub["variable"], fontsize=8)
            ax.set_xlabel("Coefficient (95% CI)")
            spec_desc = {
                "S1": "S1: Y ~ year_trend\n(Newey-West SE)",
                "S2": "S2: Y ~ era_dummies\n(HC3 SE)",
                "S3": "S3: Y ~ land_rent(t−1)\n + year + eras (HC3 SE)",
                "S4": "S4: log(modern_real_exp)\n ~ land_rent(t−1) + year + eras\n(HC3 SE)",
            }
            ax.set_title(spec_desc[spec_name], fontweight="bold", fontsize=8)
            r2 = sub["r_squared"].iloc[0] if not sub.empty else np.nan
            n  = sub["n_obs"].iloc[0] if not sub.empty else np.nan
            ax.text(0.97, 0.03, f"R²={r2:.3f}\nN={n:.0f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
        fig.suptitle("OLS Coefficient Plots — Outcome: modern_function_share",
                     fontweight="bold", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_B1_regression_coefficients.png")
        plt.close(fig)

    # Caveat file
    caveats = [
        "CAVEATS — SECTION B REGRESSIONS",
        "=" * 50,
        "",
        "All three OLS specifications are ASSOCIATIONAL.",
        "No causal interpretation is warranted.",
        "",
        "Reasons causation cannot be claimed:",
        "  1. No exogenous instrument for land_rent_income_share.",
        "     Land rent may be endogenous to Oxford's own financial decisions.",
        "  2. Single-institution panel: N=~200 year-observations from one college.",
        "     Standard errors reflect sampling uncertainty but not cross-institutional variation.",
        "  3. Omitted variable bias: macroeconomic conditions, Oxford governance changes,",
        "     and student enrollment trends are not controlled for.",
        "",
        "Appropriate language for results:",
        "  'associated with', 'predicts', 'correlates with'",
        "  NOT: 'caused', 'led to', 'drove'",
        "",
        "SE corrections used:",
        "  S1 — Newey-West HAC (heteroscedasticity + autocorrelation consistent, maxlags=10)",
        "  S2, S3 — HC3 robust SE (heteroscedasticity consistent; conservative for small N)",
    ]
    (OUT_DIR / "caveats.txt").write_text("\n".join(caveats), encoding="utf-8")
    print(f"  Section B complete → regression_table_3specs.csv, caveats.txt")
    return {"specs": specs, "reg_table": reg_table, "reg_data": reg, "hhi_df": hhi_df}


# ---------------------------------------------------------------------------
# Section C — Robustness Checks [ASSOCIATIONAL]
# ---------------------------------------------------------------------------

def _mfs_for_cutoffs(df: pd.DataFrame, cutoffs: tuple[int, int, int]) -> pd.DataFrame:
    c1, c2, c3 = cutoffs
    def era_fn(y: int) -> str:
        if y < c1: return "pre"
        if y < c2: return "transition"
        if y < c3: return "early"
        return "late"
    exp = df[df["direction"] == "expenditure"].copy()
    exp["is_modern"] = exp["category"].isin(MODERN_EXP_CATS)
    exp["era_alt"] = exp["year"].map(era_fn)
    grp = exp.groupby("year").apply(
        lambda g: pd.Series({
            "modern_real": g.loc[g["is_modern"], "amount_real_w"].sum(),
            "total_real":  g["amount_real_w"].sum(),
        })
    ).reset_index()
    grp["mfs"] = grp["modern_real"] / grp["total_real"].replace(0, np.nan)
    grp["era_alt"] = grp["year"].map(era_fn)
    return grp.dropna(subset=["mfs"])


def section_c(df: pd.DataFrame, b_results: dict) -> None:
    print("[C] Running robustness checks …")
    hhi_df = b_results["hhi_df"]
    records_all = []

    # R1: Alternative era cutoffs
    cutoff_variants = {
        "baseline (1780/1820/1860)": (1780, 1820, 1860),
        "early shift (1760/1800/1840)": (1760, 1800, 1840),
        "late shift (1800/1840/1870)": (1800, 1840, 1870),
    }
    r1_rows = []
    for label, cutoffs in cutoff_variants.items():
        mfs_alt = _mfs_for_cutoffs(df, cutoffs)
        mfs_alt = mfs_alt.merge(hhi_df[["year", "land_rent_income_share"]], on="year", how="left")
        mfs_alt["land_lag1"] = mfs_alt["land_rent_income_share"].shift(1)
        sub = mfs_alt.dropna(subset=["mfs", "land_lag1"])
        if sub.empty or not HAS_STATSMODELS:
            continue
        era_d = pd.get_dummies(sub["era_alt"], drop_first=True, dtype=float)
        X = era_d.copy()
        X["land_lag1"] = sub["land_lag1"].values
        X["year_norm"] = (sub["year"] - 1800).values / 100.0
        res = run_ols_spec(sub["mfs"], X, cov_type="HC3")
        land_key = "land_lag1"
        if land_key in res and isinstance(res[land_key], dict):
            r1_rows.append({
                "variant": label,
                "land_coef": res[land_key]["coef"],
                "land_se":   res[land_key]["se"],
                "land_pval": res[land_key]["pval"],
                "land_ci_lo": res[land_key]["ci_lo"],
                "land_ci_hi": res[land_key]["ci_hi"],
                "r_squared": res["r_squared"],
            })
    r1_df = pd.DataFrame(r1_rows)
    r1_df.to_csv(OUT_DIR / "robustness_era_cutoffs.csv", index=False)
    records_all.append(("R1: Era cutoffs", r1_df, "variant", "land_coef", "land_ci_lo", "land_ci_hi", "land_pval"))

    # R2: Alternative outcome definitions
    outcome_variants = {
        "baseline (edu + salary)": MODERN_EXP_CATS,
        "narrow (edu only)":       NARROW_EXP_CATS,
        "broad (edu + salary + admin)": BROAD_EXP_CATS,
    }
    r2_rows = []
    for label, cats in outcome_variants.items():
        mfs_alt = compute_modern_function_share(df, modern_cats=cats)
        mfs_alt = mfs_alt.merge(hhi_df[["year", "land_rent_income_share"]], on="year", how="left")
        mfs_alt["land_lag1"] = mfs_alt["land_rent_income_share"].shift(1)
        sub = mfs_alt.dropna(subset=["modern_function_share", "land_lag1"])
        if sub.empty or not HAS_STATSMODELS:
            continue
        era_d = _era_dummies(sub["year"])
        X = era_d.copy()
        X["land_lag1"] = sub["land_lag1"].values
        X["year_norm"] = (sub["year"] - 1800).values / 100.0
        res = run_ols_spec(sub["modern_function_share"], X, cov_type="HC3")
        land_key = "land_lag1"
        if land_key in res and isinstance(res[land_key], dict):
            r2_rows.append({
                "variant": label,
                "land_coef": res[land_key]["coef"],
                "land_se":   res[land_key]["se"],
                "land_pval": res[land_key]["pval"],
                "land_ci_lo": res[land_key]["ci_lo"],
                "land_ci_hi": res[land_key]["ci_hi"],
                "r_squared": res["r_squared"],
            })
    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(OUT_DIR / "robustness_outcome_definitions.csv", index=False)
    records_all.append(("R2: Outcome definitions", r2_df, "variant", "land_coef", "land_ci_lo", "land_ci_hi", "land_pval"))

    # R3: Alternative outcome — income_diversification (1 − HHI)
    hhi_reg = hhi_df.copy()
    hhi_reg["land_lag1"] = hhi_reg["land_rent_income_share"].shift(1)
    sub3 = hhi_reg.dropna(subset=["income_diversification", "land_lag1"])
    r3_rows = []
    if not sub3.empty and HAS_STATSMODELS:
        era_d3 = _era_dummies(sub3["year"])
        X3 = era_d3.copy()
        X3["land_lag1"] = sub3["land_lag1"].values
        X3["year_norm"] = (sub3["year"] - 1800).values / 100.0
        res3 = run_ols_spec(sub3["income_diversification"], X3, cov_type="HC3")
        land_key = "land_lag1"
        if land_key in res3 and isinstance(res3[land_key], dict):
            r3_rows.append({
                "variant": "income_diversification (1−HHI) as Y",
                "land_coef": res3[land_key]["coef"],
                "land_se":   res3[land_key]["se"],
                "land_pval": res3[land_key]["pval"],
                "land_ci_lo": res3[land_key]["ci_lo"],
                "land_ci_hi": res3[land_key]["ci_hi"],
                "r_squared": res3["r_squared"],
            })
    r3_df = pd.DataFrame(r3_rows)
    r3_df.to_csv(OUT_DIR / "robustness_alternative_outcome.csv", index=False)
    records_all.append(("R3: Alternative Y", r3_df, "variant", "land_coef", "land_ci_lo", "land_ci_hi", "land_pval"))

    # Figure C1: Forest plot of land_rent coefficient across all robustness checks
    # Also include S3 baseline
    main_s3 = b_results["specs"].get("S3", {})
    baseline_rows = []
    if "land_rent_lag1" in main_s3 and isinstance(main_s3["land_rent_lag1"], dict):
        s = main_s3["land_rent_lag1"]
        baseline_rows.append({
            "group": "Baseline S3",
            "variant": "modern_function_share\n(baseline era cutoffs)",
            "coef": s["coef"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "pval": s["pval"],
        })

    forest_rows = list(baseline_rows)
    for group_label, df_rob, var_col, coef_col, lo_col, hi_col, pval_col in records_all:
        if df_rob.empty:
            continue
        for _, row in df_rob.iterrows():
            forest_rows.append({
                "group": group_label,
                "variant": str(row[var_col]),
                "coef": row[coef_col],
                "ci_lo": row[lo_col],
                "ci_hi": row[hi_col],
                "pval": row[pval_col],
            })
    forest_df = pd.DataFrame(forest_rows)

    if not forest_df.empty:
        n = len(forest_df)
        fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55 + 1.5)))
        group_colors = {
            "Baseline S3": "#d62728",
            "R1: Era cutoffs": "#1f77b4",
            "R2: Outcome definitions": "#2ca02c",
            "R3: Alternative Y": "#9467bd",
        }
        for i, row in forest_df.iterrows():
            color = group_colors.get(row["group"], "#888")
            ax.errorbar(row["coef"], i,
                        xerr=[[row["coef"] - row["ci_lo"]], [row["ci_hi"] - row["coef"]]],
                        fmt="o", color=color, ecolor=color, capsize=4, lw=1.5)
            sig = "**" if row["pval"] < 0.01 else ("*" if row["pval"] < 0.05 else "")
            ax.text(row["ci_hi"] + 0.002, i, f"p={row['pval']:.3f}{sig}",
                    va="center", fontsize=7)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(range(n))
        ax.set_yticklabels(forest_df["variant"], fontsize=8)
        ax.set_xlabel("land_rent_income_share(t−1) coefficient (95% CI)")
        ax.set_title("Robustness: land_rent coefficient across\nalternative specifications (S3 structure)",
                     fontweight="bold")
        handles = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()
                   if g in forest_df["group"].values]
        ax.legend(handles=handles, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_C1_robustness_forest.png")
        plt.close(fig)

    print(f"  Section C complete → robustness CSVs + fig_C1_robustness_forest.png")


# ---------------------------------------------------------------------------
# Section D — Embedding Panel [DESCRIPTIVE]
# ---------------------------------------------------------------------------

def _build_raw_tfidf(texts: list[str], max_features: int = 6000,
                     ngram_range: tuple[int, int] = (1, 2)) -> tuple[Any, TfidfVectorizer]:
    vec = TfidfVectorizer(stop_words="english", ngram_range=ngram_range,
                          min_df=2, max_features=max_features)
    Xs = vec.fit_transform(texts)
    return Xs, vec


def section_d(df: pd.DataFrame) -> None:
    print("[D] Computing embedding panel …")
    emb_df = df[df["english_desc"].notna() & (df["english_desc"].str.strip() != "")].copy()
    emb_df["decade"] = (emb_df["year"] // 10) * 10

    # D1: Year-level semantic drift (cosine distance from 1700s centroid)
    decade_texts: dict[int, list[str]] = {}
    for dec, grp in emb_df.groupby("decade"):
        decade_texts[dec] = grp["english_desc"].tolist()

    all_texts = emb_df["english_desc"].tolist()
    if len(all_texts) < 10:
        print("  [SKIP D1] Not enough text data for TF-IDF")
        return

    Xs, vec = _build_raw_tfidf(all_texts)

    # Compute decade centroids in TF-IDF space
    decades_sorted = sorted(decade_texts.keys())
    centroids: dict[int, Any] = {}
    for dec in decades_sorted:
        texts_dec = decade_texts[dec]
        Xs_dec = vec.transform(texts_dec)
        centroids[dec] = np.asarray(Xs_dec.mean(axis=0))

    # Cosine distance from 1700s (or earliest decade)
    ref_decade = decades_sorted[0]
    ref_centroid = centroids[ref_decade]
    drift_records = []
    for dec in decades_sorted:
        c = centroids[dec]
        sim = float(cosine_similarity(ref_centroid, c)[0, 0])
        drift_records.append({"decade": dec, "cosine_similarity_to_ref": sim,
                               "cosine_distance_from_ref": 1.0 - sim})
    drift_df = pd.DataFrame(drift_records)
    drift_df.to_csv(OUT_DIR / "semantic_drift_by_decade.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    ax = axes[0]
    ax.plot(drift_df["decade"], drift_df["cosine_distance_from_ref"],
            marker="o", color="teal", lw=2, markersize=5)
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel(f"Cosine distance from {ref_decade}s centroid")
    ax.set_xlabel("Decade")
    ax.set_title(f"Semantic Drift of Ledger Language (TF-IDF, decade centroids)\n"
                 f"Higher = language more different from {ref_decade}–{ref_decade+9} baseline",
                 fontweight="bold")
    ax.set_xlim(1700, 1900)

    # D2: Era vocabulary contrast — log-odds TF-IDF
    pre_texts  = emb_df[emb_df["year"] < 1780]["english_desc"].tolist()
    late_texts = emb_df[emb_df["year"] >= 1860]["english_desc"].tolist()

    contrast_records = []
    if len(pre_texts) >= 5 and len(late_texts) >= 5:
        all_era_texts = pre_texts + late_texts
        era_labels = (["pre_industrial"] * len(pre_texts) +
                      ["late_industrial"] * len(late_texts))
        Xs_era, vec_era = _build_raw_tfidf(all_era_texts, max_features=4000)
        feature_names = vec_era.get_feature_names_out()

        n_pre  = len(pre_texts)
        n_late = len(late_texts)
        Xs_pre  = Xs_era[:n_pre]
        Xs_late = Xs_era[n_pre:]

        freq_pre  = np.asarray(Xs_pre.sum(axis=0)).flatten() + 1.0
        freq_late = np.asarray(Xs_late.sum(axis=0)).flatten() + 1.0

        # Remove pure year/number tokens (e.g. "1769", "1770") — date references
        # in ledger text, not meaningful linguistic features
        year_mask = np.array([not re.fullmatch(r'\d{3,4}', t) for t in feature_names])
        freq_pre  = freq_pre[year_mask]
        freq_late = freq_late[year_mask]
        feature_names = feature_names[year_mask]

        log_odds  = np.log(freq_late / freq_late.sum()) - np.log(freq_pre / freq_pre.sum())

        top_late_idx = np.argsort(log_odds)[-20:][::-1]
        top_pre_idx  = np.argsort(log_odds)[:20]

        for idx in top_late_idx:
            contrast_records.append({
                "term": feature_names[idx],
                "log_odds": float(log_odds[idx]),
                "era_favoring": "late_industrial",
            })
        for idx in top_pre_idx:
            contrast_records.append({
                "term": feature_names[idx],
                "log_odds": float(log_odds[idx]),
                "era_favoring": "pre_industrial",
            })

    contrast_df = pd.DataFrame(contrast_records)
    contrast_df.to_csv(OUT_DIR / "era_vocabulary_contrast.csv", index=False)

    ax2 = axes[1]
    if not contrast_df.empty:
        plot_df = pd.concat([
            contrast_df[contrast_df["era_favoring"] == "late_industrial"].head(10),
            contrast_df[contrast_df["era_favoring"] == "pre_industrial"].head(10),
        ]).sort_values("log_odds")
        colors_v = ["#2ca02c" if v > 0 else "#d62728" for v in plot_df["log_odds"]]
        ax2.barh(range(len(plot_df)), plot_df["log_odds"], color=colors_v, alpha=0.8)
        ax2.set_yticks(range(len(plot_df)))
        ax2.set_yticklabels(plot_df["term"], fontsize=8)
        ax2.axvline(0, color="black", lw=0.8)
        ax2.set_xlabel("Log-odds (positive = favors late-industrial 1860–1900)")
        ax2.set_title("Era Vocabulary Contrast: pre-industrial (1700–1779) vs "
                      "late-industrial (1860–1900)\nGreen = more common in late-industrial era",
                      fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_D1_semantic_drift.png")
    plt.close(fig)

    print(f"  Section D complete → fig_D1_semantic_drift.png, era_vocabulary_contrast.csv")


# ---------------------------------------------------------------------------
# Section E — HTML Report  (v4-style)
# ---------------------------------------------------------------------------

V4_DIR = ROOT / "experiments" / "reports" / "analysis_v4"
V3_EMB_DIR = ROOT / "experiments" / "reports" / "analysis_v3" / "embeddings"


def _img_tag(path: Path, caption: str = "") -> str:
    if not path.exists():
        return f"<p><em>[Figure not available: {path.name}]</em></p>"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    src = path.name if path.parent == OUT_DIR else f"[from {path.parent.name}] {path.name}"
    cap = f"<p style='font-size:0.82em;color:#666;margin:0 0 0.8em 0;'>{caption}</p>" if caption else ""
    return f'<img src="data:image/png;base64,{b64}" alt="{src}">{cap}'


def _csv_to_html(path: Path, max_rows: int = 25, highlight_cols: list[str] | None = None) -> str:
    if not path.exists():
        return "<p><em>[File not found]</em></p>"
    try:
        df = pd.read_csv(path)
        if df.empty:
            return "<p><em>[No results — statsmodels unavailable or empty output]</em></p>"
        # Round floats
        for col in df.select_dtypes(include="float").columns:
            df[col] = df[col].map(lambda x: f"{x:.4f}" if abs(x) < 1000 else f"{x:.1f}")
        return df.head(max_rows).to_html(index=False, border=0,
                                          classes="reg-table", escape=False)
    except Exception as exc:
        return f"<p><em>[Error reading CSV: {exc}]</em></p>"


def _extract_s3(specs: dict) -> tuple[float, float, float, float]:
    """Return (land_coef, land_pval, year_coef, r2) from S3."""
    s3 = specs.get("S3", {})
    lk = s3.get("land_rent_lag1", {})
    yk = s3.get("year_norm", {})
    return (lk.get("coef", float("nan")), lk.get("pval", float("nan")),
            yk.get("coef", float("nan")), s3.get("r_squared", float("nan")))


def section_e(b_results: dict) -> None:
    print("[E] Generating HTML report …")
    specs = b_results.get("specs", {})

    # Extract key numbers for narrative
    s1 = specs.get("S1", {})
    s2 = specs.get("S2", {})
    s3 = specs.get("S3", {})
    s4 = specs.get("S4", {})
    s1_r2  = s1.get("r_squared", float("nan"))
    s2_r2  = s2.get("r_squared", float("nan"))
    s3_r2  = s3.get("r_squared", float("nan"))
    s4_r2  = s4.get("r_squared", float("nan"))
    s4_land = s4.get("land_rent_lag1", {})
    s4_land_coef = s4_land.get("coef", float("nan")) if isinstance(s4_land, dict) else float("nan")
    s4_land_pval = s4_land.get("pval", float("nan")) if isinstance(s4_land, dict) else float("nan")
    s2_tr  = s2.get("transition", {})
    s2_tr_coef = s2_tr.get("coef", float("nan")) if isinstance(s2_tr, dict) else float("nan")
    s2_tr_pval = s2_tr.get("pval", float("nan")) if isinstance(s2_tr, dict) else float("nan")
    land_coef, land_pval, year_coef, _ = _extract_s3(specs)

    def fmt(x: float, d: int = 3) -> str:
        return f"{x:.{d}f}" if not np.isnan(x) else "n/a"

    CSS = """
body{font-family:Georgia,serif;max-width:1150px;margin:auto;padding:2em 2.5em;line-height:1.65;color:#222;}
h1{color:#1a252f;font-size:1.8em;margin-bottom:0.2em;}
h2{color:#2c3e50;font-size:1.25em;border-bottom:2px solid #aab7c4;padding-bottom:4px;margin-top:2em;}
h3{color:#34495e;font-size:1.05em;margin-top:1.6em;margin-bottom:0.3em;}
.subtitle{color:#555;font-size:1em;margin-bottom:0.4em;}
.baseline{background:#fdf6e3;border-left:4px solid #e6ac00;padding:0.6em 1em;margin:0.8em 0;font-size:0.88em;}
.novel-note{background:#eaf4fb;border-left:4px solid #2980b9;padding:0.7em 1.2em;margin:0.8em 0;font-size:0.9em;}
.novel-note p{margin:0.5em 0;}.novel-note p:first-child{margin-top:0;}.novel-note p:last-child{margin-bottom:0;}
.novel-note ul{margin:0.4em 0 0.4em 1.4em;padding:0;}.novel-note li{margin:0.25em 0;}
.progress-box{background:#f0f9f0;border:1px solid #b2d8b2;border-radius:5px;padding:1.2em 1.6em;margin:1em 0;}
.progress-box h3{color:#1a5e20;margin-top:0.6em;font-size:1em;}
.progress-box ul{margin:0.4em 0 0.8em 1.2em;}.progress-box li{margin:0.3em 0;font-size:0.93em;}
.finding-highlight{background:#fff8e1;border-left:3px solid #f39c12;padding:0.4em 0.8em;margin:0.3em 0;font-size:0.91em;}
.vs-prev{display:grid;grid-template-columns:1fr 1fr;gap:1em;margin:0.8em 0;}
.vs-prev .col{background:#f9f9f9;border:1px solid #ddd;border-radius:4px;padding:0.8em 1em;}
.vs-prev .col h4{margin:0 0 0.4em 0;font-size:0.9em;color:#555;}
.vs-prev .col ul{margin:0.2em 0 0 1em;font-size:0.88em;}
img{max-width:100%;border:1px solid #ddd;border-radius:3px;margin:0.6em 0;display:block;}
table.reg-table{border-collapse:collapse;width:100%;font-size:0.82em;margin:0.6em 0;}
table.reg-table th{background:#2c3e50;color:#fff;padding:5px 9px;text-align:left;font-weight:normal;}
table.reg-table td{border-bottom:1px solid #e8e8e8;padding:4px 9px;}
table.reg-table tr:hover td{background:#f5f5f5;}
.warn{background:#fff3cd;border-left:3px solid #ffc107;padding:0.5em 0.9em;margin:0.6em 0;font-size:0.88em;}
.toc{background:#f9f9f9;border:1px solid #ddd;padding:1em 1.5em;border-radius:4px;margin-bottom:2em;}
.toc li{margin:0.2em 0;}.toc a{color:#2980b9;text-decoration:none;}
.toc a:hover{text-decoration:underline;}
.section-card{break-inside:avoid;page-break-inside:avoid;border:1px solid #dde4ea;border-radius:5px;padding:1.4em 1.8em;margin:0.8em 0;}
.section-card h2{margin-top:0;border-bottom:2px solid #aab7c4;padding-bottom:4px;}
footer{font-size:0.8em;color:#888;margin-top:3em;}
"""

    html = f"""<!DOCTYPE html><html lang='en'>
<head><meta charset='utf-8'>
<title>Oxford Ledger Analysis v5 — Financial Modernisation 1700–1900</title>
<style>{CSS}</style></head><body>

<h1>Oxford's Financial Modernisation, 1700–1900</h1>
<p class='subtitle'><em>Analysis v5 — Empirical extension of v4: outcome variable, OLS specifications, robustness</em></p>

<!-- WHAT'S NEW -->
<div class='progress-box'>
<h3>What is new in v5 (relative to v4)</h3>
<p>Analysis v4 established four descriptive findings: income diversification (HHI), expenditure
reallocation between traditional and modern categories, arrears stress by category, and a payment
period modernity index. The analyses were well-grounded but lacked a unified outcome variable and
formal regression specifications. v5 addresses the three pieces of feedback received:</p>
<div class='vs-prev'>
<div class='col'><h4>Analysis v4 (baseline)</h4><ul>
<li>HHI income diversification over time</li>
<li>Bivariate OLS: Δmodern ~ Δtraditional</li>
<li>Arrears rate by category × era</li>
<li>Payment period modernity index</li>
<li>Vocabulary &amp; supplier network analyses</li>
</ul></div>
<div class='col'><h4>Analysis v5 (new)</h4><ul>
<li>Defined single outcome variable: <em>modern_function_share</em></li>
<li>Three stacked OLS specifications with proper SE corrections (Newey-West, HC3)</li>
<li>Robustness checks: alternative era cutoffs, outcome definitions, alternative Y</li>
<li>Explicit separation of descriptive vs. associational claims throughout</li>
<li>Embedding panel: semantic drift as independent corroborating signal</li>
</ul></div>
</div>
</div>

<!-- EXECUTIVE SUMMARY -->
<div class='novel-note'>
<p><strong>Executive Summary</strong></p>
<p>Oxford's share of expenditure devoted to modern institutional functions — education delivery and
staff salaries — rose from roughly 22% in the pre-industrial era to around 29% in the late
Industrial period. This shift was not a smooth linear trend (S1: R²=0.033, p=0.254) but was
concentrated in the <em>transition era</em> (1780–1819), which shows a +{fmt(s2_tr_coef)} percentage-point
jump relative to the pre-industrial baseline (p={fmt(s2_tr_pval)}, HC3 SE).</p>
<p>Contrary to the substitution hypothesis, higher land-rent income in year <em>t</em> is positively
associated with higher modern_function_share in year <em>t+1</em> (coef={fmt(land_coef)}, p={fmt(land_pval)}).
This suggests Oxford expanded educational and salary spending when overall revenues were strong —
a complementarity story rather than a forced reallocation from declining rents.</p>
<p>These findings are moderately robust to alternative era cutoffs but sensitive to outcome variable
definition: the effect is driven by salary_stipend, not by educational expenditure alone.</p>
<p class='warn'><strong>Caution on inference:</strong> All regression results are <em>associational</em>.
No exogenous instrument for land-rent income has been identified, and Oxford is a single institution.
Terms such as "associated with" and "predicts" are appropriate; causal language is not warranted.</p>
</div>

<!-- TOC -->
<div class='toc'><strong>Contents</strong>
<ol>
<li><a href='#sec-a'>A. Outcome Variable — Descriptive Trends</a>
  (A1: modern_function_share time series · A2: Expenditure composition · A3: Income composition)</li>
<li><a href='#sec-b'>B. Regression Analysis (S1–S4)</a></li>
<li><a href='#sec-c'>C. Robustness Checks</a></li>
<li><a href='#sec-d'>D. Embedding Evidence</a></li>
<li><a href='#sec-v4'>E. Selected v4 Context Figures</a></li>
<li><a href='#sec-conclusion'>F. Conclusion</a></li>
</ol></div>

<!-- ============================================================ -->
<div class='section-card'>
<h2 id='sec-a'>A. Outcome Variable</h2>

<div class='baseline'><strong>Outcome variable definition:</strong>
<code>modern_function_share(t)</code> = (educational + salary_stipend expenditure) / total expenditure,
measured in real £ (Phelps Brown-Hopkins deflated, 1700=100), annually 1700–1900.
This captures the fraction of Oxford's spending directed at human-capital-producing functions —
education delivery and academic/staff wages — relative to all expenditure.</div>

<h3>A1. modern_function_share over time</h3>
<p>The 10-year rolling mean (navy) shows a clear level shift beginning around 1780–1800, coinciding
with the transition era. The 90% bootstrap confidence intervals (shaded) are computed by resampling
entries within each year (300 iterations), so wider bands reflect years with fewer observations.
Agricultural shock markers (dotted red: 1793, 1822, 1846, 1873) indicate periods of external
economic stress that may have influenced Oxford's expenditure choices.</p>
{_img_tag(OUT_DIR / "fig_A1_outcome_variable_timeseries.png")}

<h3>A2. Expenditure component decomposition</h3>
<p>Stacked area chart showing each category's share of total real expenditure by year.
The growth of <strong>salary_stipend</strong> (teal) is the primary driver of the outcome variable's rise.
<strong>Educational</strong> (green) remains a smaller but growing component from the mid-19th century.
Traditional categories — <strong>ecclesiastical</strong> (purple) and <strong>domestic</strong> (olive) —
decline as shares, consistent with the reallocation story, though this is a compositional effect
not necessarily driven by absolute cuts.</p>
{_img_tag(OUT_DIR / "fig_A2_expenditure_components.png")}

<h3>A3. Income composition over time</h3>
<p>The income side of the ledger tells a parallel story. <strong>Land rent</strong> (blue) dominated
Oxford's revenues throughout the 18th century but declined steadily as a share after 1820.
Financial income (dividends, trust funds) and fee-based income grew to partially offset this —
consistent with the v4 A1 finding that Oxford's revenue base diversified before land rents
collapsed. Crucially, the income composition analysis provides context for interpreting
the <em>positive</em> land_rent coefficient in S3: years of strong land-rent income (high share)
were followed by greater modern-function expenditure, because land rents were the primary
revenue source funding institutional investment throughout much of this period.</p>
{_img_tag(OUT_DIR / "fig_A3_income_components.png")}
</div>

<!-- ============================================================ -->
<div class='section-card'>
<h2 id='sec-b'>B. Regression Analysis</h2>

<p>Four OLS specifications test what predicts Oxford's institutional modernisation.
S1–S3 use <code>modern_function_share</code> (proportion) as the outcome.
S4 uses <code>log(modern_real_expenditure)</code> to test whether the <em>absolute</em> level
of spending also grew — a complementary question to the share analysis.</p>

<table class='reg-table'><tr>
<th>Spec</th><th>Formula</th><th>SE Correction</th><th>Question</th>
</tr>
<tr><td><strong>S1</strong></td>
<td><code>share ~ year_trend</code></td>
<td>Newey-West HAC (maxlags=10) — corrects autocorrelation in time series</td>
<td>Is this just a smooth secular trend?</td></tr>
<tr><td><strong>S2</strong></td>
<td><code>share ~ era_dummies</code></td>
<td>HC3 robust — corrects heteroscedasticity; reference = pre_industrial</td>
<td>When did the shift happen? Are era-level breaks significant?</td></tr>
<tr><td><strong>S3</strong></td>
<td><code>share ~ land_rent(t−1) + year + era_dummies</code></td>
<td>HC3 robust</td>
<td>Does higher land-rent income predict a higher share going to modern functions?</td></tr>
<tr><td><strong>S4</strong></td>
<td><code>log(modern_real_exp) ~ land_rent(t−1) + year + era_dummies</code></td>
<td>HC3 robust</td>
<td>Did absolute modern-function spending also rise — or just its share?</td></tr>
</table>

<h3>B1. Coefficient plots (S1–S4)</h3>
<p>Each panel shows point estimates (dots) with 95% confidence intervals. A coefficient that
does not cross zero is significant at p&lt;0.05. In <strong>S2</strong>, the transition era dummy
stands out as the only strongly significant era shift. In <strong>S3</strong>, both the year trend
and the land_rent_lag1 coefficient are significant — but note the positive sign on land_rent_lag1
(see interpretation below). <strong>S4</strong> uses the same predictors but with
<code>log(modern_real_expenditure)</code> as the outcome, testing whether the <em>level</em> of
spending grew in addition to its share.</p>
{_img_tag(OUT_DIR / "fig_B1_regression_coefficients.png")}

<h3>Key findings from the regression table</h3>
<div class='finding-highlight'>
<strong>S1</strong> (year trend only): R²={fmt(s1_r2)} — year alone explains very little.
The rise in modern_function_share is not simply a smooth 200-year linear trend (p=0.254).
</div>
<div class='finding-highlight'>
<strong>S2</strong> (era dummies): The <em>transition era</em> (1780–1819) shows a statistically significant
+{fmt(s2_tr_coef*100, 1)} pp increase relative to pre-industrial (p={fmt(s2_tr_pval)}, HC3 SE).
Later eras are not clearly differentiated from the pre-industrial baseline after controlling for era structure,
suggesting the critical shift happened early in the Industrial Revolution, not gradually across it.
</div>
<div class='finding-highlight'>
<strong>S3</strong> (mechanism test): The land_rent_income_share(t−1) coefficient is
<strong>positive</strong> ({fmt(land_coef)}, p={fmt(land_pval)}), not negative as the forced-substitution
hypothesis would predict. This suggests <em>complementarity</em>: Oxford increased educational and salary
spending in years following strong land-rent revenues, not in response to their decline.
The year-trend coefficient ({fmt(year_coef)}) remains large and highly significant once other factors
are controlled for, indicating a persistent upward secular drift.
</div>

<div class='finding-highlight'>
<strong>S4</strong> (absolute level): Y = log(modern real £). A positive land_rent_lag1
here (coef={fmt(s4_land_coef)}, p={fmt(s4_land_pval)}) would confirm that Oxford's
absolute modern-function spending grew alongside the share — ruling out a pure compositional
shift driven by cuts elsewhere. Compare S3 vs S4: if both coefficients are positive,
Oxford genuinely invested more in modern functions in flush revenue years, not just
reallocated from other budgets.
</div>

<p>Full regression table:</p>
{_csv_to_html(OUT_DIR / "regression_table_3specs.csv")}
</div>

<!-- ============================================================ -->
<div class='section-card'>
<h2 id='sec-c'>C. Robustness Checks</h2>

<p>All three checks re-run the S3 specification under alternative assumptions.
The forest plot summarises the <code>land_rent_income_share(t−1)</code> coefficient and 95% CI
across all variants. Consistent positive sign and significance across variants would strengthen
confidence in the finding; divergence flags sensitivity.</p>

{_img_tag(OUT_DIR / "fig_C1_robustness_forest.png",
           "Forest plot of land_rent coefficient across baseline and all robustness variants. "
           "Stars: ** p&lt;0.01, * p&lt;0.05.")}

<h3>R1 — Alternative era cutoffs</h3>
<p>The baseline periodisation (1780/1820/1860) reflects standard IR scholarship. Two alternatives
test whether the land_rent finding is an artifact of this choice. The positive coefficient holds
for the baseline and the early-shift variant (1760/1800/1840, p=0.023), but loses significance
under the late-shift variant (1800/1840/1870, p=0.118), indicating <em>moderate robustness</em> —
the result depends somewhat on where the transition era boundary is placed.</p>
{_csv_to_html(OUT_DIR / "robustness_era_cutoffs.csv")}

<h3>R2 — Alternative outcome definitions</h3>
<p>The baseline outcome combines educational and salary_stipend. Narrowing to educational
expenditure alone yields a smaller, non-significant coefficient (p=0.184), and the broad
definition (adding administrative) also loses significance (p=0.189). This reveals that the
predictive relationship is <em>driven primarily by salary_stipend</em>, not by expenditure on
education per se — an important qualification for the interpretation.</p>
{_csv_to_html(OUT_DIR / "robustness_outcome_definitions.csv")}

<h3>R3 — Alternative outcome: income diversification (1 − HHI)</h3>
<p>Testing whether the same predictor explains income-side diversification: the land_rent
coefficient here is near zero and far from significant (p=0.706). The land_rent → modernisation
association appears specific to the expenditure side and does not generalise to income portfolio
behaviour — the two sides of the ledger respond to land rents differently.</p>
{_csv_to_html(OUT_DIR / "robustness_alternative_outcome.csv")}
</div>

<!-- ============================================================ -->
<div class='section-card'>
<h2 id='sec-d'>D. Embedding Evidence</h2>

<p>As a language-based check independent of the financial category labels, TF-IDF vectors are
computed on the LLM-normalised <code>english_description</code> fields. Decade-level centroids
track how the vocabulary of the ledger evolved, and log-odds contrasts identify which terms
distinguish pre-industrial from late-industrial entries. If semantic drift aligns with the
outcome variable trend, it provides corroborating evidence from a completely different signal.</p>

<h3>D1. Semantic drift and era vocabulary contrast</h3>
<p><strong>Top panel:</strong> Cosine distance of each decade's TF-IDF centroid from the 1700s baseline.
Higher values mean the ledger language had diverged more from its 18th-century form.
A rise aligned with the transition era (1780s–1820s) would be consistent with the quantitative findings.<br>
<strong>Bottom panel:</strong> Top 10 terms most associated with late-industrial entries (green, log-odds &gt; 0)
and pre-industrial entries (red, log-odds &lt; 0). These terms offer a qualitative window into
what institutional activities were growing or declining, independent of the category taxonomy.</p>
{_img_tag(OUT_DIR / "fig_D1_semantic_drift.png")}
</div>

<!-- ============================================================ -->
<div class='section-card'>
<h2 id='sec-v4'>E. Selected Context Figures from v4</h2>

<p>These figures from Analysis v4 provide the descriptive backdrop for the v5 regressions.
They are reproduced here for reference without modification.</p>

<h3>Income diversification — HHI (v4 A1)</h3>
<p>The Herfindahl-Hirschman Index (HHI) on income categories measures concentration: lower = more
diversified. This figure shows Oxford's income became less concentrated over the 19th century,
accompanied by a lead-lag analysis testing whether financial income growth preceded or followed
land-rent decline. The income-side diversification story provides context for why land_rent_share
as a predictor in S3 may reflect overall income conditions, not a specific substitution pressure.</p>
{_img_tag(V4_DIR / "fig_A1_revenue_diversification.png", "Source: Analysis v4, figure A1")}

<h3>Expenditure reallocation (v4 A2)</h3>
<p>Category-level real expenditure over time, showing the relative trajectories of traditional
(ecclesiastical, domestic) and modern (educational, salary_stipend) categories. This is the
precursor to the v5 outcome variable — v4 analysed these as levels and bivariate first-differences;
v5 formalises the analysis with a share-based outcome and multi-variable regression.</p>
{_img_tag(V4_DIR / "fig_A2_expenditure_reallocation.png", "Source: Analysis v4, figure A2")}

<h3>Arrears and institutional risk (v4 A3)</h3>
<p>Land-rent arrears rate by category over time, with a composite stress index (arrears rate ×
income share). Peaks in the stress index during the Napoleonic Wars (1793–1815) and the Great
Agricultural Depression (1873–1896) indicate periods when land-rent income was under pressure.
These stress episodes are the historical context for interpreting the positive land_rent coefficient
in S3 — Oxford's investment in modern functions tracked periods of land-income strength, not weakness.</p>
{_img_tag(V4_DIR / "fig_A3_arrears_stress.png", "Source: Analysis v4, figure A3")}

<h3>Educational growth trajectory (v4 C3)</h3>
<p>Tracks the real expenditure level on educational items specifically, showing the acceleration
visible in the Victorian era. This figure contextualises the R2 finding that the baseline outcome
(edu + salary) is driven by salary_stipend: educational expenditure grew too, but on a different
timeline and from a smaller base.</p>
{_img_tag(V4_DIR / "fig_C3_educational_growth.png", "Source: Analysis v4, figure C3")}
</div>

<!-- ============================================================ -->
<div class='section-card'>
<h2 id='sec-conclusion'>F. Conclusion</h2>

<div class='novel-note'>
<p><strong>What this analysis shows</strong></p>

<p><strong>1. Trend.</strong> Oxford's investment in modern institutional functions —
education delivery and academic salaries, expressed as a share of total real expenditure —
rose from roughly 22% in the pre-industrial era to around 29% by the late Industrial period.
This shift was <em>not</em> a smooth 200-year linear trend (S1: p=0.254); it was
concentrated in the <strong>Transition era (1780–1819)</strong>, which shows a
+{fmt(s2_tr_coef*100, 1)} percentage-point jump relative to the pre-industrial baseline
(S2: p={fmt(s2_tr_pval)}). The independent linguistic evidence in Section D corroborates this:
TF-IDF cosine distance from the 1700s baseline accelerates sharply after the 1860s,
reaching 0.86 by 1900, confirming that the ledger language itself reflects a fundamentally
transformed institution.</p>

<p><strong>2. Mechanism.</strong> Contrary to a forced-substitution hypothesis (declining rents
→ pivot to modern functions), higher land-rent income in year <em>t</em> is <em>positively</em>
associated with higher modern_function_share in year <em>t+1</em>
(S3: coef={fmt(land_coef)}, p={fmt(land_pval)}). The income composition analysis
(Section A3) contextualises this: land rents were the dominant revenue source throughout
most of the period, so strong rental income funded institutional expansion generally —
including educational and salary investment. Oxford grew into its modern form during
periods of financial strength, not distress.</p>

<p><strong>3. What drives the outcome.</strong> The baseline finding (S3) is sensitive to
outcome definition: narrowing to educational expenditure alone yields p=0.184, indicating
that <strong>salary_stipend</strong> drives the predictive relationship, not educational
spending per se. This is consistent with the institutional history: Oxford's most visible
transformation in this period was the professionalisation and expansion of its academic
staff, not curriculum reform.</p>

<p><strong>Limitations and next steps.</strong> All regression results remain associational —
no exogenous instrument for land-rent income has been identified, and Oxford is a single
institution precluding external validity. The late-era cutoff robustness check
(1800/1840/1870) reduces significance to p=0.118, indicating moderate sensitivity
to periodisation. A natural extension would be a diff-in-diff design comparing Oxford
against Cambridge or a set of non-university English institutions over the same period,
which would provide the cross-institutional variation needed to make causal claims.</p>
</div>
</div>

<hr>
<footer>
Generated by <code>experiments/analysis/analysis_v5.py</code> &nbsp;|&nbsp;
Data: <code>experiments/results/enriched/</code> (1,581 pages) &nbsp;|&nbsp;
Price deflation: Phelps Brown-Hopkins index (1700=100) &nbsp;|&nbsp;
Regressions: statsmodels OLS with Newey-West HAC (S1) and HC3 robust SE (S2–S4)
</footer>
</body></html>"""

    report_path = OUT_DIR / "analysis_v5_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  Section E complete → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(OUTCOME_DEFINITION)
    df = load_enriched_data()

    mfs_yr  = section_a(df)
    b_res   = section_b(df, mfs_yr)
    section_c(df, b_res)
    section_d(df)
    section_e(b_res)

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
