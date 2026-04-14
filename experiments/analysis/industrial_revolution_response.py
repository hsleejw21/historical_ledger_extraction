"""industrial_revolution_response.py

Novelty-phase analysis: Oxford's strategic financial restructuring during the
Industrial Revolution (1700-1900).

Novel contributions (not covered by existing literature):
  A1 – Revenue diversification: proactive vs reactive? (lead-lag cross-correlation)
  A2 – Expenditure reallocation: OLS test that cutting traditional funded modern
  A3 – Arrears by category × time (category-level, not aggregate)
  A4 – Payment period modernity index (entirely absent from Oxford literature)

Output directory: experiments/reports/analysis_v4/
"""

from __future__ import annotations

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
from scipy.stats import pearsonr, linregress

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_DIR = ROOT / "experiments" / "results" / "enriched"
OUT_DIR = ROOT / "experiments" / "reports" / "analysis_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

# ---------------------------------------------------------------------------
# Price deflation: Phelps Brown-Hopkins basket-of-consumables index (1700=100)
# Copied verbatim from analysis_enriched.py
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
_PBH_VALS = [_PBH_ANCHORS[y] for y in _PBH_YEARS]


def price_index(year: int) -> float:
    if year <= _PBH_YEARS[0]:
        return _PBH_VALS[0]
    if year >= _PBH_YEARS[-1]:
        return _PBH_VALS[-1]
    return float(np.interp(year, _PBH_YEARS, _PBH_VALS))


# ---------------------------------------------------------------------------
# Era periodization (4 eras anchored to IR scholarship)
# ---------------------------------------------------------------------------

ERAS = [
    ("pre_industrial",    1700, 1779),
    ("transition",        1780, 1819),
    ("early_industrial",  1820, 1859),
    ("late_industrial",   1860, 1900),
]
ERA_LABELS = {
    "pre_industrial":   "Pre-Industrial\n(1700–1779)",
    "transition":       "Transition\n(1780–1819)",
    "early_industrial": "Early Industrial\n(1820–1859)",
    "late_industrial":  "Late Industrial\n(1860–1900)",
}
ERA_VLINES = [1780, 1820, 1860]
ERA_ORDER = ["pre_industrial", "transition", "early_industrial", "late_industrial"]

# Agricultural shock reference years
AG_SHOCKS = {1793: "Enclosure Acts peak", 1822: "Post-Napoleonic depression", 1846: "Corn Laws repeal", 1873: "Great Agricultural Depression"}


def era_of_year(year: int) -> str:
    if year < 1780:
        return "pre_industrial"
    if year < 1820:
        return "transition"
    if year < 1860:
        return "early_industrial"
    return "late_industrial"


# ---------------------------------------------------------------------------
# Parsing helpers (copied from analysis_enriched.py)
# ---------------------------------------------------------------------------

def parse_fraction(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    mapping = {
        "¼": 0.25, "1/4": 0.25, ".25": 0.25,
        "½": 0.5,  "1/2": 0.5,  ".5":  0.5,
        "¾": 0.75, "3/4": 0.75, ".75": 0.75,
    }
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
# Data loading (adapted from analysis_enriched.py)
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
                    "page_id":               page_id,
                    "year":                  year,
                    "year_weight":           year_weight,
                    "amount":                amt,
                    "amount_weighted":       amt * year_weight,
                    "direction":             r.get("direction"),
                    "category":              r.get("category"),
                    "language":              r.get("language"),
                    "payment_period":        r.get("payment_period"),
                    "is_arrears":            r.get("is_arrears"),
                    "is_signature":          r.get("is_signature"),
                    "section_header":        r.get("section_header"),
                    "person_name":           r.get("person_name"),
                    "place_name":            r.get("place_name"),
                    "description":           r.get("description"),
                    "english_desc":          r.get("english_description"),
                    "notes":                 r.get("notes"),
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No entry rows loaded.")
    df["era"] = df["year"].map(era_of_year)
    df["price_idx"] = df["year"].map(price_index)
    df["amount_real"] = df["amount"] / (df["price_idx"] / 100.0)
    df["amount_real_w"] = df["amount_weighted"] / (df["price_idx"] / 100.0)
    print(f"  Loaded {len(df):,} records across {df['year'].nunique()} years "
          f"({df['year'].min()}–{df['year'].max()})")
    return df


def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = float(np.median(np.abs(x - med)))
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


def add_era_vlines(ax: plt.Axes, alpha: float = 0.5) -> None:
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=alpha)


# ---------------------------------------------------------------------------
# A1: Revenue Diversification — Proactive or Reactive?
# ---------------------------------------------------------------------------

INCOME_CATS = ["land_rent", "financial", "ecclesiastical", "educational",
               "administrative", "charitable", "maintenance", "salary_stipend",
               "domestic", "other"]

CAT_COLORS = {
    "land_rent":      "#8B4513",
    "financial":      "#1f77b4",
    "ecclesiastical": "#9467bd",
    "educational":    "#2ca02c",
    "administrative": "#ff7f0e",
    "charitable":     "#e377c2",
    "maintenance":    "#7f7f7f",
    "salary_stipend": "#17becf",
    "domestic":       "#bcbd22",
    "other":          "#d62728",
}


def analyse_revenue_diversification(df: pd.DataFrame) -> dict:
    print("[1/8] Analysing revenue diversification …")
    inc = df[df["direction"] == "income"].copy()

    grp = (inc.groupby(["year", "category"], as_index=False)
           .agg(amount_real=("amount_real_w", "sum")))
    total = (inc.groupby("year", as_index=False)
             .agg(total_real=("amount_real_w", "sum")))
    grp = grp.merge(total, on="year")
    grp["share"] = grp["amount_real"] / grp["total_real"].replace(0, np.nan)

    # Pivot wide
    wide = grp.pivot_table(index="year", columns="category",
                           values=["amount_real", "share"], fill_value=0.0)
    wide.columns = [f"{v}_{c}" for v, c in wide.columns]
    wide = wide.reset_index()
    # Merge total
    wide = wide.merge(total, on="year")
    wide["era"] = wide["year"].map(era_of_year)

    # HHI: sum of squared shares across all categories
    share_cols = [c for c in wide.columns if c.startswith("share_")]
    wide["hhi"] = (wide[share_cols] ** 2).sum(axis=1)
    wide["hhi_5yr"] = wide["hhi"].rolling(5, center=True, min_periods=3).mean()

    # 5-yr rolling shares for key categories
    for cat in ["land_rent", "financial", "educational"]:
        col = f"share_{cat}"
        if col in wide.columns:
            wide[f"{col}_5yr"] = wide[col].rolling(5, center=True, min_periods=3).mean()

    wide.to_csv(OUT_DIR / "revenue_diversification_yearly.csv", index=False)

    # Lead-lag: does financial income lead land rent decline?
    wide_yr = wide.set_index("year").sort_index()
    land_share = wide_yr.get("share_land_rent", pd.Series(dtype=float))
    fin_share  = wide_yr.get("share_financial", pd.Series(dtype=float))

    lag_records = []
    for k in range(-10, 11):
        d_land = land_share.diff()
        d_fin  = fin_share.diff()
        x = d_fin.shift(-k)
        common = x.notna() & d_land.notna()
        if common.sum() < 10:
            lag_records.append({"lag": k, "pearson_r": np.nan, "p_value": np.nan, "n": common.sum()})
            continue
        r, p = pearsonr(x[common], d_land[common])
        lag_records.append({"lag": k, "pearson_r": r, "p_value": p, "n": common.sum()})

    lag_df = pd.DataFrame(lag_records)
    lag_df.to_csv(OUT_DIR / "revenue_leadlag_correlations.csv", index=False)

    # Rolling 10-yr correlation between shares
    roll_records = []
    years = sorted(wide_yr.index)
    for start in range(years[0], years[-1] - 9):
        end = start + 9
        sub_land = land_share.loc[start:end].dropna()
        sub_fin  = fin_share.loc[start:end].dropna()
        idx = sub_land.index.intersection(sub_fin.index)
        if len(idx) < 5:
            continue
        r, p = pearsonr(sub_land[idx], sub_fin[idx])
        roll_records.append({"decade_start": start, "pearson_r": r, "p_value": p})
    roll_df = pd.DataFrame(roll_records)
    roll_df.to_csv(OUT_DIR / "revenue_rolling_correlation.csv", index=False)

    return {"wide": wide, "lag_df": lag_df, "roll_df": roll_df}


def plot_revenue_diversification(res: dict) -> None:
    print("[2/8] Plotting revenue diversification …")
    wide = res["wide"]
    lag_df = res["lag_df"]
    roll_df = res["roll_df"]

    years = wide["year"].values
    cats = [c for c in INCOME_CATS if f"share_{c}" in wide.columns]
    share_matrix = np.column_stack([wide[f"share_{c}"].fillna(0).values for c in cats])
    colors = [CAT_COLORS.get(c, "#aaa") for c in cats]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    # Panel 1: stacked area of income shares
    ax = axes[0]
    ax.stackplot(years, share_matrix.T, labels=cats, colors=colors, alpha=0.85)
    add_era_vlines(ax)
    for yr, lbl in AG_SHOCKS.items():
        ax.axvline(yr, color="darkred", lw=0.7, ls=":", alpha=0.7)
    ax.set_ylabel("Share of real income")
    ax.set_title("Oxford College Income: Category Shares (real £, 1700=100)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.6)
    ax.set_xlim(1700, 1900)
    ax.set_ylim(0, 1)

    # Panel 2: HHI + rolling + lead-lag verdict
    ax = axes[1]
    ax.plot(wide["year"], wide["hhi"], lw=0.7, color="grey", alpha=0.5, label="HHI (raw)")
    ax.plot(wide["year"], wide["hhi_5yr"], lw=2, color="navy", label="HHI (5-yr rolling)")
    ax.axhline(0.35, color="crimson", lw=1, ls="--", label="Diversification threshold (0.35)")
    add_era_vlines(ax)
    ax.set_ylabel("Herfindahl-Hirschman Index")
    ax.set_title("Income Concentration (HHI): Lower = More Diversified", fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(1700, 1900)

    # Annotate first year HHI < 0.35
    hhi_thresh = wide[wide["hhi_5yr"] < 0.35]
    if not hhi_thresh.empty:
        first_div = hhi_thresh["year"].min()
        ax.annotate(f"HHI < 0.35\n({first_div})", xy=(first_div, 0.35),
                    xytext=(first_div + 5, 0.42),
                    arrowprops=dict(arrowstyle="->", color="crimson"), fontsize=7, color="crimson")

    # Panel 3: Lead-lag correlation bar chart
    ax = axes[2]
    colors_bar = ["#c0392b" if r < 0 else "#27ae60" for r in lag_df["pearson_r"].fillna(0)]
    ax.bar(lag_df["lag"], lag_df["pearson_r"].fillna(0), color=colors_bar, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Lag (years): negative = financial leads land decline")
    ax.set_ylabel("Pearson r")
    ax.set_title("Lead-Lag: Did Financial Income Grow BEFORE Land Rent Fell? (Red = financial leads = proactive)",
                 fontweight="bold")

    # Annotate k_max
    valid = lag_df.dropna(subset=["pearson_r"])
    if not valid.empty:
        kmax_row = valid.loc[valid["pearson_r"].abs().idxmax()]
        verdict = "PROACTIVE" if kmax_row["lag"] < 0 else "REACTIVE"
        ax.annotate(f"Peak r={kmax_row['pearson_r']:.2f}\nlag={int(kmax_row['lag'])}y ({verdict})",
                    xy=(kmax_row["lag"], kmax_row["pearson_r"]),
                    xytext=(kmax_row["lag"] + 1.5, kmax_row["pearson_r"] * 0.8),
                    arrowprops=dict(arrowstyle="->"), fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A1_revenue_diversification.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A2: Expenditure Reallocation — OLS test
# ---------------------------------------------------------------------------

TRADITIONAL_CATS = ["ecclesiastical", "domestic"]
MODERN_CATS = ["educational", "salary_stipend"]


def analyse_expenditure_reallocation(df: pd.DataFrame) -> dict:
    print("[3/8] Analysing expenditure reallocation …")
    exp = df[df["direction"] == "expenditure"].copy()

    grp = (exp.groupby(["year", "category"], as_index=False)
           .agg(amount_real=("amount_real_w", "sum")))
    wide = grp.pivot_table(index="year", columns="category",
                           values="amount_real", fill_value=0.0)
    wide.columns = list(wide.columns)
    wide = wide.reset_index()
    wide["era"] = wide["year"].map(era_of_year)

    # 5-yr rolling per category
    all_cats = [c for c in wide.columns if c not in ["year", "era"]]
    for c in all_cats:
        wide[f"{c}_5yr"] = wide[c].rolling(5, center=True, min_periods=3).mean()

    # Era averages
    era_avg = {}
    for era_name, y_start, y_end in ERAS:
        sub = wide[(wide["year"] >= y_start) & (wide["year"] <= y_end)]
        era_avg[era_name] = {c: sub[c].mean() for c in all_cats if c in wide.columns}

    era_rows = []
    for cat in all_cats:
        row = {"category": cat}
        for era_name in ERA_ORDER:
            row[f"{era_name}_avg"] = era_avg.get(era_name, {}).get(cat, np.nan)
        pre = row.get("pre_industrial_avg", np.nan)
        if pre and pre > 0:
            for era_name in ERA_ORDER[1:]:
                row[f"pct_vs_pre_{era_name}"] = (row[f"{era_name}_avg"] - pre) / pre * 100
        era_rows.append(row)

    era_df = pd.DataFrame(era_rows)
    era_df.to_csv(OUT_DIR / "expenditure_era_pct_change.csv", index=False)
    wide.to_csv(OUT_DIR / "expenditure_reallocation_yearly.csv", index=False)

    # OLS: Δ(modern) ~ Δ(traditional) + year
    wide_s = wide.set_index("year").sort_index()
    trad_cols = [c for c in TRADITIONAL_CATS if c in wide_s.columns]
    mod_cols  = [c for c in MODERN_CATS if c in wide_s.columns]

    if trad_cols and mod_cols:
        trad_sum = wide_s[trad_cols].sum(axis=1)
        mod_sum  = wide_s[mod_cols].sum(axis=1)
        d_trad = trad_sum.diff().dropna()
        d_mod  = mod_sum.diff().dropna()
        idx = d_trad.index.intersection(d_mod.index)
        X = d_trad[idx].values
        Y = d_mod[idx].values
        sl, ic, rv, pv, se = linregress(X, Y)
        ols_result = {"slope": sl, "intercept": ic, "r_value": rv, "p_value": pv, "std_err": se}
    else:
        ols_result = {}

    return {"wide": wide, "era_df": era_df, "ols": ols_result, "all_cats": all_cats}


def plot_expenditure_reallocation(res: dict) -> None:
    print("[4/8] Plotting expenditure reallocation …")
    wide = res["wide"]
    era_df = res["era_df"]
    ols = res["ols"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: Traditional category lines (5-yr rolling)
    ax = axes[0]
    trad_palette = {"ecclesiastical": "#9467bd", "domestic": "#e377c2",
                    "maintenance": "#7f7f7f", "charitable": "#d62728"}
    for cat, color in trad_palette.items():
        col = f"{cat}_5yr"
        if col in wide.columns:
            ax.plot(wide["year"], wide[col], label=cat, color=color, lw=1.5)
    add_era_vlines(ax)
    ax.set_ylabel("Real £ (1700=100), 5-yr rolling")
    ax.set_title("Traditional Expenditure Categories (Real £)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1900)

    # Panel 2: Modern category lines
    ax = axes[1]
    mod_palette = {"educational": "#2ca02c", "salary_stipend": "#17becf",
                   "administrative": "#ff7f0e", "financial": "#1f77b4"}
    for cat, color in mod_palette.items():
        col = f"{cat}_5yr"
        if col in wide.columns:
            ax.plot(wide["year"], wide[col], label=cat, color=color, lw=1.5)
    add_era_vlines(ax)
    ax.set_ylabel("Real £ (1700=100), 5-yr rolling")
    ax.set_title("Modernising Expenditure Categories (Real £)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1900)

    # Panel 3: Grouped era bar chart for key categories
    ax = axes[2]
    bar_cats = ["ecclesiastical", "domestic", "educational", "salary_stipend"]
    bar_cats = [c for c in bar_cats if c in era_df["category"].values]
    x = np.arange(len(ERA_ORDER))
    width = 0.18
    cat_colors_bar = {"ecclesiastical": "#9467bd", "domestic": "#e377c2",
                      "educational": "#2ca02c", "salary_stipend": "#17becf"}
    for i, cat in enumerate(bar_cats):
        row = era_df[era_df["category"] == cat].iloc[0]
        vals = [row.get(f"{e}_avg", 0) or 0 for e in ERA_ORDER]
        offset = (i - len(bar_cats) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cat,
                      color=cat_colors_bar.get(cat, "#aaa"), alpha=0.85)
        # Annotate % change vs pre-industrial
        for j, (bar, era_name) in enumerate(zip(bars, ERA_ORDER)):
            if era_name != "pre_industrial":
                pct = row.get(f"pct_vs_pre_{era_name}")
                if pct is not None and not np.isnan(pct):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                            f"{pct:+.0f}%", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([ERA_LABELS[e].replace("\n", " ") for e in ERA_ORDER], fontsize=7)
    ax.set_ylabel("Mean annual real £")
    ax.set_title("Expenditure Reallocation by Era (% change vs pre-industrial annotated)",
                 fontweight="bold")
    ax.legend(fontsize=8)

    # OLS annotation
    if ols:
        sign = "+" if ols["slope"] > 0 else ""
        verdict = "CONFIRMED" if ols["slope"] > 0 and ols["p_value"] < 0.05 else "not significant"
        ax.text(0.02, 0.97, f"OLS: Δmodern = {ols['slope']:.2f}·Δtraditional + …  "
                f"(r={ols['r_value']:.2f}, p={ols['p_value']:.3f}) → {verdict}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A2_expenditure_reallocation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A3: Arrears by Category × Time
# ---------------------------------------------------------------------------

def analyse_arrears(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/8] Analysing arrears by category …")
    arr = df[df["is_arrears"].notna() & df["category"].notna()].copy()
    arr["is_arrears_bool"] = arr["is_arrears"].astype(bool)

    grp = arr.groupby(["year", "category"]).apply(
        lambda g: pd.Series({
            "n_entries":    g["year_weight"].sum(),
            "n_arrears":    (g["is_arrears_bool"] * g["year_weight"]).sum(),
            "amount_total": g["amount_real_w"].sum(),
            "amount_arrears": g.loc[g["is_arrears_bool"], "amount_real_w"].sum(),
        })
    ).reset_index()
    grp["arrears_rate"] = grp["n_arrears"] / grp["n_entries"].replace(0, np.nan)
    grp["arrears_amount_share"] = grp["amount_arrears"] / grp["amount_total"].replace(0, np.nan)
    grp["era"] = grp["year"].map(era_of_year)

    grp.to_csv(OUT_DIR / "arrears_by_category_yearly.csv", index=False)

    # Compute stress index = land_rent_arrears_rate × land_rent_income_share
    land_arr = grp[grp["category"] == "land_rent"][["year", "arrears_rate", "arrears_amount_share"]].copy()
    land_arr.columns = ["year", "land_arrears_rate", "land_amount_share"]

    inc = df[df["direction"] == "income"].copy()
    inc_total = inc.groupby("year")["amount_real_w"].sum().rename("total_income_real")
    land_inc = (inc[inc["category"] == "land_rent"]
                .groupby("year")["amount_real_w"].sum().rename("land_income_real"))
    land_share_df = pd.concat([inc_total, land_inc], axis=1).reset_index()
    land_share_df["land_income_share"] = land_share_df["land_income_real"] / land_share_df["total_income_real"].replace(0, np.nan)

    stress_df = land_arr.merge(land_share_df[["year", "land_income_share"]], on="year", how="left")
    stress_df["stress_index"] = stress_df["land_arrears_rate"] * stress_df["land_income_share"]
    stress_df.to_csv(OUT_DIR / "arrears_stress_index.csv", index=False)

    return grp


def plot_arrears(grp: pd.DataFrame) -> None:
    print("[6/8] Plotting arrears …")
    stress_df = pd.read_csv(OUT_DIR / "arrears_stress_index.csv")

    focus_cats = ["land_rent", "ecclesiastical", "financial"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Panel 1: Arrears rate per category (5-yr rolling)
    ax = axes[0]
    cat_colors = {"land_rent": "#8B4513", "ecclesiastical": "#9467bd", "financial": "#1f77b4"}
    for cat in focus_cats:
        sub = grp[grp["category"] == cat].sort_values("year")
        if sub.empty:
            continue
        roll = sub.set_index("year")["arrears_rate"].rolling(5, center=True, min_periods=3).mean()
        ax.plot(roll.index, roll.values, label=cat, color=cat_colors.get(cat, "#aaa"), lw=1.8)
    # Shade historical periods
    ax.axvspan(1793, 1815, alpha=0.08, color="orange", label="Napoleonic Wars")
    ax.axvspan(1815, 1830, alpha=0.08, color="red",    label="Post-war depression")
    ax.axvspan(1873, 1896, alpha=0.08, color="brown",  label="Great Ag. Depression")
    add_era_vlines(ax, alpha=0.4)
    ax.set_ylabel("Arrears rate (entries in arrears / total), 5-yr rolling")
    ax.set_title("Arrears Rate by Category: When Did Tenants Stop Paying?", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(1700, 1900)

    # Panel 2: Stress index
    ax = axes[1]
    stress_yr = stress_df.set_index("year").sort_index()
    ax.fill_between(stress_yr.index, stress_yr["stress_index"].fillna(0),
                    alpha=0.4, color="sienna", label="Stress index (arrears rate × land income share)")
    roll_stress = stress_yr["stress_index"].rolling(7, center=True, min_periods=3).mean()
    ax.plot(roll_stress.index, roll_stress.values, color="darkred", lw=2, label="7-yr rolling")
    # Mark 2-sigma anomalies
    zs = robust_zscore(stress_yr["stress_index"].dropna())
    anomalies = zs[zs.abs() > 2]
    ax.scatter(anomalies.index, stress_yr.loc[anomalies.index, "stress_index"],
               color="red", zorder=5, s=20, label="|z| > 2")
    ax.axvspan(1793, 1815, alpha=0.08, color="orange")
    ax.axvspan(1815, 1830, alpha=0.08, color="red")
    ax.axvspan(1873, 1896, alpha=0.08, color="brown")
    add_era_vlines(ax, alpha=0.4)
    ax.set_ylabel("Risk-weighted stress index")
    ax.set_xlabel("Year")
    ax.set_title("Institutional Risk: Land Rent Arrears × Income Dependency", fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(1700, 1900)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A3_arrears_stress.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A4: Payment Period Modernity Index
# ---------------------------------------------------------------------------

MODERNITY_SCORE = {
    "annual":       1.0,
    "half_year":    0.9,
    "sesquiannual": 0.7,
    "one_off":      0.6,
    "biennial":     0.5,
    "triennial":    0.4,
    "quadrennial":  0.3,
    "quinquennial": 0.2,
    "multi_year":   0.1,
}

PERIOD_COLORS = {
    "annual":       "#1f77b4",
    "half_year":    "#aec7e8",
    "sesquiannual": "#98df8a",
    "one_off":      "#ff7f0e",
    "biennial":     "#ffbb78",
    "triennial":    "#d62728",
    "quadrennial":  "#9467bd",
    "quinquennial": "#c5b0d5",
    "multi_year":   "#8c564b",
}


def analyse_payment_period(df: pd.DataFrame) -> pd.DataFrame:
    print("[7/8] Analysing payment period modernity …")
    pay = df[df["payment_period"].notna() &
             (df["payment_period"] != "unclear") &
             (df["payment_period"] != "")].copy()
    pay["modernity"] = pay["payment_period"].map(MODERNITY_SCORE)
    pay = pay[pay["modernity"].notna()]

    # Yearly weighted modernity index
    yearly_mod = (pay.groupby("year").apply(
        lambda g: pd.Series({
            "weighted_modernity": (g["modernity"] * g["year_weight"]).sum() / g["year_weight"].sum(),
            "total_weight": g["year_weight"].sum(),
        })
    ).reset_index())

    # Yearly period shares
    period_counts = (pay.groupby(["year", "payment_period"])["year_weight"].sum()
                     .reset_index(name="count"))
    period_totals = period_counts.groupby("year")["count"].sum().rename("total")
    period_counts = period_counts.merge(period_totals, on="year")
    period_counts["share"] = period_counts["count"] / period_counts["total"]

    period_wide = period_counts.pivot_table(index="year", columns="payment_period",
                                            values="share", fill_value=0.0).reset_index()
    period_wide = period_wide.merge(yearly_mod, on="year")
    period_wide["era"] = period_wide["year"].map(era_of_year)

    # Shannon entropy
    period_cols = [c for c in MODERNITY_SCORE.keys() if c in period_wide.columns]
    def shannon(row):
        vals = row[period_cols].astype(float).clip(lower=1e-9)
        vals = vals / vals.sum()
        return -float(np.sum(vals.values * np.log(vals.values)))
    period_wide["entropy"] = period_wide.apply(shannon, axis=1)

    # 5-yr rolling modernity
    period_wide = period_wide.sort_values("year")
    period_wide["modernity_5yr"] = period_wide["weighted_modernity"].rolling(5, center=True, min_periods=3).mean()

    # Category-level modernity by era
    cat_era_mod = (pay.groupby(["era", "category"]).apply(
        lambda g: (g["modernity"] * g["year_weight"]).sum() / g["year_weight"].sum()
    ).reset_index(name="modernity_score"))
    cat_era_mod.to_csv(OUT_DIR / "payment_modernity_by_category_era.csv", index=False)

    period_wide.to_csv(OUT_DIR / "payment_period_modernity_yearly.csv", index=False)
    return period_wide


def plot_payment_period(period_wide: pd.DataFrame) -> None:
    print("[8/8] Plotting payment period modernity …")
    period_cols = [c for c in MODERNITY_SCORE.keys() if c in period_wide.columns]
    years = period_wide["year"].values

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    # Panel 1: Stacked area of period shares
    ax = axes[0]
    share_matrix = np.column_stack([period_wide[c].fillna(0).values for c in period_cols])
    colors = [PERIOD_COLORS.get(c, "#aaa") for c in period_cols]
    ax.stackplot(years, share_matrix.T, labels=period_cols, colors=colors, alpha=0.85)
    add_era_vlines(ax)
    ax.set_ylabel("Share of entries")
    ax.set_title("Payment Period Distribution Over Time", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.6)
    ax.set_xlim(1700, 1900)

    # Panel 2: Modernity index
    ax = axes[1]
    ax.plot(period_wide["year"], period_wide["weighted_modernity"],
            lw=0.7, color="grey", alpha=0.5, label="Raw")
    ax.plot(period_wide["year"], period_wide["modernity_5yr"],
            lw=2, color="navy", label="5-yr rolling")
    ax.axhline(0.5, color="orange", lw=1, ls="--", label="Score = 0.5")
    ax.axhline(0.75, color="green", lw=1, ls="--", label="Score = 0.75")
    add_era_vlines(ax)
    ax.set_ylabel("Weighted modernity score")
    ax.set_title("Payment Period Modernity Index (0=feudal tenure, 1=annual market payments)",
                 fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(1700, 1900)
    ax.set_ylim(0, 1.05)

    # Panel 3: Category-level modernity heatmap by era
    ax = axes[2]
    cat_era = pd.read_csv(OUT_DIR / "payment_modernity_by_category_era.csv")
    pivot = cat_era.pivot_table(index="category", columns="era",
                                values="modernity_score")[ERA_ORDER].fillna(0)
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Modernity score"})
    ax.set_title("Payment Modernity by Category and Era", fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticklabels([ERA_LABELS[e].replace("\n", " ") for e in ERA_ORDER
                        if e in pivot.columns], fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A4_payment_modernization.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------------

def write_summary(div_res: dict, ols: dict, grp_arr: pd.DataFrame,
                  period_wide: pd.DataFrame) -> None:
    wide = div_res["wide"]
    lag_df = div_res["lag_df"]

    # HHI threshold
    hhi_below = wide[wide["hhi_5yr"] < 0.35]
    first_div_yr = int(hhi_below["year"].min()) if not hhi_below.empty else None

    # Lead-lag verdict
    valid_lag = lag_df.dropna(subset=["pearson_r"])
    if not valid_lag.empty:
        kmax = int(valid_lag.loc[valid_lag["pearson_r"].abs().idxmax(), "lag"])
        if kmax < 0:
            verdict = "PROACTIVE (financial grew before land declined)"
        elif kmax > 0:
            verdict = "REACTIVE (land fell first)"
        else:
            verdict = "SIMULTANEOUS (k_max=0; land and financial moved in parallel, no clear directional lead)"
    else:
        kmax, verdict = None, "inconclusive"

    # OLS
    if ols:
        ols_verdict = (f"slope={ols['slope']:.3f}, p={ols['p_value']:.3f} → "
                       + ("confirmed: cutting traditional funded modern"
                          if ols["slope"] > 0 and ols["p_value"] < 0.05
                          else "not significant"))
    else:
        ols_verdict = "OLS could not be computed"

    # Arrears
    land_arr = grp_arr[grp_arr["category"] == "land_rent"].sort_values("year")
    below5 = land_arr[land_arr["arrears_rate"] < 0.05]
    arr_below5_yr = int(below5["year"].min()) if not below5.empty else None

    # Modernity
    mod_post1820 = period_wide[period_wide["year"] >= 1820]["modernity_5yr"]
    mod_trend = "rising" if mod_post1820.is_monotonic_increasing else \
                "falling" if mod_post1820.is_monotonic_decreasing else "mixed"

    lines = [
        "OXFORD INDUSTRIAL REVOLUTION — FINANCIAL RESTRUCTURING SUMMARY",
        "=" * 65,
        f"First year HHI (5-yr rolling) < 0.35:  {first_div_yr or 'never reached threshold'}",
        f"Lead-lag k_max: {kmax} years  →  Oxford's diversification was {verdict}",
        f"Expenditure reallocation OLS: {ols_verdict}",
        f"Land rent arrears rate first fell below 5%: {arr_below5_yr or 'never'}",
        f"Payment modernity trend post-1820: {mod_trend}",
        "",
        "Agricultural shock reference lines in figures: 1793, 1822, 1846, 1873",
        "Era boundaries: 1780, 1820, 1860",
        "Price deflation: Phelps Brown-Hopkins index (1700=100)",
    ]
    (OUT_DIR / "summary_financial.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_enriched_data()

    div_res = analyse_revenue_diversification(df)
    plot_revenue_diversification(div_res)

    exp_res = analyse_expenditure_reallocation(df)
    plot_expenditure_reallocation(exp_res)

    grp_arr = analyse_arrears(df)
    plot_arrears(grp_arr)

    period_wide = analyse_payment_period(df)
    plot_payment_period(period_wide)

    write_summary(div_res, exp_res["ols"], grp_arr, period_wide)

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
