"""analysis_enriched.py

Comprehensive analysis of enriched ledger data using semantic fields added
by the enrichment pipeline (direction, category, language, payment_period,
is_arrears, place_name, person_name).

Loads enriched JSON files directly from experiments/results/enriched/ and
produces:
  - Flat CSV of all entry rows with parsed amounts and enrichment labels
  - Yearly summaries: direction balance, category shares, language shift,
    arrears rate, payment period distribution
  - Place-name and person-name frequency tables by era
  - Change-point detection on enriched metrics
  - Visualisations for all key findings

Output directory: experiments/reports/enriched_analysis/
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_DIR = ROOT / "experiments" / "results" / "enriched"
OUT_DIR = ROOT / "experiments" / "reports" / "analysis_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Price deflation: Phelps Brown-Hopkins basket-of-consumables index (1700=100)
# (same anchors as reanalysis_ledger_yearly_v2.py)
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
# Era labels
# ---------------------------------------------------------------------------

def era_of_year(year: int) -> str:
    if year < 1760:
        return "pre_1760"
    if 1760 <= year <= 1840:
        return "industrial_1760_1840"
    return "post_1840"


# ---------------------------------------------------------------------------
# Parsing helpers
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
    """Return (years_list, page_number) from a page_id like '1700_1_image' or
    '1721-1722_6_image'."""
    single = re.match(r"^(\d{4})_(\d+)_image$", page_id)
    if single:
        return [int(single.group(1))], int(single.group(2))
    span = re.match(r"^(\d{4})-(\d{4})_(\d+)_image$", page_id)
    if span:
        y1, y2, pg = int(span.group(1)), int(span.group(2)), int(span.group(3))
        if y2 < y1:
            y1, y2 = y2, y1
        return list(range(y1, y2 + 1)), pg
    # Fallback: try to extract the first 4-digit number as year
    m = re.search(r"(\d{4})", page_id)
    if m:
        return [int(m.group(1))], 1
    raise ValueError(f"Cannot parse page_id: {page_id!r}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_enriched_data() -> pd.DataFrame:
    """Load all enriched JSON files and return a flat DataFrame of entry rows."""
    records: list[dict[str, Any]] = []

    files = sorted(ENRICHED_DIR.glob("*_image_enriched.json"))
    if not files:
        raise FileNotFoundError(f"No enriched JSON files found in {ENRICHED_DIR}")

    print(f"Loading {len(files)} enriched JSON files …")

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
            print(f"  SKIP {page_id}: cannot parse year")
            continue

        year_weight = 1.0 / len(years)
        rows = payload.get("rows", [])

        for r in rows:
            if not isinstance(r, dict):
                continue
            row_type = str(r.get("row_type", "")).strip().lower()
            if row_type != "entry":
                continue

            amt = amount_decimal(r)

            for year in years:
                records.append({
                    "page_id":          page_id,
                    "year":             year,
                    "page":             page,
                    "year_weight":      year_weight,
                    "row_index":        r.get("row_index"),
                    "description":      r.get("description", ""),
                    "amount":           amt,
                    "amount_weighted":  amt * year_weight,
                    "has_amount":       amt > 0,
                    "confidence":       r.get("confidence_score", np.nan),
                    "section_header":   r.get("section_header"),
                    # Enrichment fields
                    "direction":        r.get("direction"),
                    "category":         r.get("category"),
                    "language":         r.get("language"),
                    "english_desc":     r.get("english_description"),
                    "place_name":       r.get("place_name"),
                    "person_name":      r.get("person_name"),
                    "payment_period":   r.get("payment_period"),
                    "is_signature":     r.get("is_signature"),
                    "is_arrears":       r.get("is_arrears"),
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No entry rows loaded.")

    df["era"] = df["year"].map(era_of_year)
    df["price_index"] = df["year"].map(price_index)
    df["amount_real"] = df["amount"] / (df["price_index"] / 100.0)
    df["amount_real_weighted"] = df["amount_weighted"] / (df["price_index"] / 100.0)

    print(f"  Loaded {len(df):,} entry-row records across "
          f"{df['year'].nunique()} years ({df['year'].min()}–{df['year'].max()})")
    return df


# ---------------------------------------------------------------------------
# Robust z-score for change-point detection
# ---------------------------------------------------------------------------

def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = float(np.median(np.abs(x - med)))
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


# ---------------------------------------------------------------------------
# Analysis 1: Direction balance (income vs expenditure)
# ---------------------------------------------------------------------------

def analyse_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Yearly income / expenditure split by entry count and amount."""
    entry_dir = df[df["direction"].isin(["income", "expenditure"])].copy()
    if entry_dir.empty:
        return pd.DataFrame()

    grp = (
        entry_dir.groupby(["year", "direction"], as_index=False)
        .agg(
            n_entries     = ("year_weight", "sum"),
            amount_sum    = ("amount_weighted", "sum"),
            amount_real   = ("amount_real_weighted", "sum"),
        )
    )

    # Pivot to wide: one column per direction
    wide = grp.pivot_table(
        index="year",
        columns="direction",
        values=["n_entries", "amount_sum", "amount_real"],
        fill_value=0.0,
    )
    wide.columns = ["_".join(c).strip() for c in wide.columns]
    wide = wide.reset_index()

    for col in ["n_entries_income", "n_entries_expenditure",
                "amount_sum_income", "amount_sum_expenditure",
                "amount_real_income", "amount_real_expenditure"]:
        if col not in wide.columns:
            wide[col] = 0.0

    wide["total_entries"] = wide["n_entries_income"] + wide["n_entries_expenditure"]
    wide["income_share"]  = wide["n_entries_income"] / wide["total_entries"].replace(0, np.nan)
    wide["net_amount"]    = wide["amount_sum_income"] - wide["amount_sum_expenditure"]
    wide["net_amount_real"] = wide["amount_real_income"] - wide["amount_real_expenditure"]
    wide["era"]           = wide["year"].map(era_of_year)

    return wide.sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 2: Category breakdown
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "land_rent", "ecclesiastical", "maintenance", "salary_stipend",
    "administrative", "educational", "financial", "domestic",
    "charitable", "other",
]


def analyse_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      yearly_cat  – yearly entry count and amount share per category
      era_cat     – era-level category totals and shares
    """
    cat_df = df[df["category"].notna()].copy()

    # --- yearly ---
    yearly = (
        cat_df.groupby(["year", "category"], as_index=False)
        .agg(
            n_entries   = ("year_weight", "sum"),
            amount_sum  = ("amount_weighted", "sum"),
            amount_real = ("amount_real_weighted", "sum"),
        )
    )
    yearly_total = (
        cat_df.groupby("year", as_index=False)
        .agg(total_entries=("year_weight", "sum"),
             total_amount=("amount_weighted", "sum"))
    )
    yearly = yearly.merge(yearly_total, on="year")
    yearly["entry_share"]  = yearly["n_entries"] / yearly["total_entries"].replace(0, np.nan)
    yearly["amount_share"] = yearly["amount_sum"] / yearly["total_amount"].replace(0, np.nan)
    yearly["era"] = yearly["year"].map(era_of_year)

    # --- era ---
    era_cat = (
        cat_df.groupby(["era", "category"], as_index=False)
        .agg(
            n_entries   = ("year_weight", "sum"),
            amount_sum  = ("amount_weighted", "sum"),
            amount_real = ("amount_real_weighted", "sum"),
        )
    )
    era_total = (
        cat_df.groupby("era", as_index=False)
        .agg(total_entries=("year_weight", "sum"),
             total_amount=("amount_weighted", "sum"))
    )
    era_cat = era_cat.merge(era_total, on="era")
    era_cat["entry_share"]  = era_cat["n_entries"] / era_cat["total_entries"].replace(0, np.nan)
    era_cat["amount_share"] = era_cat["amount_sum"] / era_cat["total_amount"].replace(0, np.nan)
    era_order = ["pre_1760", "industrial_1760_1840", "post_1840"]
    era_cat["era"] = pd.Categorical(era_cat["era"], categories=era_order, ordered=True)
    era_cat = era_cat.sort_values(["era", "category"])

    return yearly, era_cat


# ---------------------------------------------------------------------------
# Analysis 3: Language shift
# ---------------------------------------------------------------------------

def analyse_language(df: pd.DataFrame) -> pd.DataFrame:
    lang_df = df[df["language"].isin(["latin", "english", "mixed"])].copy()
    if lang_df.empty:
        return pd.DataFrame()

    grp = (
        lang_df.groupby(["year", "language"], as_index=False)
        .agg(n_entries=("year_weight", "sum"))
    )
    total = (
        lang_df.groupby("year", as_index=False)
        .agg(total=("year_weight", "sum"))
    )
    grp = grp.merge(total, on="year")
    grp["share"] = grp["n_entries"] / grp["total"].replace(0, np.nan)
    grp["era"] = grp["year"].map(era_of_year)

    return grp.sort_values(["year", "language"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 4: Payment period distribution
# ---------------------------------------------------------------------------

def analyse_payment_period(df: pd.DataFrame) -> pd.DataFrame:
    pp_df = df[df["payment_period"].notna() & (df["payment_period"] != "unclear")].copy()
    if pp_df.empty:
        return pd.DataFrame()

    grp = (
        pp_df.groupby(["year", "payment_period"], as_index=False)
        .agg(n_entries=("year_weight", "sum"))
    )
    total = (
        pp_df.groupby("year", as_index=False)
        .agg(total=("year_weight", "sum"))
    )
    grp = grp.merge(total, on="year")
    grp["share"] = grp["n_entries"] / grp["total"].replace(0, np.nan)
    grp["era"] = grp["year"].map(era_of_year)
    return grp.sort_values(["year", "payment_period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 5: Arrears rate
# ---------------------------------------------------------------------------

def analyse_arrears(df: pd.DataFrame) -> pd.DataFrame:
    arr = df[df["is_arrears"].notna()].copy()
    arr["is_arrears_bool"] = arr["is_arrears"].astype(bool)

    yearly = (
        arr.groupby("year", as_index=False)
        .agg(
            n_entries      = ("year_weight", "sum"),
            n_arrears      = ("is_arrears_bool",
                              lambda s: float((s * arr.loc[s.index, "year_weight"]).sum())),
            amount_arrears = ("amount_weighted",
                              lambda s: float(
                                  s[arr.loc[s.index, "is_arrears_bool"]].sum()
                              )),
            amount_total   = ("amount_weighted", "sum"),
        )
    )
    yearly["arrears_rate"]          = yearly["n_arrears"] / yearly["n_entries"].replace(0, np.nan)
    yearly["arrears_amount_share"]  = yearly["amount_arrears"] / yearly["amount_total"].replace(0, np.nan)
    yearly["era"]                   = yearly["year"].map(era_of_year)
    return yearly.sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 6: Place and person name frequency
# ---------------------------------------------------------------------------

def _split_places(place_str: str | None) -> list[str]:
    """A place_name can be 'Cuddington; South Newington; Merton' – split on ';'."""
    if not place_str:
        return []
    return [p.strip() for p in str(place_str).split(";") if p.strip()]


def analyse_names(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top place and person names by era (weighted entry count)."""
    era_order = ["pre_1760", "industrial_1760_1840", "post_1840"]

    # --- Places ---
    place_records: list[dict] = []
    for _, row in df[df["place_name"].notna()].iterrows():
        for pl in _split_places(row["place_name"]):
            place_records.append({
                "era": row["era"],
                "name": pl,
                "weight": row["year_weight"],
                "amount": row["amount_weighted"],
                "year": row["year"],
            })
    place_df = pd.DataFrame(place_records)

    place_agg = pd.DataFrame()
    if not place_df.empty:
        place_agg = (
            place_df.groupby(["era", "name"], as_index=False)
            .agg(n_entries=("weight", "sum"),
                 amount_sum=("amount", "sum"),
                 first_year=("year", "min"),
                 last_year=("year", "max"))
            .sort_values(["era", "n_entries"], ascending=[True, False])
        )
        place_agg["era"] = pd.Categorical(place_agg["era"], categories=era_order, ordered=True)
        place_agg = place_agg.sort_values(["era", "n_entries"], ascending=[True, False])

    # --- Persons ---
    person_records: list[dict] = []
    for _, row in df[df["person_name"].notna()].iterrows():
        person_records.append({
            "era": row["era"],
            "name": str(row["person_name"]).strip(),
            "weight": row["year_weight"],
            "amount": row["amount_weighted"],
            "year": row["year"],
        })
    person_df = pd.DataFrame(person_records)

    person_agg = pd.DataFrame()
    if not person_df.empty:
        person_agg = (
            person_df.groupby(["era", "name"], as_index=False)
            .agg(n_entries=("weight", "sum"),
                 amount_sum=("amount", "sum"),
                 first_year=("year", "min"),
                 last_year=("year", "max"))
        )
        person_agg["era"] = pd.Categorical(person_agg["era"], categories=era_order, ordered=True)
        person_agg = person_agg.sort_values(["era", "n_entries"], ascending=[True, False])

    return place_agg, person_agg


# ---------------------------------------------------------------------------
# Analysis 7: Category × direction cross-tab by era
# ---------------------------------------------------------------------------

def analyse_category_direction(df: pd.DataFrame) -> pd.DataFrame:
    cd = df[
        df["category"].notna() & df["direction"].isin(["income", "expenditure"])
    ].copy()
    if cd.empty:
        return pd.DataFrame()

    era_order = ["pre_1760", "industrial_1760_1840", "post_1840"]
    agg = (
        cd.groupby(["era", "category", "direction"], as_index=False)
        .agg(n_entries=("year_weight", "sum"),
             amount_sum=("amount_weighted", "sum"))
    )
    agg["era"] = pd.Categorical(agg["era"], categories=era_order, ordered=True)
    return agg.sort_values(["era", "category", "direction"])


# ---------------------------------------------------------------------------
# Analysis 8: Change-point detection on enriched metrics
# ---------------------------------------------------------------------------

def detect_enriched_change_points(
    direction_wide: pd.DataFrame,
    lang_wide: pd.DataFrame,
    arrears: pd.DataFrame,
) -> pd.DataFrame:
    """Detect change points in enriched metrics using robust z-scores."""
    records: list[dict[str, Any]] = []

    def _check(series_df: pd.DataFrame, year_col: str,
                metric_col: str, label: str) -> None:
        sub = series_df[["year", metric_col]].dropna().sort_values("year")
        if len(sub) < 5:
            return
        diff = sub[metric_col].diff().fillna(0.0)
        rz   = robust_zscore(diff.abs())
        flag = rz >= 2.5
        for i in range(len(sub)):
            records.append({
                "year":               int(sub.iloc[i]["year"]),
                "metric":             label,
                "value":              float(sub.iloc[i][metric_col]),
                "delta":              float(diff.iloc[i]),
                "abs_delta_robust_z": float(rz.iloc[i]),
                "is_change_point":    bool(flag.iloc[i]),
            })

    # Direction: income share, net amount (real)
    if not direction_wide.empty:
        _check(direction_wide, "year", "income_share",    "income_share")
        _check(direction_wide, "year", "net_amount_real", "net_amount_real")
        _check(direction_wide, "year", "amount_sum_expenditure", "expenditure_amount")
        _check(direction_wide, "year", "amount_sum_income",      "income_amount")

    # Language: English share
    if not lang_wide.empty:
        eng = lang_wide[lang_wide["language"] == "english"][["year", "share"]].copy()
        eng = eng.rename(columns={"share": "english_share"})
        _check(eng, "year", "english_share", "english_share")

    # Arrears rate
    if not arrears.empty:
        _check(arrears, "year", "arrears_rate",         "arrears_rate")
        _check(arrears, "year", "arrears_amount_share",  "arrears_amount_share")

    if not records:
        return pd.DataFrame()

    cp = pd.DataFrame(records)
    cp["era"] = cp["year"].map(era_of_year)
    return cp.sort_values(["metric", "year"])


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

ERA_VLINES = [1760, 1840]  # era boundaries


def _add_era_shading(ax: plt.Axes, ymin: float = 0, ymax: float = 1) -> None:
    ax.axvspan(1760, 1840, alpha=0.06, color="steelblue", zorder=0)
    for yr in ERA_VLINES:
        ax.axvline(yr, color="steelblue", lw=0.8, ls="--", alpha=0.5)


def plot_direction_balance(direction_wide: pd.DataFrame) -> None:
    if direction_wide.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: stacked bar (income vs expenditure amount)
    ax = axes[0]
    ax.fill_between(direction_wide["year"], direction_wide["amount_sum_income"],
                    label="Income", alpha=0.75, color="#4daf4a")
    ax.fill_between(direction_wide["year"], -direction_wide["amount_sum_expenditure"],
                    label="Expenditure", alpha=0.75, color="#e41a1c")
    ax.axhline(0, color="black", lw=0.6)
    _add_era_shading(ax)
    ax.set_ylabel("£ nominal")
    ax.set_title("Income vs Expenditure (nominal £)")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: net balance (real)
    ax = axes[1]
    net = direction_wide["net_amount_real"].fillna(0)
    ax.fill_between(direction_wide["year"], net, where=net >= 0,
                    color="#4daf4a", alpha=0.6, label="Net surplus (real)")
    ax.fill_between(direction_wide["year"], net, where=net < 0,
                    color="#e41a1c", alpha=0.6, label="Net deficit (real)")
    ax.axhline(0, color="black", lw=0.6)
    _add_era_shading(ax)
    ax.set_ylabel("£ constant 1700")
    ax.set_title("Net Balance – Income minus Expenditure (constant 1700 £)")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3: income share of entries
    ax = axes[2]
    ax.plot(direction_wide["year"], direction_wide["income_share"],
            color="purple", lw=1.2)
    _add_era_shading(ax)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Income entries / total")
    ax.set_title("Income Share of Entry Count")
    ax.set_xlabel("Year")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "direction_balance.png")
    plt.close(fig)
    print("  Saved direction_balance.png")


def plot_category_era_heatmap(era_cat: pd.DataFrame) -> None:
    if era_cat.empty:
        return

    pivot = era_cat.pivot_table(
        index="category", columns="era", values="entry_share", fill_value=0
    )
    # Reorder
    col_order = [c for c in ["pre_1760", "industrial_1760_1840", "post_1840"]
                 if c in pivot.columns]
    row_order = [r for r in CATEGORY_ORDER if r in pivot.index]
    pivot = pivot.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="YlOrRd",
                linewidths=0.4, ax=ax,
                cbar_kws={"format": mticker.PercentFormatter(xmax=1)})
    ax.set_title("Entry Share by Category and Era")
    ax.set_xlabel("")
    ax.set_ylabel("Category")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "category_era_heatmap.png")
    plt.close(fig)
    print("  Saved category_era_heatmap.png")


def plot_category_timeseries(yearly_cat: pd.DataFrame) -> None:
    if yearly_cat.empty:
        return

    cats = [c for c in CATEGORY_ORDER if c in yearly_cat["category"].unique()]
    # Pivot to wide for stacked area
    pivot = yearly_cat.pivot_table(
        index="year", columns="category", values="entry_share", fill_value=0
    )
    pivot = pivot.reindex(columns=cats, fill_value=0)

    colors = sns.color_palette("tab10", len(cats))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(pivot.index, [pivot[c] for c in cats],
                 labels=cats, colors=colors, alpha=0.8)
    _add_era_shading(ax)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Share of entry rows")
    ax.set_title("Category Composition Over Time (entry share)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "category_timeseries.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved category_timeseries.png")


def plot_language_shift(lang_df: pd.DataFrame) -> None:
    if lang_df.empty:
        return

    lang_colors = {"latin": "#d62728", "english": "#1f77b4", "mixed": "#ff7f0e"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Panel 1: raw entry count per language
    ax = axes[0]
    for lang, grp in lang_df.groupby("language"):
        ax.plot(grp["year"], grp["n_entries"], label=lang,
                color=lang_colors.get(lang, "grey"), lw=1.2, alpha=0.9)
    _add_era_shading(ax)
    ax.set_ylabel("Weighted entry count")
    ax.set_title("Entry Count by Language")
    ax.legend(fontsize=8)

    # Panel 2: share
    ax = axes[1]
    for lang, grp in lang_df.groupby("language"):
        ax.plot(grp["year"], grp["share"], label=lang,
                color=lang_colors.get(lang, "grey"), lw=1.2, alpha=0.9)
    _add_era_shading(ax)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Share of entries")
    ax.set_title("Language Share Over Time")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "language_shift.png")
    plt.close(fig)
    print("  Saved language_shift.png")


def plot_arrears(arrears: pd.DataFrame) -> None:
    if arrears.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax = axes[0]
    ax.plot(arrears["year"], arrears["arrears_rate"] * 100,
            color="#e41a1c", lw=1.2)
    # 10-yr rolling mean
    roll = arrears.set_index("year")["arrears_rate"].rolling(10, center=True).mean()
    ax.plot(roll.index, roll * 100, color="darkred", lw=2, ls="--",
            label="10-yr rolling mean")
    _add_era_shading(ax)
    ax.set_ylabel("% of entries")
    ax.set_title("Arrears Rate (entries flagged as is_arrears=True)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(arrears["year"], arrears["arrears_amount_share"] * 100,
            color="#e41a1c", lw=1.2)
    _add_era_shading(ax)
    ax.set_ylabel("% of total amount")
    ax.set_title("Arrears Amount Share")
    ax.set_xlabel("Year")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "arrears_rate.png")
    plt.close(fig)
    print("  Saved arrears_rate.png")


def plot_payment_period(pp_df: pd.DataFrame) -> None:
    if pp_df.empty:
        return

    era_order = ["pre_1760", "industrial_1760_1840", "post_1840"]
    era_agg = (
        pp_df.groupby(["era", "payment_period"], as_index=False)
        .agg(n_entries=("n_entries", "sum"))
    )
    era_total = era_agg.groupby("era")["n_entries"].sum().rename("total")
    era_agg = era_agg.join(era_total, on="era")
    era_agg["share"] = era_agg["n_entries"] / era_agg["total"]
    era_agg["era"] = pd.Categorical(era_agg["era"], categories=era_order, ordered=True)

    pivot = era_agg.pivot_table(
        index="payment_period", columns="era", values="share", fill_value=0
    )
    pivot = pivot[[c for c in era_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="barh", ax=ax, colormap="Set2", width=0.7)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Share of entries")
    ax.set_title("Payment Period Distribution by Era")
    ax.legend(title="Era", fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "payment_period_era.png")
    plt.close(fig)
    print("  Saved payment_period_era.png")


def plot_top_places_persons(
    place_agg: pd.DataFrame,
    person_agg: pd.DataFrame,
    top_n: int = 15,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    era_order = ["pre_1760", "industrial_1760_1840", "post_1840"]
    era_labels = {"pre_1760": "Pre-1760",
                  "industrial_1760_1840": "1760–1840",
                  "post_1840": "Post-1840"}

    for ax, era in zip(axes, era_order):
        sub = place_agg[place_agg["era"] == era].nlargest(top_n, "n_entries")
        if sub.empty:
            ax.set_visible(False)
            continue
        bars = ax.barh(sub["name"].tolist()[::-1],
                       sub["n_entries"].tolist()[::-1],
                       color="#4878cf", alpha=0.8)
        ax.set_title(f"Top {top_n} Places – {era_labels.get(era, era)}", fontsize=9)
        ax.set_xlabel("Weighted entry count")
        ax.tick_params(labelsize=7)

    fig.suptitle("Most Frequent Place Names by Era", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_places_by_era.png")
    plt.close(fig)
    print("  Saved top_places_by_era.png")

    # Person names
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, era in zip(axes, era_order):
        sub = person_agg[person_agg["era"] == era].nlargest(top_n, "n_entries")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.barh(sub["name"].tolist()[::-1],
                sub["n_entries"].tolist()[::-1],
                color="#e49444", alpha=0.8)
        ax.set_title(f"Top {top_n} Persons – {era_labels.get(era, era)}", fontsize=9)
        ax.set_xlabel("Weighted entry count")
        ax.tick_params(labelsize=7)

    fig.suptitle("Most Frequent Person / Role Names by Era", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_persons_by_era.png")
    plt.close(fig)
    print("  Saved top_persons_by_era.png")


def plot_change_points(cp: pd.DataFrame) -> None:
    if cp.empty:
        return

    flagged = cp[cp["is_change_point"]].copy()
    if flagged.empty:
        return

    metrics = flagged["metric"].unique()
    fig, axes = plt.subplots(
        len(metrics), 1,
        figsize=(12, 3 * len(metrics)),
        sharex=True,
        squeeze=False,
    )

    for ax, metric in zip(axes[:, 0], metrics):
        full = cp[cp["metric"] == metric].sort_values("year")
        ax.plot(full["year"], full["value"], color="steelblue", lw=1.0, alpha=0.7)
        pts = flagged[flagged["metric"] == metric]
        ax.scatter(pts["year"], pts["value"], color="#e41a1c", zorder=5, s=30)
        _add_era_shading(ax)
        ax.set_ylabel(metric, fontsize=8)

    axes[-1, 0].set_xlabel("Year")
    fig.suptitle("Enriched Metric Change Points (robust z ≥ 2.5)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "enriched_change_points.png")
    plt.close(fig)
    print("  Saved enriched_change_points.png")


def plot_category_amount_era(era_cat: pd.DataFrame) -> None:
    """Bar chart: total real amount by category, faceted by era."""
    if era_cat.empty:
        return

    era_order = ["pre_1760", "industrial_1760_1840", "post_1840"]
    era_labels = {"pre_1760": "Pre-1760",
                  "industrial_1760_1840": "1760–1840",
                  "post_1840": "Post-1840"}
    cats = [c for c in CATEGORY_ORDER if c in era_cat["category"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, era in zip(axes, era_order):
        sub = era_cat[era_cat["era"] == era].copy()
        sub = sub.set_index("category").reindex(cats).fillna(0)
        ax.bar(range(len(cats)), sub["amount_real"],
               color=sns.color_palette("tab10", len(cats)), alpha=0.85)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
        ax.set_title(era_labels.get(era, era))
        ax.set_ylabel("£ constant 1700")

    fig.suptitle("Total Real Amount by Category and Era", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "category_amount_by_era.png")
    plt.close(fig)
    print("  Saved category_amount_by_era.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Enriched Ledger Analysis")
    print("=" * 60)

    # 1. Load data
    df = load_enriched_data()

    # Save master flat CSV
    master_path = OUT_DIR / "all_enriched_entries.csv"
    df.to_csv(master_path, index=False)
    print(f"  Saved {master_path.name} ({len(df):,} rows)")

    # 2. Direction analysis
    print("\n[1/8] Direction balance …")
    dir_wide = analyse_direction(df)
    if not dir_wide.empty:
        dir_wide.to_csv(OUT_DIR / "yearly_direction_balance.csv", index=False)
        print(f"       {len(dir_wide)} year rows saved")

    # 3. Category analysis
    print("[2/8] Category breakdown …")
    yearly_cat, era_cat = analyse_categories(df)
    yearly_cat.to_csv(OUT_DIR / "yearly_category_shares.csv", index=False)
    era_cat.to_csv(OUT_DIR / "era_category_shares.csv", index=False)
    print(f"       {len(yearly_cat)} yearly-category rows, "
          f"{len(era_cat)} era-category rows")

    # 4. Language shift
    print("[3/8] Language shift …")
    lang_df = analyse_language(df)
    if not lang_df.empty:
        lang_df.to_csv(OUT_DIR / "yearly_language_shares.csv", index=False)

        # Find the year Latin share fell below English share
        lang_wide = lang_df.pivot_table(
            index="year", columns="language", values="share", fill_value=0
        ).reset_index()
        if "latin" in lang_wide and "english" in lang_wide:
            crossover = lang_wide[lang_wide["english"] > lang_wide["latin"]]
            if not crossover.empty:
                first = int(crossover.iloc[0]["year"])
                print(f"       Latin→English crossover year: {first}")

    # 5. Payment period
    print("[4/8] Payment period …")
    pp_df = analyse_payment_period(df)
    if not pp_df.empty:
        pp_df.to_csv(OUT_DIR / "yearly_payment_period_shares.csv", index=False)

    # 6. Arrears
    print("[5/8] Arrears rate …")
    arrears = analyse_arrears(df)
    arrears.to_csv(OUT_DIR / "yearly_arrears_rate.csv", index=False)

    # 7. Place & person names
    print("[6/8] Place and person names …")
    place_agg, person_agg = analyse_names(df)
    if not place_agg.empty:
        place_agg.to_csv(OUT_DIR / "place_name_frequency.csv", index=False)
        print(f"       {len(place_agg)} place-name rows")
    if not person_agg.empty:
        person_agg.to_csv(OUT_DIR / "person_name_frequency.csv", index=False)
        print(f"       {len(person_agg)} person-name rows")

    # 8. Category × direction
    print("[7/8] Category × direction …")
    cat_dir = analyse_category_direction(df)
    if not cat_dir.empty:
        cat_dir.to_csv(OUT_DIR / "category_direction_era.csv", index=False)

    # 9. Change-point detection on enriched metrics
    print("[8/8] Enriched change-point detection …")
    lang_wide_for_cp = pd.DataFrame()
    if not lang_df.empty:
        lang_wide_for_cp = lang_df.pivot_table(
            index="year", columns="language", values="share", fill_value=0
        ).reset_index()
        lang_wide_for_cp.columns.name = None
    cp = detect_enriched_change_points(dir_wide, lang_df, arrears)
    if not cp.empty:
        cp.to_csv(OUT_DIR / "enriched_change_points.csv", index=False)
        flagged = cp[cp["is_change_point"]].sort_values("abs_delta_robust_z", ascending=False)
        print(f"       {len(flagged)} change points detected across "
              f"{flagged['metric'].nunique()} metrics")
        if not flagged.empty:
            print("       Top 10 by robust z:")
            for _, r in flagged.head(10).iterrows():
                print(f"         {int(r['year'])}  {r['metric']:30s}  z={r['abs_delta_robust_z']:.2f}")

    # ---------------------------------------------------------------------------
    # Visualisations
    # ---------------------------------------------------------------------------
    print("\nGenerating visualisations …")

    plot_direction_balance(dir_wide)
    plot_category_era_heatmap(era_cat)
    plot_category_timeseries(yearly_cat)
    plot_language_shift(lang_df)
    plot_arrears(arrears)
    if not pp_df.empty:
        plot_payment_period(pp_df)
    if not place_agg.empty and not person_agg.empty:
        plot_top_places_persons(place_agg, person_agg)
    if not cp.empty:
        plot_change_points(cp)
    plot_category_amount_era(era_cat)

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    total_entries = float(df["year_weight"].sum())
    print(f"Total entry rows (weighted):  {total_entries:,.0f}")
    print(f"Years covered:                {df['year'].min()}–{df['year'].max()}")
    print(f"Pages:                        {df['page_id'].nunique():,}")

    print("\nDirection distribution (weighted entry count):")
    dir_counts = (
        df[df["direction"].notna()]
        .groupby("direction")["year_weight"].sum()
        .sort_values(ascending=False)
    )
    for d, n in dir_counts.items():
        print(f"  {d:20s}: {n:8,.0f}  ({100*n/dir_counts.sum():.1f}%)")

    print("\nCategory distribution (weighted entry count, top 10):")
    cat_counts = (
        df[df["category"].notna()]
        .groupby("category")["year_weight"].sum()
        .sort_values(ascending=False)
    )
    for c, n in cat_counts.items():
        print(f"  {c:20s}: {n:8,.0f}  ({100*n/cat_counts.sum():.1f}%)")

    print("\nLanguage distribution (weighted entry count):")
    lang_counts = (
        df[df["language"].notna()]
        .groupby("language")["year_weight"].sum()
        .sort_values(ascending=False)
    )
    for l, n in lang_counts.items():
        print(f"  {l:20s}: {n:8,.0f}  ({100*n/lang_counts.sum():.1f}%)")

    print("\nArrears summary by era:")
    era_arr = (
        df[df["is_arrears"].notna()]
        .groupby("era")
        .apply(lambda g: pd.Series({
            "arrears_rate": float(g["is_arrears"].astype(bool).multiply(g["year_weight"]).sum()
                                  / g["year_weight"].sum()),
            "n_entries": float(g["year_weight"].sum()),
        }))
        .reset_index()
    )
    for _, row in era_arr.iterrows():
        print(f"  {row['era']:30s}: arrears rate = {row['arrears_rate']*100:.1f}%  "
              f"(n={row['n_entries']:,.0f})")

    print(f"\nOutputs written to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
