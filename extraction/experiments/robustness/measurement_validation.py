"""
experiments/robustness/measurement_validation.py

Tests whether the key findings from the enriched analysis are robust to
reasonable alternative choices in data construction and processing.

Parameter dimensions tested
----------------------------
1. Era boundaries
   Baseline:  pre_1760 / 1760–1840 / post_1840
   Alt A:     pre_1750 / 1750–1850 / post_1850  (shift both boundaries by -10 yr)
   Alt B:     pre_1780 / 1780–1840 / post_1840  (conservative industrial start)
   Alt C:     pre_1760 / 1760–1800 / 1800–1840 / post_1840 (finer split at 1800)

2. Price deflation index
   Baseline:  Phelps Brown–Hopkins consumables basket (interpolated)
   Alt A:     No deflation — nominal pounds only
   Alt B:     Allen Southern England consumables basket (alternative series)

3. Year-weight for multi-year pages
   Baseline:  Equal weight across all years in the span (1/n_years each)
   Alt A:     Assign all weight to the first (earliest) year
   Alt B:     Assign all weight to the last (latest) year

4. Arrears treatment
   Baseline:  Include all entries including is_arrears=True
   Alt A:     Exclude arrears entries from income/direction analysis

5. Section-header direction validation
   (Not a parameter sweep — a data-quality check)
   For entries where section_header implies a known direction
   (Recepta → income; Soluta/Stipendia → expenditure), compare the
   LLM-assigned direction against the header-implied direction.

6. Change-point detection threshold
   Baseline:  robust z-score ≥ 2.5
   Alt A:     z ≥ 2.0 (more sensitive)
   Alt B:     z ≥ 3.0 (more conservative)

Outputs
-------
experiments/reports/robustness/measurement/
  era_sensitivity.csv          — category shares under different era cuts
  deflation_sensitivity.csv    — real income/expenditure under different deflators
  yearweight_sensitivity.csv   — category shares under different year-weight schemes
  arrears_sensitivity.csv      — direction balance with/without arrears
  header_validation.csv        — LLM direction vs header-implied direction
  changepoint_sensitivity.csv  — change points detected at different thresholds
  measurement_summary.txt      — human-readable summary of all checks

Usage
-----
    .venv/bin/python -m experiments.robustness.measurement_validation
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
ENRICHED_DIR = ROOT / "experiments" / "results" / "enriched"
OUT_DIR      = ROOT / "experiments" / "reports" / "robustness" / "measurement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Price index definitions
# ---------------------------------------------------------------------------
#
# IMPORTANT NOTE ON DATA PROVENANCE
# ----------------------------------
# The baseline index used in analysis_enriched.py is labelled "Phelps Brown-Hopkins"
# but a cross-check against the published PBH series (rebased to 1700=100 from
# the original 1451-1475=100 base, as reported in Phelps Brown & Hopkins 1956,
# Economica 23(92):296-314) reveals systematic discrepancies for 1820-1850:
# the baseline values are 10-20% LOWER than the genuine PBH series for that
# period (e.g. 1840: baseline=170.3, genuine PBH≈212, ONS/Feinstein≈213).
# This pattern is instead consistent with Clark (2005, J.Pol.Econ. 113(6):
# 1307-1340), which takes a more "optimistic" view of post-Napoleonic price
# deflation. The baseline is therefore relabelled as "Clark-consistent" below.
#
# Alternative index used for sensitivity testing:
# O'Donoghue, Goulding & Allen (2004) ONS composite series ("Consumer Price
# Inflation Since 1750", Economic Trends 604:38-46), which uses Feinstein (1972)
# as its backbone for 1750-1870 and represents the "pessimist" estimate of
# real price levels during the Industrial Revolution. Values rebased to 1700=100
# from the ONS composite annual series.

# Baseline: values from analysis_enriched.py (Clark-consistent, rebased 1700=100)
_BASELINE: dict[int, float] = {
    1700: 100.0, 1710: 103.7, 1720: 101.3, 1730:  93.6,
    1740: 100.0, 1750: 104.7, 1760: 115.7, 1770: 125.4,
    1780: 138.3, 1790: 145.2, 1800: 203.4, 1810: 269.0,
    1820: 213.7, 1830: 175.4, 1840: 170.3, 1850: 161.2,
    1860: 175.4, 1870: 193.0, 1880: 182.4, 1890: 160.0,
    1900: 169.7,
}

# Alternative: O'Donoghue et al. (2004) ONS/Feinstein composite, rebased 1700=100
# Key difference: 1820-1850 is 10-20% higher than the baseline, reflecting
# Feinstein's more pessimistic estimate of post-Napoleonic price deflation.
_ODONOGHUE: dict[int, float] = {
    1700: 100.0, 1710: 103.0, 1720: 103.5, 1730:  96.0,
    1740: 102.0, 1750: 107.0, 1760: 113.3, 1770: 120.7,
    1780: 126.8, 1790: 144.0, 1800: 193.6, 1810: 271.0,
    1820: 250.5, 1830: 200.1, 1840: 212.9, 1850: 174.3,
    1860: 174.3, 1870: 184.0, 1880: 177.4, 1890: 171.3,
    1900: 170.3,
}

# Nominal: no deflation applied (income share is a count ratio so this is
# mathematically identical to any deflator, but included for completeness)
def _nominal_index(year: int) -> float:
    return 100.0


def _interp_index(table: dict[int, float]) -> "callable[[int], float]":
    yrs  = sorted(table)
    vals = [table[y] for y in yrs]
    def fn(year: int) -> float:
        if year <= yrs[0]:  return vals[0]
        if year >= yrs[-1]: return vals[-1]
        return float(np.interp(year, yrs, vals))
    return fn


PRICE_INDICES = {
    "baseline_clark":   _interp_index(_BASELINE),
    "odonoghue_feinstein": _interp_index(_ODONOGHUE),
    "nominal":          _nominal_index,
}

# ---------------------------------------------------------------------------
# Era boundary schemes
# ---------------------------------------------------------------------------

ERA_SCHEMES: dict[str, "callable[[int], str]"] = {
    "baseline": lambda y: (
        "pre_1760" if y < 1760 else
        "industrial_1760_1840" if y <= 1840 else
        "post_1840"
    ),
    "alt_a_shifted": lambda y: (
        "pre_1750" if y < 1750 else
        "mid_1750_1850" if y <= 1850 else
        "post_1850"
    ),
    "alt_b_late_start": lambda y: (
        "pre_1780" if y < 1780 else
        "industrial_1780_1840" if y <= 1840 else
        "post_1840"
    ),
    "alt_c_finer": lambda y: (
        "pre_1760" if y < 1760 else
        "early_industrial_1760_1800" if y <= 1800 else
        "late_industrial_1800_1840" if y <= 1840 else
        "post_1840"
    ),
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_money(v: Any) -> float:
    if v is None or v == "": return 0.0
    if isinstance(v, (int, float)):
        vf = float(v)
        return 0.0 if np.isnan(vf) else vf
    s = re.sub(r"[^0-9.\-]", "", str(v).strip())
    try: return float(s)
    except ValueError: return 0.0


def parse_fraction(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)): return 0.0
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().lower()
    for frac, val in [("¼", 0.25), ("1/4", 0.25), ("½", 0.5), ("1/2", 0.5),
                      ("¾", 0.75), ("3/4", 0.75)]:
        if s == frac: return val
    try: return float(s)
    except ValueError: return 0.0


def amount_decimal(row: dict) -> float:
    p = parse_money(row.get("amount_pounds"))
    s = parse_money(row.get("amount_shillings"))
    d = parse_money(row.get("amount_pence_whole"))
    f = parse_fraction(row.get("amount_pence_fraction"))
    return p + s / 20.0 + (d + f) / 240.0


def parse_page_id(pid: str) -> tuple[list[int], int]:
    if m := re.match(r"^(\d{4})_(\d+)_image$", pid):
        return [int(m.group(1))], int(m.group(2))
    if m := re.match(r"^(\d{4})-(\d{4})_(\d+)_image$", pid):
        y1, y2, pg = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return list(range(min(y1,y2), max(y1,y2)+1)), pg
    if m := re.search(r"(\d{4})", pid):
        return [int(m.group(1))], 1
    raise ValueError(f"Cannot parse: {pid!r}")


# ---------------------------------------------------------------------------
# Data loading — load once, apply different settings downstream
# ---------------------------------------------------------------------------

def load_all_entries() -> pd.DataFrame:
    """Load all enriched files into a flat DataFrame (raw fields, no derived columns)."""
    files = sorted(ENRICHED_DIR.glob("*_image_enriched.json"))
    if not files:
        raise FileNotFoundError(f"No enriched JSON files in {ENRICHED_DIR}")

    print(f"Loading {len(files)} enriched files...")
    records = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as e:
            print(f"  SKIP {fp.name}: {e}")
            continue

        page_id = payload.get("page_id") or fp.stem.replace("_enriched", "")
        try:
            years, _ = parse_page_id(page_id)
        except ValueError:
            continue

        for row in payload.get("rows", []):
            if not isinstance(row, dict): continue
            if row.get("row_type") != "entry": continue

            amt = amount_decimal(row)
            records.append({
                "page_id":       page_id,
                "row_index":     row.get("row_index"),
                "n_years":       len(years),
                "years_list":    years,  # store as list; will explode below
                "amount_raw":    amt,
                "direction":     row.get("direction"),
                "category":      row.get("category"),
                "language":      row.get("language"),
                "payment_period": row.get("payment_period"),
                "is_arrears":    row.get("is_arrears"),
                "section_header": row.get("section_header", ""),
                "description":   row.get("description", ""),
            })

    df_raw = pd.DataFrame(records)
    print(f"  Loaded {len(df_raw):,} raw entry rows")

    # Explode years: one row per (page_id, row_index, year)
    rows_exploded = []
    for _, r in df_raw.iterrows():
        for yr in r["years_list"]:
            rows_exploded.append({
                "page_id":       r["page_id"],
                "row_index":     r["row_index"],
                "year":          yr,
                "year_weight":   1.0 / r["n_years"],
                "amount_raw":    r["amount_raw"],
                "direction":     r["direction"],
                "category":      r["category"],
                "language":      r["language"],
                "payment_period": r["payment_period"],
                "is_arrears":    r["is_arrears"],
                "section_header": r["section_header"],
                "description":   r["description"],
            })

    df = pd.DataFrame(rows_exploded)
    df["decade"] = ((df["year"] // 10) * 10).astype(int)
    print(f"  Exploded to {len(df):,} year-rows")
    return df


# ---------------------------------------------------------------------------
# Derived columns under different settings
# ---------------------------------------------------------------------------

def apply_settings(
    df: pd.DataFrame,
    era_fn: "callable[[int], str]",
    price_fn: "callable[[int], float]",
    year_weight_mode: str = "equal",
    exclude_arrears: bool = False,
) -> pd.DataFrame:
    """Return a copy of df with settings-specific derived columns."""
    out = df.copy()

    # Year weight
    if year_weight_mode == "equal":
        pass  # already set
    elif year_weight_mode == "first_year_only":
        # Re-weight so only the first (min) year row gets full weight
        # Since the data is already exploded, we flag the first year per entry
        min_year = out.groupby(["page_id", "row_index"])["year"].transform("min")
        out["year_weight"] = np.where(out["year"] == min_year, 1.0, 0.0)
    elif year_weight_mode == "last_year_only":
        max_year = out.groupby(["page_id", "row_index"])["year"].transform("max")
        out["year_weight"] = np.where(out["year"] == max_year, 1.0, 0.0)

    # Era
    out["era"] = out["year"].map(era_fn)

    # Price index & real amount
    out["price_idx"]     = out["year"].map(price_fn)
    out["amount_nominal"] = out["amount_raw"] * out["year_weight"]
    out["amount_real"]    = out["amount_nominal"] / (out["price_idx"] / 100.0)

    # Arrears filter
    if exclude_arrears:
        out = out[~out["is_arrears"].astype(bool)]

    return out


# ---------------------------------------------------------------------------
# Key metrics per setting
# ---------------------------------------------------------------------------

def metric_category_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Category share (by entry count, weighted) per era."""
    cat = df[df["category"].notna()].copy()
    grp = cat.groupby(["era", "category"])["year_weight"].sum().reset_index()
    total = cat.groupby("era")["year_weight"].sum().rename("total")
    grp = grp.join(total, on="era")
    grp["share"] = grp["year_weight"] / grp["total"]
    return grp[["era", "category", "share"]].sort_values(["era", "category"])


def metric_direction_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Income share and net real amount per decade."""
    d = df[df["direction"].isin(["income", "expenditure"])].copy()
    pivot = d.groupby(["decade", "direction"]).agg(
        n_entries   = ("year_weight", "sum"),
        amount_real = ("amount_real", "sum"),
    ).unstack(fill_value=0)
    pivot.columns = ["_".join(c) for c in pivot.columns]
    for col in ["n_entries_income", "n_entries_expenditure",
                "amount_real_income", "amount_real_expenditure"]:
        if col not in pivot.columns: pivot[col] = 0.0
    pivot["income_share"]  = pivot["n_entries_income"] / (
        pivot["n_entries_income"] + pivot["n_entries_expenditure"]
    ).replace(0, np.nan)
    pivot["net_real"] = pivot["amount_real_income"] - pivot["amount_real_expenditure"]
    return pivot.reset_index()[["decade", "income_share", "net_real"]]


def metric_language_shares(df: pd.DataFrame) -> pd.DataFrame:
    """English language share per decade."""
    lang = df[df["language"].isin(["latin", "english", "mixed"])].copy()
    grp = lang.groupby(["decade", "language"])["year_weight"].sum().unstack(fill_value=0)
    total = grp.sum(axis=1)
    for lg in ["english", "latin", "mixed"]:
        if lg not in grp.columns: grp[lg] = 0.0
    grp["english_share"] = grp["english"] / total.replace(0, np.nan)
    return grp[["english_share"]].reset_index()


def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = float(np.median(np.abs(x - med)))
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


def metric_change_points(
    direction_wide: pd.DataFrame,
    lang_wide: pd.DataFrame,
    threshold: float = 2.5,
) -> pd.DataFrame:
    """Detect change points in income_share and english_share at given z threshold."""
    records = []

    def _check(sub: pd.DataFrame, year_col: str, val_col: str, label: str) -> None:
        sub = sub[[year_col, val_col]].dropna().sort_values(year_col)
        if len(sub) < 5: return
        diff = sub[val_col].diff().fillna(0.0)
        rz   = robust_zscore(diff.abs())
        for i in range(len(sub)):
            records.append({
                "year":     int(sub.iloc[i][year_col]),
                "metric":   label,
                "value":    float(sub.iloc[i][val_col]),
                "abs_z":    float(rz.iloc[i]),
                "is_cp":    bool(rz.iloc[i] >= threshold),
            })

    if not direction_wide.empty:
        _check(direction_wide, "decade", "income_share", "income_share")
    if not lang_wide.empty:
        _check(lang_wide, "decade", "english_share", "english_share")

    df = pd.DataFrame(records)
    return df[df["is_cp"]] if not df.empty else df


# ---------------------------------------------------------------------------
# Section-header direction validation
# ---------------------------------------------------------------------------

# Map section header keywords → implied direction
_HEADER_INCOME_KW   = {"recepta", "arrearagia recepta", "arrearagia"}
_HEADER_EXPEND_KW   = {"soluta", "stipendia", "soluta ordinaria", "soluta varia"}


def header_implied_direction(header: str) -> str | None:
    """Return 'income', 'expenditure', or None if header doesn't imply direction."""
    if not header:
        return None
    h = header.strip().lower()
    # Check expenditure keywords first (they tend to be more specific)
    for kw in _HEADER_EXPEND_KW:
        if h.startswith(kw):
            return "expenditure"
    for kw in _HEADER_INCOME_KW:
        if kw in h:
            return "income"
    return None


def validate_header_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Compare LLM-assigned direction vs. section-header-implied direction."""
    df = df.copy()
    df["header_direction"] = df["section_header"].apply(header_implied_direction)

    # Only rows where header gives a clear signal
    has_signal = df["header_direction"].notna()
    llm_clear  = df["direction"].isin(["income", "expenditure"])

    sub = df[has_signal & llm_clear].copy()
    if sub.empty:
        return pd.DataFrame({"n_signal_rows": [0], "agreement_rate": [float("nan")]})

    sub["agree"] = sub["direction"] == sub["header_direction"]

    overall = sub["agree"].mean()
    by_header = (
        sub.groupby("header_direction")["agree"]
        .agg(n_rows="count", agree_rate="mean")
        .reset_index()
    )
    by_header.loc[len(by_header)] = ["OVERALL", len(sub), overall]

    confusion = pd.crosstab(
        sub["header_direction"], sub["direction"],
        rownames=["header_implies"], colnames=["llm_says"],
    )
    return by_header, confusion


# ---------------------------------------------------------------------------
# Main sensitivity sweep
# ---------------------------------------------------------------------------

def run_sensitivity_sweep(df_base: pd.DataFrame) -> None:
    """Run all sensitivity checks and save CSVs."""

    summary_lines: list[str] = []

    # ------------------------------------------------------------------
    # 1. Era boundary sensitivity
    # ------------------------------------------------------------------
    print("\n[1/6] Era boundary sensitivity...")
    era_rows = []
    for scheme_name, era_fn in ERA_SCHEMES.items():
        applied = apply_settings(df_base, era_fn=era_fn,
                                 price_fn=PRICE_INDICES["baseline_clark"])
        cat_shares = metric_category_shares(applied)
        for _, r in cat_shares.iterrows():
            era_rows.append({
                "scheme":   scheme_name,
                "era":      r["era"],
                "category": r["category"],
                "share":    round(float(r["share"]), 4),
            })

    era_df = pd.DataFrame(era_rows)
    era_df.to_csv(OUT_DIR / "era_sensitivity.csv", index=False)

    # Pivot for readability: land_rent share per era per scheme
    land_pivot = era_df[era_df["category"] == "land_rent"].pivot_table(
        index="era", columns="scheme", values="share"
    )
    summary_lines.append("\n=== ERA SENSITIVITY (land_rent share) ===")
    summary_lines.append(land_pivot.to_string())

    # ------------------------------------------------------------------
    # 2. Price deflation sensitivity
    # ------------------------------------------------------------------
    print("[2/9] Price deflation sensitivity...")
    defl_rows = []
    for idx_name, price_fn in PRICE_INDICES.items():
        applied = apply_settings(df_base, era_fn=ERA_SCHEMES["baseline"],
                                 price_fn=price_fn)
        bal = metric_direction_balance(applied)
        bal["deflator"] = idx_name
        defl_rows.append(bal)

    defl_df = pd.concat(defl_rows, ignore_index=True)
    defl_df.to_csv(OUT_DIR / "deflation_sensitivity.csv", index=False)

    # Show income_share by decade across deflators
    inc_pivot = defl_df.pivot_table(
        index="decade", columns="deflator", values="income_share"
    )
    summary_lines.append("\n=== DEFLATION SENSITIVITY (income_share by decade) ===")
    summary_lines.append(inc_pivot.to_string())

    # ------------------------------------------------------------------
    # 3. Year-weight sensitivity
    # ------------------------------------------------------------------
    print("[3/9] Year-weight sensitivity...")
    yw_rows = []
    for yw_mode in ["equal", "first_year_only", "last_year_only"]:
        applied = apply_settings(df_base, era_fn=ERA_SCHEMES["baseline"],
                                 price_fn=PRICE_INDICES["baseline_clark"],
                                 year_weight_mode=yw_mode)
        cat_shares = metric_category_shares(applied)
        cat_shares["year_weight_mode"] = yw_mode
        yw_rows.append(cat_shares)

    yw_df = pd.concat(yw_rows, ignore_index=True)
    yw_df.to_csv(OUT_DIR / "yearweight_sensitivity.csv", index=False)

    land_yw_pivot = yw_df[yw_df["category"] == "land_rent"].pivot_table(
        index="era", columns="year_weight_mode", values="share"
    )
    summary_lines.append("\n=== YEAR-WEIGHT SENSITIVITY (land_rent share) ===")
    summary_lines.append(land_yw_pivot.to_string())

    # ------------------------------------------------------------------
    # 4. Arrears treatment
    # ------------------------------------------------------------------
    print("[4/9] Arrears treatment sensitivity...")
    arr_rows = []
    for excl in [False, True]:
        applied = apply_settings(df_base, era_fn=ERA_SCHEMES["baseline"],
                                 price_fn=PRICE_INDICES["baseline_clark"],
                                 exclude_arrears=excl)
        bal = metric_direction_balance(applied)
        bal["exclude_arrears"] = excl
        arr_rows.append(bal)

    arr_df = pd.concat(arr_rows, ignore_index=True)
    arr_df.to_csv(OUT_DIR / "arrears_sensitivity.csv", index=False)

    arr_pivot = arr_df.pivot_table(
        index="decade", columns="exclude_arrears", values="income_share"
    )
    arr_pivot.columns = ["include_arrears", "exclude_arrears"]
    arr_pivot["diff"] = arr_pivot["include_arrears"] - arr_pivot["exclude_arrears"]
    summary_lines.append("\n=== ARREARS SENSITIVITY (income_share by decade) ===")
    summary_lines.append(arr_pivot.round(4).to_string())

    # ------------------------------------------------------------------
    # 5. Section-header direction validation
    # ------------------------------------------------------------------
    print("[5/9] Section-header direction validation...")
    applied_base = apply_settings(df_base, era_fn=ERA_SCHEMES["baseline"],
                                  price_fn=PRICE_INDICES["baseline_clark"])
    by_header, confusion = validate_header_direction(applied_base)
    by_header.to_csv(OUT_DIR / "header_validation.csv", index=False)
    confusion.to_csv(OUT_DIR / "header_confusion_matrix.csv")

    summary_lines.append("\n=== HEADER VALIDATION (LLM direction vs header-implied) ===")
    summary_lines.append(by_header.to_string(index=False))
    summary_lines.append("Confusion matrix:")
    summary_lines.append(confusion.to_string())

    # ------------------------------------------------------------------
    # 6. Change-point threshold sensitivity
    # ------------------------------------------------------------------
    print("[6/9] Change-point threshold sensitivity...")
    applied_base = apply_settings(df_base, era_fn=ERA_SCHEMES["baseline"],
                                  price_fn=PRICE_INDICES["baseline_clark"])
    dir_wide  = metric_direction_balance(applied_base)
    lang_wide = metric_language_shares(applied_base)

    cp_rows = []
    for threshold in [2.0, 2.5, 3.0]:
        cp = metric_change_points(dir_wide, lang_wide, threshold=threshold)
        n_cp = len(cp)
        years = sorted(cp["year"].tolist()) if not cp.empty else []
        cp_rows.append({
            "threshold": threshold,
            "n_change_points": n_cp,
            "change_point_years": str(years),
        })

    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv(OUT_DIR / "changepoint_sensitivity.csv", index=False)
    summary_lines.append("\n=== CHANGE-POINT THRESHOLD SENSITIVITY ===")
    summary_lines.append(cp_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Confidence threshold sensitivity
    # ------------------------------------------------------------------
    print("[7/9] Confidence threshold sensitivity...")
    # Re-load confidence scores from the JSON files (not stored in df_base)
    conf_records = []
    for fp in sorted(ENRICHED_DIR.glob("*_image_enriched.json")):
        try:
            with open(fp, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        page_id = payload.get("page_id") or fp.stem.replace("_enriched", "")
        try:
            years, _ = parse_page_id(page_id)
        except ValueError:
            continue
        for row in payload.get("rows", []):
            if not isinstance(row, dict): continue
            if row.get("row_type") != "entry": continue
            conf = row.get("confidence_score")
            if conf is None: continue
            for yr in years:
                conf_records.append({
                    "page_id":   page_id,
                    "row_index": row.get("row_index"),
                    "year":      yr,
                    "year_weight": 1.0 / len(years),
                    "confidence": float(conf),
                    "direction":  row.get("direction"),
                    "category":   row.get("category"),
                    "is_arrears": row.get("is_arrears"),
                })

    conf_df = pd.DataFrame(conf_records)
    conf_df["decade"] = ((conf_df["year"] // 10) * 10).astype(int)
    conf_df["era"]    = conf_df["year"].map(ERA_SCHEMES["baseline"])

    conf_rows = []
    for threshold, label in [
        (0.0,  "all_entries"),
        (0.90, "conf_ge_0.90"),
        (0.95, "conf_ge_0.95"),
    ]:
        sub = conf_df[conf_df["confidence"] >= threshold] if threshold > 0 else conf_df

        # Land-rent share by era
        cat_sub = sub[sub["category"].notna()].copy()
        total_by_era = cat_sub.groupby("era")["year_weight"].sum()
        land_sub = cat_sub[cat_sub["category"] == "land_rent"].groupby("era")["year_weight"].sum()
        for era in total_by_era.index:
            share = land_sub.get(era, 0.0) / total_by_era[era] if total_by_era[era] > 0 else float("nan")
            conf_rows.append({
                "confidence_threshold": label,
                "era": era,
                "n_entries": int(total_by_era[era]),
                "land_rent_share": round(float(share), 4),
            })

    conf_result = pd.DataFrame(conf_rows)
    conf_result.to_csv(OUT_DIR / "confidence_sensitivity.csv", index=False)

    # Income share by decade
    inc_conf_rows = []
    for threshold, label in [
        (0.0,  "all_entries"),
        (0.90, "conf_ge_0.90"),
        (0.95, "conf_ge_0.95"),
    ]:
        sub = conf_df[conf_df["confidence"] >= threshold] if threshold > 0 else conf_df
        d = sub[sub["direction"].isin(["income", "expenditure"])].copy()
        grp = d.groupby(["decade", "direction"])["year_weight"].sum().unstack(fill_value=0)
        for col in ["income", "expenditure"]:
            if col not in grp.columns: grp[col] = 0.0
        grp["income_share"] = grp["income"] / (grp["income"] + grp["expenditure"]).replace(0, float("nan"))
        grp["threshold"] = label
        grp = grp[["income_share", "threshold"]].reset_index()
        inc_conf_rows.append(grp)

    inc_conf_df = pd.concat(inc_conf_rows, ignore_index=True)
    inc_conf_df.to_csv(OUT_DIR / "confidence_income_share.csv", index=False)

    summary_lines.append("\n=== CONFIDENCE THRESHOLD SENSITIVITY (land_rent share by era) ===")
    pivot_conf = conf_result.pivot_table(
        index="era", columns="confidence_threshold", values="land_rent_share"
    )
    summary_lines.append(pivot_conf.to_string())

    # ------------------------------------------------------------------
    # 8. Sparse-year exclusion sensitivity
    # ------------------------------------------------------------------
    print("[8/9] Sparse-year exclusion sensitivity...")
    # Count unique pages per year
    pages_per_year = df_base.groupby("year")["page_id"].nunique()

    sparse_rows = []
    for min_pages, label in [
        (1,  "all_years"),
        (2,  "min_2_pages"),
        (3,  "min_3_pages"),
        (5,  "min_5_pages"),
    ]:
        included_years = set(pages_per_year[pages_per_year >= min_pages].index)
        sub = df_base[df_base["year"].isin(included_years)].copy()
        sub["era"] = sub["year"].map(ERA_SCHEMES["baseline"])

        # Land-rent share by era
        cat_sub = sub[sub["category"].notna()].copy()
        total_by_era = cat_sub.groupby("era")["year_weight"].sum()
        land_sub = cat_sub[cat_sub["category"] == "land_rent"].groupby("era")["year_weight"].sum()
        for era in total_by_era.index:
            share = land_sub.get(era, 0.0) / total_by_era[era] if total_by_era[era] > 0 else float("nan")
            sparse_rows.append({
                "min_pages_per_year": label,
                "n_years_included":   int((sub["year"].isin(included_years) & True).sum() > 0) if False else len(included_years),
                "era": era,
                "land_rent_share": round(float(share), 4),
            })

    sparse_df = pd.DataFrame(sparse_rows)
    sparse_df.to_csv(OUT_DIR / "sparse_year_sensitivity.csv", index=False)
    pivot_sparse = sparse_df.pivot_table(
        index="era", columns="min_pages_per_year", values="land_rent_share"
    )
    summary_lines.append("\n=== SPARSE-YEAR EXCLUSION SENSITIVITY (land_rent share by era) ===")
    summary_lines.append(pivot_sparse.to_string())

    # ------------------------------------------------------------------
    # 9. Financial category share sensitivity (count + amount)
    # ------------------------------------------------------------------
    print("[9b/9] Financial category share sensitivity...")
    # By era cuts — count share
    fin_era_rows = []
    for scheme_name, era_fn in ERA_SCHEMES.items():
        applied = apply_settings(df_base, era_fn=era_fn,
                                 price_fn=PRICE_INDICES["baseline_clark"])
        cat = applied[applied["category"].notna()].copy()
        cat["era"] = cat["year"].map(era_fn)
        era_total = cat.groupby("era")["year_weight"].sum()
        era_fin   = cat[cat["category"] == "financial"].groupby("era")["year_weight"].sum()
        for era in era_total.index:
            share = era_fin.get(era, 0.0) / era_total[era] if era_total[era] > 0 else float("nan")
            fin_era_rows.append({
                "scheme": scheme_name,
                "era":    era,
                "financial_count_share": round(float(share), 4),
            })

    # By confidence threshold — count share
    conf_fin_rows = []
    for threshold, label in [(0.0, "all_entries"), (0.90, "conf_ge_0.90"), (0.95, "conf_ge_0.95")]:
        conf_records = []
        for fp in sorted(ENRICHED_DIR.glob("*_image_enriched.json")):
            try:
                with open(fp, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue
            page_id = payload.get("page_id") or fp.stem.replace("_enriched", "")
            try:
                years, _ = parse_page_id(page_id)
            except ValueError:
                continue
            for row in payload.get("rows", []):
                if not isinstance(row, dict): continue
                if row.get("row_type") != "entry": continue
                conf = row.get("confidence_score")
                if conf is None: continue
                if float(conf) < threshold: continue
                for yr in years:
                    era = ERA_SCHEMES["baseline"](yr)
                    conf_records.append({
                        "year_weight": 1.0 / len(years),
                        "category": row.get("category"),
                        "era": era,
                    })
        cdf = pd.DataFrame(conf_records)
        if cdf.empty: continue
        cdf_cat = cdf[cdf["category"].notna()]
        era_total = cdf_cat.groupby("era")["year_weight"].sum()
        era_fin   = cdf_cat[cdf_cat["category"] == "financial"].groupby("era")["year_weight"].sum()
        for era in era_total.index:
            share = era_fin.get(era, 0.0) / era_total[era] if era_total[era] > 0 else float("nan")
            conf_fin_rows.append({
                "confidence_threshold": label,
                "era": era,
                "financial_count_share": round(float(share), 4),
            })

    # Amount-based financial share by era (baseline only)
    applied_base = apply_settings(df_base, era_fn=ERA_SCHEMES["baseline"],
                                  price_fn=PRICE_INDICES["baseline_clark"])
    cat_amt = applied_base[applied_base["category"].notna()].copy()
    era_total_amt = cat_amt.groupby("era")["amount_real"].sum()
    fin_amt_share_rows = []
    for cat in cat_amt["category"].unique():
        era_cat_amt = cat_amt[cat_amt["category"] == cat].groupby("era")["amount_real"].sum()
        for era in era_total_amt.index:
            share = era_cat_amt.get(era, 0.0) / era_total_amt[era] if era_total_amt[era] > 0 else float("nan")
            fin_amt_share_rows.append({
                "category": cat,
                "era": era,
                "amount_share": round(float(share), 4),
            })

    fin_era_df   = pd.DataFrame(fin_era_rows)
    conf_fin_df  = pd.DataFrame(conf_fin_rows)
    amt_share_df = pd.DataFrame(fin_amt_share_rows)

    fin_era_df.to_csv(OUT_DIR / "financial_era_sensitivity.csv", index=False)
    conf_fin_df.to_csv(OUT_DIR / "financial_confidence_sensitivity.csv", index=False)
    amt_share_df.to_csv(OUT_DIR / "category_amount_share_by_era.csv", index=False)

    summary_lines.append("\n=== FINANCIAL SHARE BY ERA (count, different era cuts) ===")
    piv = fin_era_df[fin_era_df["scheme"] == "baseline"].copy()
    summary_lines.append(piv.to_string(index=False))
    summary_lines.append("\n=== FINANCIAL SHARE BY ERA (count, confidence thresholds) ===")
    summary_lines.append(conf_fin_df.pivot_table(
        index="era", columns="confidence_threshold", values="financial_count_share"
    ).to_string())
    summary_lines.append("\n=== KEY CATEGORY AMOUNT SHARES BY ERA ===")
    key_cats = ["financial", "land_rent", "educational", "ecclesiastical"]
    summary_lines.append(
        amt_share_df[amt_share_df["category"].isin(key_cats)]
        .pivot_table(index="era", columns="category", values="amount_share")
        .to_string()
    )

    # ------------------------------------------------------------------
    # 9b. Educational direction robustness
    # ------------------------------------------------------------------
    print("[9c/9] Educational direction robustness...")
    edu_rows = []
    # Vary: confidence threshold
    for threshold, label in [(0.0, "all_entries"), (0.90, "conf_ge_0.90"), (0.95, "conf_ge_0.95")]:
        conf_records = []
        for fp in sorted(ENRICHED_DIR.glob("*_image_enriched.json")):
            try:
                with open(fp, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue
            page_id = payload.get("page_id") or fp.stem.replace("_enriched", "")
            try:
                years, _ = parse_page_id(page_id)
            except ValueError:
                continue
            for row in payload.get("rows", []):
                if not isinstance(row, dict): continue
                if row.get("row_type") != "entry": continue
                if row.get("category") != "educational": continue
                if row.get("direction") not in ["income", "expenditure"]: continue
                conf = row.get("confidence_score")
                if conf is None or float(conf) < threshold: continue
                for yr in years:
                    era = ERA_SCHEMES["baseline"](yr)
                    conf_records.append({
                        "year_weight": 1.0 / len(years),
                        "direction": row.get("direction"),
                        "era": era,
                    })
        edf = pd.DataFrame(conf_records)
        if edf.empty: continue
        grp = edf.groupby(["era", "direction"])["year_weight"].sum().unstack(fill_value=0)
        for col in ["income", "expenditure"]:
            if col not in grp.columns: grp[col] = 0.0
        grp["income_pct"] = grp["income"] / (grp["income"] + grp["expenditure"]).replace(0, float("nan"))
        grp["n_entries_income"] = grp["income"].round(1)
        grp["n_entries_expenditure"] = grp["expenditure"].round(1)
        for era, row2 in grp.iterrows():
            edu_rows.append({
                "confidence_threshold": label,
                "era": era,
                "income_pct": round(float(row2["income_pct"]), 4),
                "n_entries_income": float(row2["n_entries_income"]),
                "n_entries_expenditure": float(row2["n_entries_expenditure"]),
            })

    # Vary: era cuts (baseline direction profile)
    for scheme_name, era_fn in ERA_SCHEMES.items():
        sub = df_base[
            (df_base["category"] == "educational") &
            (df_base["direction"].isin(["income", "expenditure"]))
        ].copy()
        sub["era"] = sub["year"].map(era_fn)
        grp = sub.groupby(["era", "direction"])["year_weight"].sum().unstack(fill_value=0)
        for col in ["income", "expenditure"]:
            if col not in grp.columns: grp[col] = 0.0
        grp["income_pct"] = grp["income"] / (grp["income"] + grp["expenditure"]).replace(0, float("nan"))
        for era, row2 in grp.iterrows():
            edu_rows.append({
                "confidence_threshold": f"era_scheme_{scheme_name}",
                "era": era,
                "income_pct": round(float(row2["income_pct"]), 4),
                "n_entries_income": float(grp.loc[era, "income"]),
                "n_entries_expenditure": float(grp.loc[era, "expenditure"]),
            })

    edu_df = pd.DataFrame(edu_rows)
    edu_df.to_csv(OUT_DIR / "educational_direction_robustness.csv", index=False)
    summary_lines.append("\n=== EDUCATIONAL DIRECTION (% income) ROBUSTNESS ===")
    conf_only = edu_df[edu_df["confidence_threshold"].isin(
        ["all_entries", "conf_ge_0.90", "conf_ge_0.95"]
    )]
    if not conf_only.empty:
        piv = conf_only.pivot_table(
            index="era", columns="confidence_threshold", values="income_pct"
        )
        summary_lines.append(piv.to_string())

    # ------------------------------------------------------------------
    # 9c. Language shift timing robustness
    # ------------------------------------------------------------------
    print("[9d/9] Language shift timing robustness...")
    lang_timing_rows = []
    pages_per_year = df_base.groupby("year")["page_id"].nunique()

    for min_pages, label in [(1, "all_years"), (2, "min_2_pages"), (3, "min_3_pages")]:
        included_years = set(pages_per_year[pages_per_year >= min_pages].index)
        sub = df_base[
            df_base["year"].isin(included_years) &
            df_base["language"].isin(["latin", "english", "mixed"])
        ].copy()
        sub["era"] = sub["year"].map(ERA_SCHEMES["baseline"])
        # Decade-level english share
        sub["decade"] = ((sub["year"] // 10) * 10).astype(int)
        grp = sub.groupby(["decade", "language"])["year_weight"].sum().unstack(fill_value=0)
        total = grp.sum(axis=1)
        for lng in ["english", "latin", "mixed"]:
            if lng not in grp.columns: grp[lng] = 0.0
        grp["english_share"] = grp["english"] / total.replace(0, float("nan"))
        # First decade where English exceeds 50%
        over50 = grp[grp["english_share"] >= 0.5]
        first50 = int(over50.index.min()) if not over50.empty else None
        # Era-level English share
        era_grp = sub.groupby(["era", "language"])["year_weight"].sum().unstack(fill_value=0)
        for lng in ["english", "latin", "mixed"]:
            if lng not in era_grp.columns: era_grp[lng] = 0.0
        era_total = era_grp.sum(axis=1)
        era_grp["english_share"] = era_grp["english"] / era_total.replace(0, float("nan"))
        for era, row2 in era_grp.iterrows():
            lang_timing_rows.append({
                "coverage": label,
                "era": era,
                "english_share": round(float(row2["english_share"]), 4),
                "first_decade_over_50pct": first50,
            })

    lang_timing_df = pd.DataFrame(lang_timing_rows)
    lang_timing_df.to_csv(OUT_DIR / "language_shift_timing.csv", index=False)
    summary_lines.append("\n=== LANGUAGE SHIFT TIMING (English share by era) ===")
    if not lang_timing_df.empty:
        piv = lang_timing_df.pivot_table(
            index="era", columns="coverage", values="english_share"
        )
        summary_lines.append(piv.to_string())
        first50_vals = lang_timing_df[["coverage","first_decade_over_50pct"]].drop_duplicates()
        summary_lines.append(first50_vals.to_string(index=False))

    # ------------------------------------------------------------------
    # 9. Income-expenditure balance check (internal validity)
    # ------------------------------------------------------------------
    print("[9/9] Income-expenditure balance check...")
    applied_base = apply_settings(df_base, era_fn=ERA_SCHEMES["baseline"],
                                  price_fn=PRICE_INDICES["baseline_clark"])
    ie = applied_base[applied_base["direction"].isin(["income", "expenditure"])].copy()
    ie_era = ie.groupby(["era", "direction"]).agg(
        n_entries     = ("year_weight", "sum"),
        amount_nominal = ("amount_nominal", "sum"),
    ).unstack(fill_value=0)
    ie_era.columns = ["_".join(c) for c in ie_era.columns]
    for col in ["n_entries_income", "n_entries_expenditure",
                "amount_nominal_income", "amount_nominal_expenditure"]:
        if col not in ie_era.columns: ie_era[col] = 0.0
    ie_era = ie_era.reset_index()
    ie_era["entry_income_share"] = (
        ie_era["n_entries_income"]
        / (ie_era["n_entries_income"] + ie_era["n_entries_expenditure"])
    ).round(4)
    ie_era["amount_income_share"] = (
        ie_era["amount_nominal_income"]
        / (ie_era["amount_nominal_income"] + ie_era["amount_nominal_expenditure"])
    ).round(4)
    ie_era["balance_ratio"] = (
        ie_era["amount_nominal_income"] / ie_era["amount_nominal_expenditure"].replace(0, float("nan"))
    ).round(4)
    ie_era = ie_era[["era", "n_entries_income", "n_entries_expenditure",
                      "entry_income_share", "amount_nominal_income",
                      "amount_nominal_expenditure", "amount_income_share", "balance_ratio"]]
    ie_era.to_csv(OUT_DIR / "income_expenditure_balance.csv", index=False)
    summary_lines.append("\n=== INCOME-EXPENDITURE BALANCE BY ERA ===")
    summary_lines.append(ie_era.to_string(index=False))

    # ------------------------------------------------------------------
    # Write summary
    # ------------------------------------------------------------------
    summary_path = OUT_DIR / "measurement_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("MEASUREMENT VALIDATION SUMMARY\n")
        fh.write("=" * 60 + "\n")
        fh.write(
            "This report shows how key metrics change under alternative\n"
            "parameter choices in data construction and processing.\n"
            "Stable values across alternatives = robust findings.\n\n"
        )
        fh.write("\n".join(summary_lines))
        fh.write("\n")

    print(f"\nAll measurement outputs -> {OUT_DIR}")
    print(f"Summary -> {summary_path}")
    print("\n" + "\n".join(summary_lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df_base = load_all_entries()
    run_sensitivity_sweep(df_base)


if __name__ == "__main__":
    main()
