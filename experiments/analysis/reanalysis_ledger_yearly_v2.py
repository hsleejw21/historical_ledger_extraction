"""reanalysis_ledger_yearly_v2.py
Improvements over v1:
  1. Price deflation: Phelps Brown-Hopkins index (1700=100) converts nominal
     pounds to constant 1700 pounds, testing whether "scale expanded later"
     survives after removing inflation.
  2. Extended stop-word list: removes time-template words (year/years/yr/yrs),
     accounting-format words (balance/total/account/…), and currency-unit words
     (pound/shilling/pence/…) so that retained terms reflect actual transaction
     content.
  3. Relative variables: entry_ratio (n_entry / n_data), data_rows_per_page,
     and header_diversity added to the yearly summary and change-point detection,
     separating "more transactions" from "more detailed accounting".
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "experiments" / "results" / "ledger.xlsx"
OUT_DIR = ROOT / "experiments" / "reports" / "analysis_v2"
CACHE_DIR = ROOT / "experiments" / "results" / "cache"

# ---------------------------------------------------------------------------
# 1. Price deflation: Phelps Brown-Hopkins basket-of-consumables index
#    Source: Phelps Brown & Hopkins (1956), "Seven Centuries of the Prices of
#    Consumables, Compared with Builders' Wage-rates", Economica.
#    Values below are per-decade anchors normalised so that 1700 = 100.
#    Intermediate years are linearly interpolated.
# ---------------------------------------------------------------------------
_PBH_ANCHORS: dict[int, float] = {
    1700: 100.0,
    1710: 103.7,
    1720: 101.3,
    1730:  93.6,
    1740: 100.0,
    1750: 104.7,
    1760: 115.7,
    1770: 125.4,
    1780: 138.3,
    1790: 145.2,
    1800: 203.4,
    1810: 269.0,
    1820: 213.7,
    1830: 175.4,
    1840: 170.3,
    1850: 161.2,
    1860: 175.4,
    1870: 193.0,
    1880: 182.4,
    1890: 160.0,
    1900: 169.7,
}

_anchor_years = sorted(_PBH_ANCHORS)
_anchor_values = [_PBH_ANCHORS[y] for y in _anchor_years]


def price_index(year: int) -> float:
    """Return the Phelps Brown-Hopkins price index for *year* (1700 = 100).
    Values outside [1700, 1900] are clipped to the nearest anchor."""
    if year <= _anchor_years[0]:
        return _anchor_values[0]
    if year >= _anchor_years[-1]:
        return _anchor_values[-1]
    return float(np.interp(year, _anchor_years, _anchor_values))


# ---------------------------------------------------------------------------
# 2. Extended stop-word list
#    Union of sklearn's English stop words + ledger-specific noise words.
# ---------------------------------------------------------------------------
_LEDGER_EXTRA_STOPS: frozenset[str] = frozenset({
    # Time-template words
    "year", "years", "yr", "yrs",
    # Accounting-format / template words
    "total", "totall", "balance", "carried", "forward", "brought",
    "folio", "fol", "ff", "account", "accounts", "acct",
    "received", "paid", "payment", "payments",
    "sum", "sums", "amount", "amounts",
    # Currency-unit words
    "pound", "pounds", "shilling", "shillings", "pence", "penny",
    "sterling", "currency",
    # Generic abbreviations without economic meaning
    "ob", "viz", "ie", "per",
})

STOP_WORDS: list[str] = sorted(ENGLISH_STOP_WORDS | _LEDGER_EXTRA_STOPS)

# Token pattern: require at least one letter and at least 2 characters
# (eliminates pure-digit tokens such as "10", "01", etc.)
_TOKEN_PATTERN = r"(?u)\b[a-z][a-z]+\b"


# ---------------------------------------------------------------------------
# Utility helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def parse_fraction(value: Any) -> float:
    if pd.isna(value):
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


def parse_money_number(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = re.sub(r"[^0-9.\-]", "", str(value).strip())
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def normalize_text(s: Any) -> str:
    if pd.isna(s):
        return ""
    text = str(s).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def era_of_year(year: int) -> str:
    if year < 1760:
        return "pre_1760"
    if 1760 <= year <= 1840:
        return "industrial_1760_1840"
    return "post_1840"


def parse_sheet_years(sheet: str) -> tuple[list[int], int] | None:
    single = re.match(r"^(\d{4})_(\d+)_image$", sheet)
    if single:
        return [int(single.group(1))], int(single.group(2))

    span = re.match(r"^(\d{4})-(\d{4})_(\d+)_image$", sheet)
    if not span:
        return None

    y1, y2, page = int(span.group(1)), int(span.group(2)), int(span.group(3))
    if y2 < y1 and (y1 - y2) > 5:
        y2 = y1 + 1
    if y2 < y1:
        y1, y2 = y2, y1
    return list(range(y1, y2 + 1)), page


def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = float(np.median(np.abs(x - med)))
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


# ---------------------------------------------------------------------------
# Data loading (unchanged)
# ---------------------------------------------------------------------------

def build_side_map(cache_dir: Path) -> dict[tuple[str, int], str]:
    """Build (sheet, row_idx) -> side map from supervisor cache json.

    If a row has side="left" or side="right", we treat that entry as potentially
    coming from a debit/credit two-sided layout and later apply a 0.5 amount factor
    to avoid double counting when both sides are recorded on the same page.
    """
    side_map: dict[tuple[str, int], str] = {}
    if not cache_dir.exists():
        return side_map

    for fp in sorted(cache_dir.glob("*_image_supervisor_gemini-flash.json")):
        page_id = fp.name.replace("_supervisor_gemini-flash.json", "")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            continue

        for r in rows:
            if not isinstance(r, dict):
                continue
            idx = r.get("row_index")
            side = (r.get("side") or "").strip().lower()
            if isinstance(idx, int) and side in {"left", "right"}:
                side_map[(page_id, idx)] = side
    return side_map


def load_data(workbook_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(workbook_path)
    rows: list[dict[str, Any]] = []
    side_map = build_side_map(CACHE_DIR)

    for sheet in xl.sheet_names:
        parsed = parse_sheet_years(sheet)
        if parsed is None:
            continue
        years, page = parsed
        year_weight = 1.0 / len(years)

        df = pd.read_excel(workbook_path, sheet_name=sheet)
        req = ["Type", "Description", "£ (Pounds)", "s (Shillings)", "d (Pence)", "d Fraction"]
        if not set(req).issubset(df.columns):
            continue

        for i, r in df.iterrows():
            row_type = str(r["Type"]).strip().lower()
            if row_type not in {"header", "entry", "total"}:
                continue

            pounds    = parse_money_number(r["£ (Pounds)"])
            shillings = parse_money_number(r["s (Shillings)"])
            pence     = parse_money_number(r["d (Pence)"])
            frac      = parse_fraction(r["d Fraction"])
            amount_decimal = pounds + (shillings / 20.0) + ((pence + frac) / 240.0)
            row_idx = int(i) + 1
            side = side_map.get((sheet, row_idx), "")
            # Heuristic: if side exists, treat value as one side of a debit/credit pair.
            side_adjustment = 0.5 if side in {"left", "right"} else 1.0
            amount_decimal_adjusted = amount_decimal * side_adjustment

            desc_raw  = "" if pd.isna(r["Description"]) else str(r["Description"])
            desc_norm = normalize_text(desc_raw)

            for year in years:
                rows.append({
                    "sheet":           sheet,
                    "sheet_year_span": (
                        "-".join([str(years[0]), str(years[-1])])
                        if len(years) > 1 else str(year)
                    ),
                    "year":            year,
                    "page":            page,
                    "row_idx":         row_idx,
                    "row_type":        row_type,
                    "side":            side,
                    "side_adjustment": side_adjustment,
                    "description_raw": desc_raw,
                    "description_norm": desc_norm,
                    "amount_decimal":  amount_decimal,
                    "amount_decimal_adjusted": amount_decimal_adjusted,
                    "year_weight":     year_weight,
                    "amount_weighted": amount_decimal_adjusted * year_weight,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No rows loaded from workbook.")
    out["era"] = out["year"].map(era_of_year)
    return out


# ---------------------------------------------------------------------------
# 3. Yearly numeric summary — now includes relative variables
# ---------------------------------------------------------------------------

def build_yearly_numeric(df: pd.DataFrame) -> pd.DataFrame:
    data_rows = df[df["row_type"].isin(["entry", "total"])].copy()

    yearly = (
        data_rows.groupby("year", as_index=False)
        .agg(
            n_pages       = ("page",         "nunique"),
            n_data_rows   = ("year_weight",   "sum"),
            n_entry_rows  = ("year_weight",   lambda s: float(
                s[data_rows.loc[s.index, "row_type"] == "entry"].sum()
            )),
            n_total_rows  = ("year_weight",   lambda s: float(
                s[data_rows.loc[s.index, "row_type"] == "total"].sum()
            )),
            amount_sum    = ("amount_weighted", "sum"),
            amount_mean   = ("amount_decimal_adjusted",  "mean"),
            amount_median = ("amount_decimal_adjusted",  "median"),
            amount_p90    = ("amount_decimal_adjusted",  lambda s: float(np.quantile(s, 0.90))),
            amount_max    = ("amount_decimal_adjusted",  "max"),
        )
        .sort_values("year")
    )

    # Round row counts
    for col in ["n_data_rows", "n_entry_rows", "n_total_rows"]:
        yearly[col] = yearly[col].round(3)

    yearly["era"] = yearly["year"].map(era_of_year)

    # Side-aware diagnostic columns
    yearly_side = (
        data_rows.assign(has_side=data_rows["side"].isin(["left", "right"]).astype(float))
        .groupby("year", as_index=False)
        .agg(
            n_side_rows=("has_side", "sum"),
            side_row_ratio=("has_side", "mean"),
        )
    )
    yearly = yearly.merge(yearly_side, on="year", how="left")
    yearly["n_side_rows"] = yearly["n_side_rows"].fillna(0.0).round(3)
    yearly["side_row_ratio"] = yearly["side_row_ratio"].fillna(0.0).round(4)

    # --- Relative variables (Issue 3) ---
    # entry_ratio: fraction of data rows that are entry rows (not total rows).
    # Rising ratio → more granular sub-entries; stable → consistent structure.
    yearly["entry_ratio"] = (
        yearly["n_entry_rows"] / yearly["n_data_rows"].replace(0, np.nan)
    ).round(4)

    # data_rows_per_page: normalises volume by the number of pages that year.
    # Distinguishes "genuinely more activity" from "more pages digitised".
    yearly["data_rows_per_page"] = (
        yearly["n_data_rows"] / yearly["n_pages"].replace(0, np.nan)
    ).round(3)

    # --- Price deflation (Issue 1) ---
    yearly["price_index_1700base"] = yearly["year"].map(price_index).round(2)

    # amount_sum_real: nominal amount_sum expressed in constant 1700 pounds.
    # Formula: real = nominal / (price_index / 100)
    yearly["amount_sum_real"] = (
        yearly["amount_sum"] / (yearly["price_index_1700base"] / 100.0)
    ).round(4)

    # amount_median_real: deflated per-entry median.
    yearly["amount_median_real"] = (
        yearly["amount_median"] / (yearly["price_index_1700base"] / 100.0)
    ).round(6)

    return yearly


# ---------------------------------------------------------------------------
# Change-point detection — extended to relative variables
# ---------------------------------------------------------------------------

def detect_change_points(yearly: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "amount_sum",       # nominal
        "amount_sum_real",  # real (deflated)
        "n_data_rows",
        "amount_median",
        "entry_ratio",      # NEW: accounting structure shift
        "data_rows_per_page",  # NEW: normalised volume
    ]
    records: list[dict[str, Any]] = []

    for metric in metrics:
        if metric not in yearly.columns:
            continue
        diff = yearly[metric].diff().fillna(0.0)
        rz   = robust_zscore(diff.abs())
        flag = rz >= 2.5
        for i in range(len(yearly)):
            records.append({
                "year":               int(yearly.iloc[i]["year"]),
                "metric":             metric,
                "delta":              float(diff.iloc[i]),
                "abs_delta_robust_z": float(rz.iloc[i]),
                "is_change_point":    bool(flag.iloc[i]),
            })

    cp = pd.DataFrame(records)
    return cp.sort_values(["is_change_point", "abs_delta_robust_z"], ascending=[False, False])


# ---------------------------------------------------------------------------
# Text embeddings — with extended stop words (Issue 2)
# ---------------------------------------------------------------------------

def yearly_text_embeddings(df: pd.DataFrame):
    data_rows = df[df["row_type"].isin(["entry", "total"])].copy()
    data_rows = data_rows[data_rows["description_norm"] != ""]

    year_docs = (
        data_rows.groupby("year")["description_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index(name="doc")
        .sort_values("year")
    )

    tfidf = TfidfVectorizer(
        stop_words=STOP_WORDS,
        token_pattern=_TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=2,
        max_features=3500,
    )
    X = tfidf.fit_transform(year_docs["doc"])
    return year_docs, tfidf, X


def cluster_years(year_docs, X_text, yearly_numeric):
    n_years = X_text.shape[0]
    n_terms = X_text.shape[1]
    if n_years < 3:
        out = year_docs[["year"]].copy()
        out["cluster"] = 0
        out["era"] = out["year"].map(era_of_year)
        return out, pd.DataFrame()

    n_comp = max(2, min(50, n_years - 1, n_terms - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_text_emb = svd.fit_transform(X_text)

    num_cols = ["amount_sum_real", "n_data_rows", "amount_median_real", "entry_ratio"]
    available = [c for c in num_cols if c in yearly_numeric.columns]
    num = yearly_numeric.set_index("year").loc[year_docs["year"], available]
    X_num = StandardScaler().fit_transform(num.fillna(num.median()))
    X = np.hstack([X_text_emb, X_num])

    best_k = 2
    best_score = -1.0
    for k in range(2, min(8, n_years - 1) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    model = KMeans(n_clusters=best_k, random_state=42, n_init=50)
    labels = model.fit_predict(X)

    out = year_docs[["year"]].copy()
    out["cluster"] = labels
    out["era"] = out["year"].map(era_of_year)

    prof = (
        out.groupby(["cluster", "era"], as_index=False)
        .size()
        .rename(columns={"size": "n_years"})
        .sort_values(["cluster", "era"])
    )
    return out, prof


def top_terms_by_year(year_docs, tfidf, X_text, top_n: int = 15) -> pd.DataFrame:
    terms = np.array(tfidf.get_feature_names_out())
    records: list[dict[str, Any]] = []
    for i, y in enumerate(year_docs["year"].tolist()):
        row = X_text[i].toarray().ravel()
        idx = np.argsort(row)[-top_n:][::-1]
        for rank, j in enumerate(idx, start=1):
            if row[j] <= 0:
                continue
            records.append({"year": int(y), "rank": rank, "term": terms[j], "tfidf": float(row[j])})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Era term analysis — with extended stop words (Issue 2)
# ---------------------------------------------------------------------------

def era_term_analysis(df: pd.DataFrame):
    data_rows = df[df["row_type"].isin(["entry", "total"])].copy()
    data_rows = data_rows[data_rows["description_norm"] != ""]

    era_docs = (
        data_rows.groupby("era")["description_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .reindex(["pre_1760", "industrial_1760_1840", "post_1840"])
        .fillna("")
    )

    vec = CountVectorizer(
        stop_words=STOP_WORDS,
        token_pattern=_TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
    )
    X = vec.fit_transform(era_docs.tolist())
    terms = np.array(vec.get_feature_names_out())
    cnt   = pd.DataFrame(X.toarray(), index=era_docs.index, columns=terms)
    freq  = cnt.div(cnt.sum(axis=1).replace(0, 1), axis=0)

    top_records: list[dict[str, Any]] = []
    for era in freq.index:
        top = freq.loc[era].sort_values(ascending=False).head(30)
        for rank, (term, val) in enumerate(top.items(), start=1):
            top_records.append({"era": era, "rank": rank, "term": term, "relative_freq": float(val)})
    era_top = pd.DataFrame(top_records)

    comp = pd.DataFrame({"term": terms})
    comp["pre"]  = freq.loc["pre_1760",             terms].values
    comp["ind"]  = freq.loc["industrial_1760_1840",  terms].values
    comp["post"] = freq.loc["post_1840",              terms].values
    comp["delta_post_vs_pre"] = comp["post"] - comp["pre"]
    comp["delta_ind_vs_pre"]  = comp["ind"]  - comp["pre"]

    emerging = pd.concat([
        comp.sort_values("delta_ind_vs_pre",  ascending=False).head(80).assign(change="industrial_vs_pre"),
        comp.sort_values("delta_post_vs_pre", ascending=False).head(80).assign(change="post_vs_pre"),
    ], ignore_index=True)

    era_term_long = (
        freq.reset_index()
        .rename(columns={"index": "era"})
        .melt(id_vars="era", var_name="term", value_name="relative_freq")
    )
    return era_top, emerging, era_term_long


# ---------------------------------------------------------------------------
# Header account analysis — with extended stop words (Issue 2)
# ---------------------------------------------------------------------------

def header_account_analysis(df: pd.DataFrame):
    headers = df[df["row_type"] == "header"].copy()
    headers = headers[headers["description_norm"] != ""]

    first_app = (
        headers.groupby("description_norm", as_index=False)
        .agg(
            n_occurrences = ("description_norm", "count"),
            n_years       = ("year", "nunique"),
            first_year    = ("year", "min"),
            last_year     = ("year", "max"),
            sample_text   = ("description_raw", "first"),
        )
        .sort_values(["n_years", "n_occurrences"], ascending=[False, False])
    )
    first_app["first_era"] = first_app["first_year"].map(era_of_year)

    yearly_counts = (
        headers.groupby("year", as_index=False)
        .agg(
            n_header_rows        = ("row_type",         "count"),
            n_unique_header_texts= ("description_norm", "nunique"),
        )
        .sort_values("year")
    )

    # header_diversity: ratio of unique texts to total header rows.
    # High ratio → each year used distinct headers; low → repetitive labels.
    yearly_counts["header_diversity"] = (
        yearly_counts["n_unique_header_texts"]
        / yearly_counts["n_header_rows"].replace(0, np.nan)
    ).round(4)

    era_docs = (
        headers.groupby("era")["description_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .reindex(["pre_1760", "industrial_1760_1840", "post_1840"])
        .fillna("")
    )
    vec = CountVectorizer(
        stop_words=STOP_WORDS,
        token_pattern=_TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=2,
        max_features=3000,
    )
    X = vec.fit_transform(era_docs.tolist())
    terms = np.array(vec.get_feature_names_out())
    mat   = pd.DataFrame(X.toarray(), index=era_docs.index, columns=terms)
    freq  = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)

    era_top_records: list[dict[str, Any]] = []
    for era in freq.index:
        top = freq.loc[era].sort_values(ascending=False).head(40)
        for rank, (term, val) in enumerate(top.items(), start=1):
            era_top_records.append({"era": era, "rank": rank, "term": term, "relative_freq": float(val)})

    return first_app, yearly_counts, pd.DataFrame(era_top_records)


def recurring_accounts(df: pd.DataFrame) -> pd.DataFrame:
    data_rows = df[df["row_type"].isin(["entry", "total"])].copy()
    data_rows = data_rows[data_rows["description_norm"] != ""]

    g = (
        data_rows.groupby("description_norm", as_index=False)
        .agg(
            n_occurrences = ("description_norm", "count"),
            first_year    = ("year", "min"),
            last_year     = ("year", "max"),
            sample_text   = ("description_raw", "first"),
        )
        .sort_values("n_occurrences", ascending=False)
    )
    g["first_era"] = g["first_year"].map(era_of_year)
    return g


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    yearly: pd.DataFrame,
    cp: pd.DataFrame,
    year_clusters: pd.DataFrame,
    era_top: pd.DataFrame,
    header_yearly: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Nominal amount_sum + row volume
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(yearly["year"], yearly["amount_sum"], color="#1f77b4", marker="o", linewidth=2, label="amount_sum (nominal)")
    ax1.set_ylabel("Total amount — nominal £", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["n_data_rows"], color="#ff7f0e", marker=".", alpha=0.7, label="n_data_rows")
    ax2.set_ylabel("Number of data rows", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax1.set_title("Year-level total amount (nominal) and row volume")
    ax1.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yearly_amount_and_volume.png", dpi=180)
    plt.close(fig)

    # 2. Nominal vs real amount_sum comparison (NEW for v2)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(yearly["year"], yearly["amount_sum"],      label="Nominal £", color="#1f77b4", linewidth=2)
    ax.plot(yearly["year"], yearly["amount_sum_real"], label="Real £ (1700 prices)", color="#d62728", linewidth=2, linestyle="--")
    ax.axvspan(1760, 1840, color="grey", alpha=0.15, label="Industrial era")
    ax.set_title("Nominal vs. real amount_sum (Phelps Brown-Hopkins deflation, base 1700)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total amount (£)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nominal_vs_real_amount.png", dpi=180)
    plt.close(fig)

    # 3. Median and p90
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(yearly["year"], yearly["amount_median"],      marker="o", label="median (nominal)")
    ax.plot(yearly["year"], yearly["amount_median_real"], marker="o", label="median (real)", linestyle="--")
    ax.plot(yearly["year"], yearly["amount_p90"],         marker="o", label="p90 (nominal)", alpha=0.6)
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title("Year-level amount distribution (nominal and real)")
    ax.set_xlabel("Year")
    ax.set_ylabel("£")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yearly_median_p90.png", dpi=180)
    plt.close(fig)

    # 4. Change points on nominal amount_sum
    cp_amount = cp[(cp["metric"] == "amount_sum") & (cp["is_change_point"])].copy()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(yearly["year"], yearly["amount_sum"], marker="o", color="#1f77b4")
    for _, r in cp_amount.iterrows():
        y = int(r["year"])
        mask = yearly["year"] == y
        if mask.any():
            val = float(yearly.loc[mask, "amount_sum"].iloc[0])
            ax.axvline(y, color="red", linestyle="--", alpha=0.45)
            ax.scatter([y], [val], color="red", zorder=3)
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title("Detected change points on yearly total amount (nominal)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total amount (nominal £)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "change_points_amount_sum.png", dpi=180)
    plt.close(fig)

    # 5. entry_ratio over time (NEW for v2)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(yearly["year"], yearly["entry_ratio"], color="#2ca02c", marker=".", linewidth=1.5)
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title("Entry ratio over time (n_entry_rows / n_data_rows)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Entry ratio")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "entry_ratio_over_time.png", dpi=180)
    plt.close(fig)

    # 6. data_rows_per_page over time (NEW for v2)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(yearly["year"], yearly["data_rows_per_page"], color="#9467bd", marker=".", linewidth=1.5)
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title("Data rows per page over time (normalised row density)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rows / page")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "data_rows_per_page.png", dpi=180)
    plt.close(fig)

    # 7. header_diversity over time (NEW for v2)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(header_yearly["year"], header_yearly["header_diversity"], color="#8c564b", marker=".", linewidth=1.5)
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title("Header diversity over time (unique header texts / header rows)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Header diversity ratio")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "header_diversity_over_time.png", dpi=180)
    plt.close(fig)

    # 8. Year cluster timeline
    if not year_clusters.empty:
        fig, ax = plt.subplots(figsize=(13, 3.5))
        sns.scatterplot(data=year_clusters, x="year", y="cluster", hue="cluster", palette="tab10", ax=ax, s=55)
        ax.axvspan(1760, 1840, color="grey", alpha=0.15)
        ax.set_title("Year-level embedding clusters (v2: real amounts + clean text)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cluster")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "year_embedding_clusters_timeline.png", dpi=180)
        plt.close(fig)

    # 9. Era top terms heatmap
    heat = era_top[era_top["rank"] <= 15].copy()
    if not heat.empty:
        p = heat.pivot_table(index="term", columns="era", values="relative_freq", aggfunc="max").fillna(0.0)
        p = p.loc[p.max(axis=1).sort_values(ascending=False).head(25).index]
        fig, ax = plt.subplots(figsize=(9, 10))
        sns.heatmap(p, cmap="YlGnBu", ax=ax)
        ax.set_title("Top terms by era — after extended stop-word filtering (v2)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "era_top_terms_heatmap.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df = load_data(WORKBOOK)
    df.to_csv(OUT_DIR / "all_rows_with_amounts.csv", index=False)

    print("Building yearly numeric summary (with side-aware adjustment + relative variables + deflation)…")
    yearly = build_yearly_numeric(df)
    yearly.to_csv(OUT_DIR / "yearly_numeric_summary_v2.csv", index=False)

    print("Detecting change points…")
    cp = detect_change_points(yearly)
    cp.to_csv(OUT_DIR / "yearly_change_points_v2.csv", index=False)

    print("Building text embeddings (extended stop words)…")
    year_docs, tfidf, X_text = yearly_text_embeddings(df)
    y_terms = top_terms_by_year(year_docs, tfidf, X_text, top_n=15)
    y_terms.to_csv(OUT_DIR / "year_top_terms_tfidf_v2.csv", index=False)

    print("Clustering years…")
    year_clusters, cluster_profile = cluster_years(year_docs, X_text, yearly)
    year_clusters.to_csv(OUT_DIR / "year_embedding_clusters_v2.csv", index=False)
    cluster_profile.to_csv(OUT_DIR / "year_cluster_profile_by_era_v2.csv", index=False)

    print("Era term analysis…")
    era_top, emerging, era_term_long = era_term_analysis(df)
    era_top.to_csv(OUT_DIR / "era_top_terms_v2.csv", index=False)
    emerging.to_csv(OUT_DIR / "era_emerging_terms_v2.csv", index=False)
    era_term_long.to_csv(OUT_DIR / "era_term_frequencies_long_v2.csv", index=False)

    print("Header account analysis…")
    header_first_app, header_yearly_counts, header_era_top = header_account_analysis(df)
    header_first_app.to_csv(OUT_DIR / "header_account_first_appearance_v2.csv", index=False)
    header_yearly_counts.to_csv(OUT_DIR / "header_account_yearly_counts_v2.csv", index=False)
    header_era_top.to_csv(OUT_DIR / "header_account_era_top_terms_v2.csv", index=False)

    print("Recurring accounts…")
    accounts = recurring_accounts(df)
    accounts.to_csv(OUT_DIR / "recurring_descriptions_first_appearance_v2.csv", index=False)

    print("Generating plots…")
    make_plots(yearly, cp, year_clusters, era_top, header_yearly_counts)

    # ------------------------------------------------------------------
    # Build summary report
    # ------------------------------------------------------------------
    n_years = yearly["year"].nunique()

    # Era stats: both nominal and real
    era_stats = (
        yearly.groupby("era", as_index=False)
        .agg(
            years                   = ("year",              "count"),
            mean_amount_sum_nominal = ("amount_sum",         "mean"),
            mean_amount_sum_real    = ("amount_sum_real",    "mean"),
            median_amount_median_nominal = ("amount_median", "median"),
            median_amount_median_real    = ("amount_median_real", "median"),
            mean_entry_ratio        = ("entry_ratio",        "mean"),
            mean_data_rows_per_page = ("data_rows_per_page", "mean"),
        )
        .sort_values("era")
    )
    era_table = era_stats.to_string(index=False)

    # Nominal vs real change: sanity-check sentence
    pre_nom  = era_stats.loc[era_stats["era"] == "pre_1760",  "mean_amount_sum_nominal"].values[0]
    post_nom = era_stats.loc[era_stats["era"] == "post_1840", "mean_amount_sum_nominal"].values[0]
    pre_real = era_stats.loc[era_stats["era"] == "pre_1760",  "mean_amount_sum_real"].values[0]
    post_real= era_stats.loc[era_stats["era"] == "post_1840", "mean_amount_sum_real"].values[0]
    nom_ratio  = post_nom / pre_nom  if pre_nom  > 0 else float("nan")
    real_ratio = post_real / pre_real if pre_real > 0 else float("nan")

    # Change points on nominal amount_sum
    cp_amount = cp[(cp["metric"] == "amount_sum") & (cp["is_change_point"])].head(10)
    cp_text = "\n".join([
        f"- {int(r.year)}: delta={r.delta:,.2f}, robust_z={r.abs_delta_robust_z:.2f}"
        for r in cp_amount.itertuples(index=False)
    ]) or "- No strong change point detected"

    # Change points on entry_ratio
    cp_er = cp[(cp["metric"] == "entry_ratio") & (cp["is_change_point"])].head(5)
    cp_er_text = "\n".join([
        f"- {int(r.year)}: delta={r.delta:.4f}, robust_z={r.abs_delta_robust_z:.2f}"
        for r in cp_er.itertuples(index=False)
    ]) or "- No strong change point on entry_ratio"

    summary = f"""# Reanalysis Summary v2 (Year-level)

## What is new in v2

### Issue 1 — Price deflation
Nominal pound amounts have been deflated using the **Phelps Brown-Hopkins price
index** (base: 1700 = 100) to produce constant-price (real) equivalents.
This allows testing whether the observed growth in transaction scale persists
after removing inflation.

- Post-1840 vs pre-1760 nominal ratio : {nom_ratio:.1f}×
- Post-1840 vs pre-1760 real ratio    : {real_ratio:.1f}×

A real ratio substantially smaller than the nominal ratio indicates that part
of the apparent expansion was driven by rising prices (especially the
post-1800 Napoleonic-era spike to ~270 on the index), not real activity growth.

### Issue 2 — Extended stop-word list
The following word categories are now excluded from all TF-IDF and
CountVectorizer analyses:
- **Time template**: year, years, yr, yrs
- **Accounting format**: total, balance, carried, forward, brought, account,
  acct, received, paid, payment, payments, sum, amounts
- **Currency units**: pound, pounds, shilling, shillings, pence, penny, sterling
- **Generic abbreviations**: ob, viz, ie, per

A token pattern `[a-z][a-z]+` also eliminates pure-digit tokens (e.g., "10").

### Issue 3 — Relative variables
Three new relative variables are added to the yearly summary:
- **entry_ratio** = n_entry_rows / n_data_rows
  Separates real transaction activity from additional subtotal rows.
- **data_rows_per_page** = n_data_rows / n_pages
  Normalises row volume by the number of pages covered each year.
- **header_diversity** = n_unique_header_texts / n_header_rows
  Tests whether header growth reflects new account categories or repetition.

Both entry_ratio and data_rows_per_page are included in change-point detection.

---

## Dataset overview
- Years covered: {n_years}
- Data rows (entry+total): {int((df['row_type'].isin(['entry','total'])).sum()):,}
- Header rows: {int((df['row_type'] == 'header').sum()):,}
- Stop words added beyond sklearn English: {len(_LEDGER_EXTRA_STOPS)}

## Era-level summary (nominal and real)
{era_table}

## Change points — yearly amount_sum (nominal)
{cp_text}

## Change points — entry_ratio (accounting structure)
{cp_er_text}

## Key outputs
- yearly_numeric_summary_v2.csv   (includes entry_ratio, data_rows_per_page,
                                    price_index_1700base, amount_sum_real,
                                    amount_median_real)
- yearly_change_points_v2.csv     (now includes entry_ratio and
                                    data_rows_per_page metrics)
- year_top_terms_tfidf_v2.csv
- year_embedding_clusters_v2.csv
- era_top_terms_v2.csv            (stop-word-filtered)
- era_emerging_terms_v2.csv
- header_account_yearly_counts_v2.csv  (includes header_diversity)

## Key figures
- nominal_vs_real_amount.png      (NEW: deflation comparison)
- yearly_amount_and_volume.png
- yearly_median_p90.png           (now includes real median)
- change_points_amount_sum.png
- entry_ratio_over_time.png       (NEW)
- data_rows_per_page.png          (NEW)
- header_diversity_over_time.png  (NEW)
- year_embedding_clusters_timeline.png
- era_top_terms_heatmap.png       (clean terms, no noise words)
"""

    (OUT_DIR / "REANALYSIS_SUMMARY_v2.md").write_text(summary, encoding="utf-8")
    print(f"Done. Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
