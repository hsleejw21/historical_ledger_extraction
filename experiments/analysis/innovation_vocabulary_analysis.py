"""innovation_vocabulary_analysis.py

Novelty-phase analysis: curriculum reform timing, scholarship explosion,
and organizational complexity growth in Oxford's 1700-1900 accounts.

Novel contributions:
  B1 – Innovation vocabulary emergence: does science spending precede the 1854 Act?
  B2 – Scholarship/prize trajectory: income-to-expenditure directional reversal
  B3 – Section header diversity as organizational complexity signal

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
import numpy as np
import pandas as pd
import seaborn as sns

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
# Price deflation (copied from analysis_enriched.py)
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
# Era definitions
# ---------------------------------------------------------------------------

ERAS = [
    ("pre_industrial",    1700, 1779),
    ("transition",        1780, 1819),
    ("early_industrial",  1820, 1859),
    ("late_industrial",   1860, 1900),
]
ERA_ORDER = ["pre_industrial", "transition", "early_industrial", "late_industrial"]
ERA_LABELS = {
    "pre_industrial":   "Pre-Industrial\n(1700–1779)",
    "transition":       "Transition\n(1780–1819)",
    "early_industrial": "Early Industrial\n(1820–1859)",
    "late_industrial":  "Late Industrial\n(1860–1900)",
}
ERA_VLINES = [1780, 1820, 1860]
REFORM_VLINES = {1854: "Oxford Act 1854", 1877: "Oxford & Cambridge Act 1877"}


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
    mapping = {"¼": 0.25, "1/4": 0.25, ".25": 0.25,
               "½": 0.5,  "1/2": 0.5,  ".5":  0.5,
               "¾": 0.75, "3/4": 0.75, ".75": 0.75}
    return mapping.get(s, 0.0) if s in mapping else (float(s) if re.match(r"^[\d.]+$", s) else 0.0)


def parse_money(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return 0.0 if np.isnan(v) else v
    s = re.sub(r"[^0-9.\-]", "", str(value).strip())
    return float(s) if s else 0.0


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
# Data loading (entry rows)
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
                    "page_id":         page_id,
                    "year":            year,
                    "year_weight":     year_weight,
                    "amount":          amt,
                    "amount_weighted": amt * year_weight,
                    "direction":       r.get("direction"),
                    "category":        r.get("category"),
                    "language":        r.get("language"),
                    "payment_period":  r.get("payment_period"),
                    "is_arrears":      r.get("is_arrears"),
                    "person_name":     r.get("person_name"),
                    "place_name":      r.get("place_name"),
                    "description":     r.get("description", ""),
                    "english_desc":    r.get("english_description", ""),
                    "section_header":  r.get("section_header", ""),
                    "notes":           r.get("notes", ""),
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No entry rows loaded.")
    df["era"] = df["year"].map(era_of_year)
    df["price_idx"] = df["year"].map(price_index)
    df["amount_real"] = df["amount"] / (df["price_idx"] / 100.0)
    df["amount_real_w"] = df["amount_weighted"] / (df["price_idx"] / 100.0)
    print(f"  Loaded {len(df):,} records across {df['year'].nunique()} years")
    return df


def add_era_vlines(ax: plt.Axes, alpha: float = 0.5) -> None:
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=alpha)


def add_reform_vlines(ax: plt.Axes) -> None:
    for yr, lbl in REFORM_VLINES.items():
        ax.axvline(yr, color="darkblue", lw=1.2, ls="-.", alpha=0.8)
        ax.text(yr + 0.5, ax.get_ylim()[1] * 0.95, lbl, fontsize=6,
                color="darkblue", rotation=90, va="top")


# ---------------------------------------------------------------------------
# Innovation vocabulary
# ---------------------------------------------------------------------------

INNOVATION_VOCAB: dict[str, list[str]] = {
    "scientific_professorships": [
        "professor", "natural philosophy", "chemistry",
        "geology", "mineralogy", "anatomy", "botany", "mathematics", "astronomy",
    ],
    "competitive_examinations": [
        "examination", "moderations", "finals", "honour",
        "honours", "class list", "literae humaniores", "greats",
    ],
    "scholarships_prizes": [
        "scholarship", "exhibitioner", "exhibition",
        "bursary", "prize", "demy", "postmaster", "open scholarship",
    ],
    "lectures_teaching": [
        "lecture", "lecturer", "tutor", "tutorial", "teaching", "reader",
    ],
    "scientific_infrastructure": [
        "laboratory", "apparatus", "instrument", "scientific",
        "experiment", "specimen", "museum",
    ],
    "reform_modernization": [
        "commission", "reform", "statute", "ordinance",
        "university act", "endowment",
    ],
}

GROUP_COLORS = {
    "scientific_professorships": "#e41a1c",
    "competitive_examinations":  "#377eb8",
    "scholarships_prizes":       "#4daf4a",
    "lectures_teaching":         "#ff7f00",
    "scientific_infrastructure": "#984ea3",
    "reform_modernization":      "#a65628",
}


# ---------------------------------------------------------------------------
# B1: Innovation Vocabulary Emergence
# ---------------------------------------------------------------------------

def analyse_innovation_vocabulary(df: pd.DataFrame) -> pd.DataFrame:
    print("[1/6] Analysing innovation vocabulary …")
    text_df = df[df["english_desc"].notna()].copy()
    text_df["text_lower"] = text_df["english_desc"].str.lower().fillna("")

    # Per-year total entries
    yearly_total = (text_df.groupby("year")["year_weight"].sum().rename("total_entries"))

    rows = []
    kw_rows = []

    for group, keywords in INNOVATION_VOCAB.items():
        group_hit_yearly: dict[int, float] = {}
        for kw in keywords:
            hit = text_df["text_lower"].str.contains(re.escape(kw), na=False)
            kw_yearly = (text_df[hit].groupby("year")["year_weight"].sum())
            kw_yearly = kw_yearly.reindex(yearly_total.index, fill_value=0.0)
            kw_share = kw_yearly / yearly_total.replace(0, np.nan)

            # First/peak year for keyword
            first_yr = int(text_df[hit]["year"].min()) if hit.any() else None
            peak_yr  = int(kw_yearly.idxmax()) if kw_yearly.sum() > 0 else None

            for era_name, y_start, y_end in ERAS:
                era_count = kw_yearly[(kw_yearly.index >= y_start) & (kw_yearly.index <= y_end)].sum()
                kw_rows.append({
                    "keyword": kw, "group": group,
                    "first_year": first_yr, "peak_year": peak_yr,
                    "peak_count": float(kw_yearly.max()),
                    f"{era_name}_count": float(era_count),
                })

            # Accumulate group score per year
            for yr, sh in kw_share.items():
                group_hit_yearly[yr] = group_hit_yearly.get(yr, 0.0) + (sh if not np.isnan(sh) else 0.0)

        # Build group series
        group_series = pd.Series(group_hit_yearly).reindex(yearly_total.index, fill_value=0.0)
        group_5yr = group_series.rolling(5, center=True, min_periods=3).mean()

        # Emergence threshold: pre-1780 mean + 2σ
        pre1780 = group_5yr[(group_5yr.index >= 1700) & (group_5yr.index < 1780)].dropna()
        if len(pre1780) >= 5:
            threshold = pre1780.mean() + 2 * pre1780.std()
            above = group_5yr[group_5yr > threshold]
            emerge_yr = int(above.index.min()) if not above.empty else None
        else:
            threshold, emerge_yr = None, None

        for yr in yearly_total.index:
            rows.append({
                "year": yr,
                "group": group,
                "group_score": group_hit_yearly.get(yr, 0.0),
                "group_score_5yr": float(group_5yr.get(yr, np.nan)) if yr in group_5yr.index else np.nan,
                "emergence_year": emerge_yr,
                "threshold": threshold,
            })

    long_df = pd.DataFrame(rows)

    # Pivot to wide
    wide = long_df.pivot_table(index="year", columns="group",
                               values="group_score", fill_value=0.0).reset_index()
    wide["era"] = wide["year"].map(era_of_year)
    wide.to_csv(OUT_DIR / "innovation_vocabulary_yearly.csv", index=False)

    # Keyword details
    kw_df = pd.DataFrame(kw_rows)
    kw_df = kw_df.groupby(["keyword", "group", "first_year", "peak_year", "peak_count"]).sum().reset_index()
    kw_df.to_csv(OUT_DIR / "innovation_keyword_details.csv", index=False)

    # Emergence summary
    emerge_df = (long_df.groupby("group")[["emergence_year", "threshold"]]
                 .first().reset_index())
    emerge_df.to_csv(OUT_DIR / "innovation_emergence_years.csv", index=False)

    return long_df


def plot_innovation_vocabulary(long_df: pd.DataFrame) -> None:
    print("[2/6] Plotting innovation vocabulary …")
    emerge_df = long_df.groupby("group")[["emergence_year", "threshold"]].first().reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))

    # Panel 1: Group score lines (5-yr rolling)
    ax = axes[0]
    for group in INNOVATION_VOCAB.keys():
        sub = long_df[long_df["group"] == group].sort_values("year")
        ax.plot(sub["year"], sub["group_score_5yr"], label=group,
                color=GROUP_COLORS[group], lw=2)
    add_era_vlines(ax)
    for yr, lbl in REFORM_VLINES.items():
        ax.axvline(yr, color="darkblue", lw=1.2, ls="-.", alpha=0.8)
    ax.set_ylabel("Group vocabulary score (keyword entries / total), 5-yr rolling")
    ax.set_title("Innovation Vocabulary Emergence in Oxford Ledgers 1700–1900\n"
                 "(Blue vertical lines: Reform Acts 1854 & 1877)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(1700, 1900)
    # Add reform labels
    ymax = ax.get_ylim()[1]
    for yr, lbl in REFORM_VLINES.items():
        ax.text(yr + 0.5, ymax * 0.9, lbl, fontsize=6, color="darkblue", rotation=90, va="top")

    # Panel 2: Era heatmap
    ax = axes[1]
    wide = long_df.pivot_table(index="year", columns="group",
                               values="group_score", fill_value=0.0).reset_index()
    wide["era"] = wide["year"].map(era_of_year)
    groups = list(INNOVATION_VOCAB.keys())
    era_scores = {}
    for era_name in ERA_ORDER:
        sub = wide[wide["era"] == era_name]
        era_scores[era_name] = {g: sub[g].mean() if g in sub.columns else 0.0 for g in groups}
    heat_df = pd.DataFrame(era_scores, index=groups)[ERA_ORDER]
    # Normalise each row to [0,1] for visual clarity
    heat_norm = heat_df.div(heat_df.max(axis=1).replace(0, 1), axis=0)
    sns.heatmap(heat_norm, ax=ax, annot=heat_df.round(4), fmt=".4f",
                cmap="YlOrBr", linewidths=0.5,
                cbar_kws={"label": "Normalised score (row max = 1)"})
    ax.set_title("Innovation Vocabulary: Era-level Heatmap\n"
                 "(Values = raw mean score; colour = row-normalised)", fontweight="bold")
    ax.set_xticklabels([ERA_LABELS[e].replace("\n", " ") for e in ERA_ORDER], fontsize=7)
    ax.set_ylabel("")

    # Panel 3: Emergence year bar chart
    ax = axes[2]
    emerge_valid = emerge_df.dropna(subset=["emergence_year"]).sort_values("emergence_year")
    if not emerge_valid.empty:
        colors_bar = [GROUP_COLORS[g] for g in emerge_valid["group"]]
        bars = ax.barh(emerge_valid["group"], emerge_valid["emergence_year"],
                       color=colors_bar, alpha=0.85)
        ax.set_xlabel("Year of emergence (first year 5-yr score > pre-1780 mean + 2σ)")
        ax.set_title("When Did Each Innovation Vocabulary Group Emerge?", fontweight="bold")
        # Add reform reference lines
        for yr, lbl in REFORM_VLINES.items():
            ax.axvline(yr, color="darkblue", lw=1.2, ls="-.", alpha=0.8)
            ax.text(yr, len(emerge_valid) - 0.5, lbl, fontsize=6, color="darkblue",
                    rotation=0, va="bottom", ha="right")
        # Annotate "before/after 1854"
        for _, row in emerge_valid.iterrows():
            note = "pre-legislation" if row["emergence_year"] < 1854 else "post-1854"
            ax.text(row["emergence_year"] + 1, emerge_valid["group"].tolist().index(row["group"]),
                    note, va="center", fontsize=6)
    else:
        ax.text(0.5, 0.5, "No emergence years detected", transform=ax.transAxes, ha="center")
    ax.set_xlim(1700, 1910)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_B1_innovation_vocabulary.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# B2: Scholarship and Prize Trajectory
# ---------------------------------------------------------------------------

SCHOLARSHIP_KWS = r"scholarship|scholar|exhibitioner|exhibition|bursary|demy|postmaster|open scholarship"
PRIZE_KWS       = r"\bprize\b|award"


def analyse_scholarship_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Analysing scholarship trajectory …")
    educ = df[df["category"] == "educational"].copy()
    educ["text_lower"] = educ["english_desc"].fillna("").str.lower()

    educ["is_scholar"] = educ["text_lower"].str.contains(SCHOLARSHIP_KWS, na=False)
    educ["is_prize"]   = educ["text_lower"].str.contains(PRIZE_KWS, na=False)

    yearly = educ.groupby("year").apply(lambda g: pd.Series({
        "n_educational":          g["year_weight"].sum(),
        "n_scholarship":          (g["is_scholar"] * g["year_weight"]).sum(),
        "n_prize":                (g["is_prize"] * g["year_weight"]).sum(),
        "scholar_income":         (g[g["is_scholar"] & (g["direction"] == "income")]["year_weight"]).sum(),
        "scholar_expenditure":    (g[g["is_scholar"] & (g["direction"] == "expenditure")]["year_weight"]).sum(),
    })).reset_index()

    yearly["scholarship_intensity"] = yearly["n_scholarship"] / yearly["n_educational"].replace(0, np.nan)
    yearly["prize_intensity"]       = yearly["n_prize"] / yearly["n_educational"].replace(0, np.nan)
    yearly["scholar_income_share"]  = yearly["scholar_income"] / (yearly["scholar_income"] + yearly["scholar_expenditure"]).replace(0, np.nan)
    yearly["era"] = yearly["year"].map(era_of_year)

    yearly.to_csv(OUT_DIR / "scholarship_prize_trajectory.csv", index=False)
    return yearly


def plot_scholarship_trajectory(yearly: pd.DataFrame) -> None:
    print("[4/6] Plotting scholarship trajectory …")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # Panel 1: Intensity + absolute count (dual axis)
    ax = axes[0]
    ax2 = ax.twinx()
    roll_int = yearly.set_index("year")["scholarship_intensity"].rolling(5, center=True, min_periods=3).mean()
    roll_prize = yearly.set_index("year")["prize_intensity"].rolling(5, center=True, min_periods=3).mean()
    ax.plot(roll_int.index, roll_int.values, color="#4daf4a", lw=2, label="Scholarship intensity (5-yr)")
    ax.plot(roll_prize.index, roll_prize.values, color="#ff7f00", lw=2, ls="--", label="Prize intensity (5-yr)")
    ax2.bar(yearly["year"], yearly["n_scholarship"], color="grey", alpha=0.3, width=0.8, label="Abs. scholarship count")
    ax.set_ylabel("Share of educational entries")
    ax2.set_ylabel("Absolute count (weighted)", color="grey")
    ax.set_title("Scholarship and Prize Intensity in Oxford Educational Accounts", fontweight="bold")
    add_era_vlines(ax)
    for yr, lbl in REFORM_VLINES.items():
        ax.axvline(yr, color="darkblue", lw=1.2, ls="-.", alpha=0.8)
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    ax.set_xlim(1700, 1900)

    # Panel 2: Direction split by era (stacked bar)
    ax = axes[1]
    era_dir = yearly.groupby("era")[["scholar_income", "scholar_expenditure"]].sum()
    era_dir = era_dir.reindex(ERA_ORDER).fillna(0)
    x = np.arange(len(ERA_ORDER))
    total = era_dir["scholar_income"] + era_dir["scholar_expenditure"]
    ax.bar(x, era_dir["scholar_income"] / total.replace(0, np.nan),
           label="Income (fees from scholars)", color="#1f77b4", alpha=0.85)
    ax.bar(x, era_dir["scholar_expenditure"] / total.replace(0, np.nan),
           bottom=era_dir["scholar_income"] / total.replace(0, np.nan),
           label="Expenditure (awards to scholars)", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ERA_LABELS[e].replace("\n", " ") for e in ERA_ORDER], fontsize=8)
    ax.set_ylabel("Share of scholarship entries")
    ax.set_title("Scholarship Direction: Income (Collecting Fees) vs Expenditure (Paying Awards)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_B2_scholarship_trajectory.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# B3: Section Header Diversity (load raw JSON, not entry rows)
# ---------------------------------------------------------------------------

def load_header_data() -> pd.DataFrame:
    """Load all header rows from enriched JSONs (not filtered by row_type=entry)."""
    records: list[dict] = []
    files = sorted(ENRICHED_DIR.glob("*_image_enriched.json"))
    print(f"[load headers] Scanning {len(files)} files …")
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        page_id = payload.get("page_id") or fp.name.replace("_enriched.json", "")
        try:
            years, _ = parse_page_id(page_id)
        except ValueError:
            continue
        year_weight = 1.0 / len(years)
        for r in payload.get("rows", []):
            if not isinstance(r, dict):
                continue
            if str(r.get("row_type", "")).strip().lower() != "header":
                continue
            desc = str(r.get("description", "")).strip()
            if not desc:
                continue
            # Normalise: lowercase, strip year-like tokens
            norm = re.sub(r"\b\d{4}[\.:]?\b", "", desc.lower()).strip()
            norm = re.sub(r"\s+", " ", norm).strip()
            if not norm or re.match(r"^[\d\s.,:;]+$", norm):
                continue
            for year in years:
                records.append({
                    "year": year,
                    "year_weight": year_weight,
                    "header_raw": desc,
                    "header_norm": norm,
                })
    return pd.DataFrame(records)


def analyse_header_diversity(df_entries: pd.DataFrame) -> pd.DataFrame:
    print("[5/6] Analysing section header diversity …")
    hdr = load_header_data()
    if hdr.empty:
        print("  No header rows found.")
        return pd.DataFrame()

    hdr["era"] = hdr["year"].map(era_of_year)

    # Pages per year (from entries df)
    pages_per_year = df_entries.groupby("year")["year_weight"].sum().rename("pages_proxy")

    # Track cumulative set of unique headers
    years_sorted = sorted(hdr["year"].unique())
    seen_headers: set[str] = set()
    yearly_rows = []

    for yr in years_sorted:
        sub = hdr[hdr["year"] == yr]
        unique_this_yr = set(sub["header_norm"].unique())
        new_this_yr = unique_this_yr - seen_headers
        seen_headers |= unique_this_yr

        # Check innovation vocab match
        all_text = " ".join(unique_this_yr)
        n_modern = sum(
            1 for kws in INNOVATION_VOCAB.values()
            for kw in kws
            if kw.lower() in all_text
        )

        yearly_rows.append({
            "year": yr,
            "n_unique_headers": len(unique_this_yr),
            "n_new_headers": len(new_this_yr),
            "n_modern_headers": n_modern,
            "cumulative_vocab": len(seen_headers),
        })

    result = pd.DataFrame(yearly_rows)
    result = result.merge(pages_per_year, on="year", how="left")
    result["header_diversity"] = result["n_unique_headers"] / result["pages_proxy"].replace(0, np.nan)
    result["era"] = result["year"].map(era_of_year)

    result.to_csv(OUT_DIR / "header_diversity_yearly.csv", index=False)
    return result


def plot_header_diversity(result: pd.DataFrame) -> None:
    print("[6/6] Plotting header diversity …")
    if result.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # Panel 1: Unique headers + diversity (dual axis)
    ax = axes[0]
    ax2 = ax.twinx()
    roll_unique = result.set_index("year")["n_unique_headers"].rolling(5, center=True, min_periods=3).mean()
    roll_div    = result.set_index("year")["header_diversity"].rolling(5, center=True, min_periods=3).mean()
    ax.plot(roll_unique.index, roll_unique.values, color="navy", lw=2, label="Unique headers (5-yr rolling)")
    ax2.plot(roll_div.index, roll_div.values, color="crimson", lw=1.5, ls="--",
             label="Header diversity (per page, 5-yr)")
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=0.5)
    for yr, lbl in REFORM_VLINES.items():
        ax.axvline(yr, color="darkblue", lw=1.2, ls="-.", alpha=0.8)
    ax.set_ylabel("Unique account section headers")
    ax2.set_ylabel("Header diversity (unique / pages proxy)", color="crimson")
    ax.set_title("Organizational Complexity: Growth of Account Section Headers 1700–1900",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="center left", fontsize=7)
    ax.set_xlim(1700, 1900)

    # Panel 2: New headers per year (pulse chart)
    ax = axes[1]
    colors_pulse = ["#1f77b4" if yr >= 1860 else "#aaaaaa" for yr in result["year"]]
    ax.bar(result["year"], result["n_new_headers"], color=colors_pulse, width=0.9, alpha=0.8)
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=0.5)
    for yr, lbl in REFORM_VLINES.items():
        ax.axvline(yr, color="darkblue", lw=1.2, ls="-.", alpha=0.8)
    ax.set_ylabel("New (first-ever) account headers")
    ax.set_xlabel("Year")
    ax.set_title("New Account Headers Per Year: Innovation Pulses (blue = post-1860)",
                 fontweight="bold")
    ax.set_xlim(1700, 1900)

    # Add era labels
    era_mids = {"pre_industrial": 1739, "transition": 1799, "early_industrial": 1839, "late_industrial": 1879}
    ymax = ax.get_ylim()[1]
    for era_name, mid in era_mids.items():
        ax.text(mid, ymax * 0.92, era_name.replace("_", "\n"), ha="center", fontsize=6, color="grey")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_B3_header_diversity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_enriched_data()

    long_df = analyse_innovation_vocabulary(df)
    plot_innovation_vocabulary(long_df)

    sch = analyse_scholarship_trajectory(df)
    plot_scholarship_trajectory(sch)

    hdr_result = analyse_header_diversity(df)
    plot_header_diversity(hdr_result)

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
