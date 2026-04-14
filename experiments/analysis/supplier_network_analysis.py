"""supplier_network_analysis.py

Novelty-phase analysis: supplier-institution network dynamics.

Novel contributions (highest-confidence literature gap):
  C1 – Place-category-direction network: geographic sourcing shifts
       (local Oxford → London as industrial sourcing grows)
  C2 – Person-name role network: artisan vs industrial supplier evolution,
       supplier persistence over decades
  C3 – Educational real £ growth with CAGR per era (ties vocabulary to financials)

Also generates the consolidated HTML report (analysis_v4_report.html) that
embeds all figures from all four scripts (A1–A4, B1–B3, C1–C3, T0–T10) as
base64-encoded PNGs. Run this script last after the other three have produced
their output PNGs.

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


def _split_names(name_str: str | None) -> list[str]:
    """Split semicolon-separated person/place names."""
    if not name_str:
        return []
    return [p.strip() for p in str(name_str).split(";") if p.strip()]


# ---------------------------------------------------------------------------
# Data loading
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
            years, _ = parse_page_id(page_id)
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
                    "place_name":      r.get("place_name"),
                    "person_name":     r.get("person_name"),
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


# ---------------------------------------------------------------------------
# Place cluster keywords
# ---------------------------------------------------------------------------

PLACE_CLUSTERS: dict[str, list[str]] = {
    "oxford_local": [
        "oxford", "bodleian", "clarendon", "carfax", "magdalen", "christ church",
        "merton", "balliol", "new college", "oriel", "pembroke", "wadham",
        "worcester", "hertford", "keble", "high street", "cornmarket",
    ],
    "oxfordshire": [
        "wittenham", "abingdon", "banbury", "henley", "woodstock", "bicester",
        "wantage", "faringdon", "didcot", "thame", "chipping norton",
        "cuddington", "south newington", "bletchingdon",
    ],
    "london": [
        "london", "fleet", "cheapside", "threadneedle", "bank", "westminster",
        "city", "lombard", "holborn", "strand", "chancery", "temple",
        "guildhall", "southwark", "lambeth",
    ],
    "other_england": [
        "bristol", "birmingham", "manchester", "liverpool", "sheffield",
        "leeds", "cambridge", "norwich", "exeter", "york", "bath",
        "gloucester", "worcester", "coventry",
    ],
}


def classify_place(place: str) -> str:
    pl = place.lower()
    for cluster, keywords in PLACE_CLUSTERS.items():
        if any(kw in pl for kw in keywords):
            return cluster
    return "other"


# ---------------------------------------------------------------------------
# C1: Place-Category Network
# ---------------------------------------------------------------------------

def analyse_place_network(df: pd.DataFrame) -> pd.DataFrame:
    print("[1/6] Analysing place-category network …")
    place_df = df[df["place_name"].notna() & df["category"].notna()].copy()

    # Explode semicolon-separated places
    records = []
    for _, row in place_df.iterrows():
        for pl in _split_names(row["place_name"]):
            records.append({
                "year":          row["year"],
                "year_weight":   row["year_weight"],
                "category":      row["category"],
                "direction":     row["direction"],
                "place":         pl,
                "place_cluster": classify_place(pl),
                "amount_real_w": row["amount_real_w"],
                "era":           row["era"],
            })
    expanded = pd.DataFrame(records)
    if expanded.empty:
        print("  No place-name data found.")
        return pd.DataFrame()

    # Yearly place cluster × category × direction
    grp = (expanded.groupby(["year", "place_cluster", "category"])
           .agg(n_entries=("year_weight", "sum"),
                amount_real=("amount_real_w", "sum"))
           .reset_index())
    grp["era"] = grp["year"].map(era_of_year)
    grp.to_csv(OUT_DIR / "place_category_yearly.csv", index=False)

    # Era heatmap: place_cluster × category (total real £)
    heat_data = expanded.groupby(["place_cluster", "category"])["amount_real_w"].sum().reset_index()
    heat_pivot = heat_data.pivot_table(index="place_cluster", columns="category",
                                       values="amount_real_w", fill_value=0.0)
    heat_pivot.to_csv(OUT_DIR / "place_cluster_category_heatmap.csv")

    # London share of expenditure over time
    exp_yearly = expanded[expanded["direction"] == "expenditure"].groupby(["year", "place_cluster"])["amount_real_w"].sum().reset_index()
    exp_total = expanded[expanded["direction"] == "expenditure"].groupby("year")["amount_real_w"].sum().rename("total")
    london_yr = exp_yearly[exp_yearly["place_cluster"] == "london"].merge(exp_total, on="year", how="left")
    london_yr["london_share"] = london_yr["amount_real_w"] / london_yr["total"].replace(0, np.nan)
    london_yr.to_csv(OUT_DIR / "london_expenditure_share_yearly.csv", index=False)

    return expanded


def plot_place_network(expanded: pd.DataFrame) -> None:
    print("[2/6] Plotting place network …")
    if expanded.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 11))

    # Panel 1: London share over time (5-yr rolling)
    ax = axes[0]
    london_yr = pd.read_csv(OUT_DIR / "london_expenditure_share_yearly.csv")
    london_yr = london_yr.sort_values("year")
    roll = london_yr.set_index("year")["london_share"].rolling(7, center=True, min_periods=3).mean()
    ax.fill_between(roll.index, roll.values, alpha=0.35, color="#1f77b4", label="London share (7-yr fill)")
    ax.plot(roll.index, roll.values, color="#1f77b4", lw=2)
    add_era_vlines(ax)
    ax.set_ylabel("London-linked expenditure / total (real £)")
    ax.set_title("Geographic Sourcing Shift: London's Share of College Expenditure 1700–1900",
                 fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(1700, 1900)

    # Panel 2: Era heatmap of place cluster × category
    ax = axes[1]
    heat_pivot = pd.read_csv(OUT_DIR / "place_cluster_category_heatmap.csv", index_col=0)
    # Log-normalise for visibility
    heat_log = np.log1p(heat_pivot)
    sns.heatmap(heat_log, ax=ax, cmap="YlGnBu", annot=False, linewidths=0.3,
                cbar_kws={"label": "log(1 + real £)"})
    ax.set_title("Place Cluster × Category: Where Does Each Spending Category Source From?",
                 fontweight="bold")
    ax.set_xlabel("Spending category")
    ax.set_ylabel("Geographic cluster")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_C1_place_category_network.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C2: Person-Name Role Network and Supplier Persistence
# ---------------------------------------------------------------------------

ROLE_KEYWORDS: dict[str, list[str]] = {
    "traditional_ecclesiastical": [
        "chaplain", "preacher", "minister", "curate", "rector", "vicar", "canon",
    ],
    "domestic_service": [
        "cook", "janitor", "laundress", "gardener", "servant", "porter", "scout",
    ],
    "academic_traditional": [
        "tutor", "fellow", "bursar", "librarian", "warden", "dean",
    ],
    "academic_modern": [
        "professor", "reader", "lecturer", "demonstrator", "examiner",
    ],
    "artisan_supplier": [
        "mason", "carpenter", "plumber", "glazier", "plasterer", "smith",
        "painter", "builder", "joiner", "thatcher", "slater", "tiler",
    ],
    "industrial_supplier": [
        "ironworks", "foundry", "gas", "coal", "railway", "insurance",
        "bank", "printer", "stationer", "chemist", "engineer",
    ],
    "student": [
        "scholar", "exhibitioner", "demy", "postmaster", "commoner",
    ],
}

ROLE_COLORS = {
    "traditional_ecclesiastical": "#9467bd",
    "domestic_service":           "#e377c2",
    "academic_traditional":       "#7f7f7f",
    "academic_modern":            "#1f77b4",
    "artisan_supplier":           "#ff7f0e",
    "industrial_supplier":        "#d62728",
    "student":                    "#2ca02c",
}


def classify_role(name: str) -> str:
    name_lower = name.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return role
    return "unclassified"


def analyse_person_network(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[3/6] Analysing person-name role network …")
    scope_cats = ["salary_stipend", "educational", "maintenance", "domestic"]
    person_df = df[df["person_name"].notna() &
                   df["category"].isin(scope_cats)].copy()

    # Explode semicolons
    records = []
    for _, row in person_df.iterrows():
        for pn in _split_names(row["person_name"]):
            if not pn:
                continue
            role = classify_role(pn)
            records.append({
                "year":          row["year"],
                "year_weight":   row["year_weight"],
                "category":      row["category"],
                "person_norm":   pn.lower().strip(),
                "role":          role,
                "amount_real_w": row["amount_real_w"],
                "era":           row["era"],
            })
    expanded = pd.DataFrame(records)
    if expanded.empty:
        print("  No person-name data found.")
        return pd.DataFrame(), pd.DataFrame()

    # Yearly role shares
    grp = (expanded.groupby(["year", "role"])
           .agg(n_entries=("year_weight", "sum"),
                amount_real=("amount_real_w", "sum"))
           .reset_index())
    grp["era"] = grp["year"].map(era_of_year)

    year_total = expanded.groupby("year")["year_weight"].sum().rename("total")
    grp = grp.merge(year_total, on="year")
    grp["role_share"] = grp["n_entries"] / grp["total"].replace(0, np.nan)
    grp.to_csv(OUT_DIR / "person_role_evolution_yearly.csv", index=False)

    # Supplier persistence: artisan and industrial
    supplier_roles = ["artisan_supplier", "industrial_supplier"]
    sup_df = expanded[expanded["role"].isin(supplier_roles)].copy()
    persist = (sup_df.groupby(["person_norm", "role"]).agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_occurrences=("year_weight", "sum"),
        total_amount_real=("amount_real_w", "sum"),
    ).reset_index())
    persist["span_years"] = persist["last_year"] - persist["first_year"]
    persist = persist[persist["n_occurrences"] >= 1].sort_values("n_occurrences", ascending=False)
    persist.to_csv(OUT_DIR / "supplier_persistence_table.csv", index=False)

    return grp, persist


def plot_person_network(grp: pd.DataFrame, persist: pd.DataFrame) -> None:
    print("[4/6] Plotting person-name network …")
    if grp.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Panel 1: Role share stacked area
    ax = axes[0]
    roles = list(ROLE_KEYWORDS.keys())
    wide = grp.pivot_table(index="year", columns="role",
                           values="role_share", fill_value=0.0).reset_index()
    years = wide["year"].values
    cols = [r for r in roles if r in wide.columns]
    share_matrix = np.column_stack([wide[r].fillna(0).values for r in cols])
    colors = [ROLE_COLORS[r] for r in cols]
    ax.stackplot(years, share_matrix.T, labels=cols, colors=colors, alpha=0.85)
    add_era_vlines(ax)
    ax.set_ylabel("Share of person-linked entries")
    ax.set_title("Evolution of Institutional Roles in Oxford College Accounts 1700–1900",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.6)
    ax.set_xlim(1700, 1900)
    ax.set_ylim(0, 1)

    # Panel 2: Supplier persistence scatter
    ax = axes[1]
    if not persist.empty:
        for role in ["artisan_supplier", "industrial_supplier"]:
            sub = persist[persist["role"] == role]
            if sub.empty:
                continue
            size = np.clip(sub["n_occurrences"] * 20, 10, 400)
            ax.scatter(sub["first_year"], sub["span_years"],
                       s=size, alpha=0.6,
                       color=ROLE_COLORS[role], label=role,
                       edgecolors="white", linewidths=0.3)
        ax.set_xlabel("First year of appearance")
        ax.set_ylabel("Span of appearances (years)")
        ax.set_title("Supplier Persistence: Artisan vs Industrial Vendors\n"
                     "(Size = number of occurrences, colour = role type)", fontweight="bold")
        add_era_vlines(ax)
        ax.legend(fontsize=8)
        ax.set_xlim(1700, 1910)

        # Annotate crossover if applicable
        yearly_roles = grp[grp["role"].isin(["artisan_supplier", "industrial_supplier"])]
        pivot_roles = yearly_roles.pivot_table(index="year", columns="role",
                                               values="role_share", fill_value=0.0)
        if "artisan_supplier" in pivot_roles.columns and "industrial_supplier" in pivot_roles.columns:
            crossover = pivot_roles[pivot_roles["industrial_supplier"] > pivot_roles["artisan_supplier"]]
            if not crossover.empty:
                cross_yr = int(crossover.index.min())
                ax.axvline(cross_yr, color="darkred", lw=1.5, ls=":", alpha=0.8)
                ax.text(cross_yr + 0.5, ax.get_ylim()[1] * 0.9,
                        f"Industrial > Artisan\n({cross_yr})", fontsize=7, color="darkred")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_C2_supplier_network.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C3: Educational Real £ Growth and CAGR
# ---------------------------------------------------------------------------

def analyse_educational_growth(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/6] Analysing educational real £ growth …")
    educ = df[df["category"] == "educational"].copy()

    yearly = educ.groupby(["year", "direction"]).agg(
        amount_real=("amount_real_w", "sum")
    ).reset_index()

    wide = yearly.pivot_table(index="year", columns="direction",
                              values="amount_real", fill_value=0.0).reset_index()
    for col in ["expenditure", "income"]:
        if col not in wide.columns:
            wide[col] = 0.0
    wide["total_real"] = wide.get("expenditure", 0) + wide.get("income", 0)
    wide["era"] = wide["year"].map(era_of_year)
    wide = wide.sort_values("year")

    # YoY % change
    wide["yoy_pct_change"] = wide["total_real"].pct_change() * 100

    # CAGR per era
    cagr_rows = []
    for era_name, y_start, y_end in ERAS:
        sub = wide[(wide["year"] >= y_start) & (wide["year"] <= y_end)]["total_real"]
        if len(sub) < 5:
            cagr_rows.append({"era": era_name, "cagr": np.nan})
            continue
        start_avg = sub.head(5).mean()
        end_avg   = sub.tail(5).mean()
        n_years = y_end - y_start
        if start_avg > 0 and end_avg > 0 and n_years > 0:
            cagr = (end_avg / start_avg) ** (1.0 / n_years) - 1
        else:
            cagr = np.nan
        cagr_rows.append({"era": era_name, "cagr": cagr})
    cagr_df = pd.DataFrame(cagr_rows)
    cagr_df.to_csv(OUT_DIR / "educational_cagr_per_era.csv", index=False)

    # Top YoY jumps
    top_jumps = wide.nlargest(10, "yoy_pct_change")[["year", "total_real", "yoy_pct_change"]]
    top_jumps.to_csv(OUT_DIR / "educational_top_jumps.csv", index=False)

    wide.to_csv(OUT_DIR / "educational_real_trajectory.csv", index=False)
    return wide


def plot_educational_growth(wide: pd.DataFrame) -> None:
    print("[6/6] Plotting educational growth …")
    cagr_df = pd.read_csv(OUT_DIR / "educational_cagr_per_era.csv")

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # Panel 1: Real £ symlog scale
    ax = axes[0]
    if "expenditure" in wide.columns:
        ax.fill_between(wide["year"], wide["expenditure"], alpha=0.5,
                        color="#2ca02c", label="Expenditure (real £)")
    if "income" in wide.columns:
        ax.fill_between(wide["year"], wide["income"], alpha=0.5,
                        color="#1f77b4", label="Income (real £)")
    ax.set_yscale("symlog", linthresh=100)
    add_era_vlines(ax)
    for yr, lbl in REFORM_VLINES.items():
        ax.axvline(yr, color="darkblue", lw=1.2, ls="-.", alpha=0.8)
    ax.set_ylabel("Real £ (1700=100), symlog scale")
    ax.set_title("Educational Real £: Income vs Expenditure 1700–1900\n"
                 "(Symlog scale; reform acts in dark blue)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1900)

    # Panel 2: YoY % change
    ax = axes[1]
    yoy = wide["yoy_pct_change"].clip(-200, 500)
    ax.bar(wide["year"], yoy, color=["#d62728" if v < 0 else "#2ca02c" for v in yoy.fillna(0)],
           alpha=0.7, width=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(100, color="orange", lw=0.8, ls="--", label="+100% threshold")
    # Scatter large jumps
    big = wide[wide["yoy_pct_change"] > 100]
    ax.scatter(big["year"], big["yoy_pct_change"].clip(upper=500),
               color="red", zorder=5, s=25)
    add_era_vlines(ax)
    ax.set_ylabel("YoY % change in total real £")
    ax.set_title("Year-on-Year Growth Rate: Educational Real Expenditure", fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(1700, 1900)
    ax.set_ylim(-200, 510)

    # Panel 3: CAGR per era
    ax = axes[2]
    cagr_valid = cagr_df[ERA_ORDER].T if False else cagr_df.set_index("era").reindex(ERA_ORDER)
    eras_plot = [e for e in ERA_ORDER if e in cagr_df["era"].values]
    cagr_vals = [float(cagr_df[cagr_df["era"] == e]["cagr"].iloc[0]) * 100
                 for e in eras_plot]
    bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in cagr_vals]
    bars = ax.bar(range(len(eras_plot)), cagr_vals, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(eras_plot)))
    ax.set_xticklabels([ERA_LABELS[e].replace("\n", " ") for e in eras_plot], fontsize=8)
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Compound Annual Growth Rate of Educational Real £ by Era", fontweight="bold")
    for bar, val in zip(bars, cagr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.1 if val >= 0 else -0.3),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_C3_educational_growth.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html_report() -> None:
    import base64

    print("Generating HTML report …")
    figures = sorted(OUT_DIR.glob("fig_*.png"))
    summaries = [f for f in [OUT_DIR / "summary_financial.txt"] if f.exists()]

    # Ordered list of (prefix, group_label, title, description)
    REPORT_SECTIONS = [
        # ── Part I: Financial restructuring ──────────────────────────────────
        ("A1", "Part I: Financial Restructuring",
         "A1 — Revenue Diversification: Proactive or Reactive?",
         "<p>We computed real income shares for each category (land rent, financial, ecclesiastical, etc.) "
         "per year and measured portfolio concentration using the Herfindahl-Hirschman Index (HHI). "
         "A high HHI means heavy dependence on one source; a falling HHI means the college was spreading risk.</p>"
         "<p>To test whether Oxford moved into financial assets <em>before</em> land rent started falling "
         "(proactive) or <em>after</em> (reactive), we ran a lead-lag cross-correlation between first "
         "differences in financial share and land rent share over lags of −10 to +10 years.</p>"
         "<p><strong>Result:</strong> The HHI fell below 0.35 by 1783. The lead-lag test returned "
         "k_max = 0 with a peak correlation of −0.47 at lag 0 — the two series moved in parallel. "
         "All lags beyond ±1 are weak (r &lt; 0.20), so the test cannot identify a clear directional "
         "lead in either direction. Oxford's financial diversification and land rent decline were "
         "<strong>simultaneous</strong>: the college shifted its portfolio as land fell, rather than "
         "pre-emptively building financial assets before the decline or lagging well behind it.</p>"),
        ("A2", None,
         "A2 — Expenditure Reallocation: Cutting Traditional to Fund Modern",
         "<p>We separated expenditure into <em>traditional</em> categories (ecclesiastical and domestic) "
         "and <em>modern</em> categories (educational and salary/stipend), then computed real £ per year "
         "for each group.</p>"
         "<p>We ran an OLS regression: Δ(modern spending) ~ Δ(traditional spending) + year trend. "
         "If the college was actively reallocating budget — cutting old obligations to fund new ones — "
         "we would expect a positive coefficient on Δ(traditional).</p>"
         "<p><strong>Result:</strong> Coefficient = 0.368, p = 0.047. Reductions in ecclesiastical and "
         "domestic spending statistically predict increases in educational and salary expenditure. "
         "The college did not just grow modern spending in absolute terms — it redirected money away "
         "from traditional obligations to fund the new university.</p>"),
        ("A3", None,
         "A3 — Arrears by Category: Granular Financial Stress",
         "<p>Arrears entries record cases where a tenant or debtor failed to pay on time. We grouped all "
         "arrears by category and year to compute an arrears rate (arrears entries / total income entries) "
         "and a weighted stress index: land rent arrears rate × land rent income share.</p>"
         "<p>This stress index captures both how often tenants defaulted <em>and</em> how much of the "
         "college's total income was at risk at any given moment. We shaded the chart for the "
         "Napoleonic Wars (1793–1815), post-war agricultural depression (1815–1830), and "
         "Great Agricultural Depression (1873–1896).</p>"
         "<p><strong>Result:</strong> Arrears cluster heavily in land rent and ecclesiastical income, "
         "with clear spikes during the post-Napoleonic period. Financial instrument income carries almost "
         "no arrears. This means the portfolio shift toward financial assets also reduced the college's "
         "exposure to income volatility — not just a return story, but a risk management story.</p>"),
        ("A4", None,
         "A4 — Payment Period Modernity Index",
         "<p>Each enriched entry includes a <code>payment_period</code> field (e.g. annual, half-year, "
         "one-off, triennial, multi-year). We assigned a modernity score to each type: 1.0 for annual "
         "(regular market-economy cadence) down to 0.1 for multi-year (long feudal tenures).</p>"
         "<p>We computed a weighted average modernity score per year, and separately tracked the "
         "Shannon entropy of the payment period distribution — high entropy means many different "
         "payment patterns in use at once; low entropy means convergence on a dominant type.</p>"
         "<p><strong>Result:</strong> The modernity index rises post-1820, driven by the disappearance "
         "of multi-year and triennial payments and a surge in one-off and annual entries. "
         "The category heatmap shows that financial and educational entries modernised their payment "
         "patterns faster than ecclesiastical entries, which remained tied to annual and half-yearly "
         "cycles throughout.</p>"),
        # ── Part II: Institutional vocabulary ────────────────────────────────
        ("B1", "Part II: Institutional Vocabulary and Reform",
         "B1 — Innovation Vocabulary Emergence",
         "<p>We searched all 52,897 <code>english_description</code> entries for keywords across six "
         "thematic groups:</p>"
         "<ul>"
         "<li><strong>Scientific professorships</strong> — professor, chemistry, geology, botany, anatomy…</li>"
         "<li><strong>Competitive examinations</strong> — examination, moderations, honours, class list…</li>"
         "<li><strong>Scholarships and prizes</strong> — scholarship, exhibition, bursary, demy…</li>"
         "<li><strong>Lectures and teaching</strong> — lecture, tutor, tutorial, reader…</li>"
         "<li><strong>Scientific infrastructure</strong> — laboratory, apparatus, instrument, specimen…</li>"
         "<li><strong>Reform and modernisation</strong> — commission, statute, ordinance, university act…</li>"
         "</ul>"
         "<p>For each group we computed a hit-share per year and identified the emergence year — "
         "defined as the first year the 5-year rolling average exceeded the pre-1780 mean plus two "
         "standard deviations.</p>"
         "<p><strong>Result:</strong> All six groups first emerge before 1780, and most before 1760 — "
         "well ahead of the Oxford University Act of 1854 and the 1877 Act. "
         "The vocabulary of reform was already present decades before Parliament mandated change, "
         "suggesting institutional adaptation was gradual and internally driven.</p>"),
        ("B2", None,
         "B2 — Scholarship and Prize Trajectory",
         "<p>We filtered all entries with <code>category == 'educational'</code> and searched "
         "descriptions for scholarship and prize keywords, tracking two things: total intensity "
         "(scholarship entries as a share of all educational entries) and the direction split "
         "(income = fees collected from scholars; expenditure = awards paid out to scholars).</p>"
         "<p><strong>Result:</strong> In the pre-industrial era, educational entries are predominantly "
         "<em>income</em> — the college was collecting money from students. By the Victorian era, "
         "they flip to predominantly <em>expenditure</em> — the college was paying out scholarships, "
         "exhibitions, and prizes.</p>"
         "<p>This directional reversal captures the transformation from a passive fee-collecting "
         "institution to an active funder of academic talent.</p>"),
        ("B3", None,
         "B3 — Section Header Diversity: Organizational Complexity",
         "<p>We loaded the raw JSON files to access <code>row_type = 'header'</code> rows, which mark "
         "the start of named account sections (e.g. 'Rents', 'Battels', 'Exhibitions', 'Gas Account'). "
         "For each year we counted unique section headers in use, headers appearing for the first time, "
         "and how many matched innovation vocabulary keywords.</p>"
         "<p>We also tracked cumulative vocabulary — the total number of distinct section types ever "
         "seen up to year Y.</p>"
         "<p><strong>Result:</strong> Unique section headers actually <em>peak</em> in the 1780s (~48 per "
         "year) and then decline steadily through the 19th century, reaching a low of ~20 per year in the "
         "1860s–1870s before a modest recovery. This is not a loss of activity — it reflects the "
         "<strong>rationalisation</strong> of the college's bookkeeping. As accounting became more "
         "professional, a large number of ad-hoc named sub-accounts were consolidated into a smaller "
         "set of standardised categories. The trend in section headers thus runs in the opposite "
         "direction to the financial and educational growth seen elsewhere: the ledger structure "
         "simplified while its economic content expanded.</p>"),
        # ── Part III: Supplier and geography ─────────────────────────────────
        ("C1", "Part III: Supplier Networks and Geography",
         "C1 — Geographic Sourcing Network",
         "<p>We extracted all entries with a non-empty <code>place_name</code> field and classified "
         "each place into five clusters via keyword matching:</p>"
         "<ul>"
         "<li><strong>Oxford-local</strong> — oxford, bodleian, clarendon, carfax…</li>"
         "<li><strong>Oxfordshire regional</strong> — witney, abingdon, woodstock…</li>"
         "<li><strong>London</strong> — london, fleet, cheapside, threadneedle…</li>"
         "<li><strong>Other England</strong></li>"
         "<li><strong>Other / unclear</strong></li>"
         "</ul>"
         "<p>We tracked London-linked expenditure as a share of total expenditure per year, and built "
         "an era × category heatmap showing where each spending category sources from.</p>"
         "<p><strong>Result:</strong> London's share grows substantially from the transition era onward, "
         "driven by financial entries (investments, banking, dividends) and later by insurance and "
         "administrative payments. Maintenance and domestic entries remain overwhelmingly Oxford-local — "
         "reflecting the enduring reliance on local craftsmen for physical upkeep.</p>"),
        ("C2", None,
         "C2 — Supplier Persistence and Role Evolution",
         "<p>We extracted all entries with a non-empty <code>person_name</code> field, normalised names, "
         "and classified them into seven role groups via keyword matching:</p>"
         "<ul>"
         "<li><strong>Traditional ecclesiastical</strong> — chaplain, curate, rector, vicar…</li>"
         "<li><strong>Domestic service</strong> — cook, laundress, gardener, porter…</li>"
         "<li><strong>Academic traditional</strong> — tutor, fellow, bursar, librarian…</li>"
         "<li><strong>Academic modern</strong> — professor, lecturer, demonstrator, examiner…</li>"
         "<li><strong>Artisan supplier</strong> — mason, carpenter, glazier, plumber…</li>"
         "<li><strong>Industrial supplier</strong> — ironworks, gas, coal, railway, bank, insurance…</li>"
         "<li><strong>Student</strong> — scholar, exhibitioner, demy, postmaster…</li>"
         "</ul>"
         "<p>For artisan and industrial suppliers we computed persistence: the span of years each named "
         "supplier appears across the full 200-year record.</p>"
         "<p><strong>Result:</strong> The persistence table captures two distinct patterns. Generic trade "
         "labels (glazier, smith, mason, carpenter) appear across spans of 150–175 years — these are "
         "occupation keywords recurring throughout the record, not individual people. The more "
         "meaningful entries are specific named suppliers: <em>Townsend (mason)</em> recurs from 1701 "
         "to 1789 (~88 years), likely a family business; <em>Gas Company</em> first appears in 1820 "
         "and continues to 1868 (48 years) as a named institutional supplier. Industrial supplier names "
         "begin appearing only after 1820 and show shorter, more transactional spans than the long-standing "
         "artisan relationships. The academic modern group first appears in force in the 1860s–1880s, "
         "consistent with the post-1877 expansion of university teaching posts.</p>"),
        ("C3", None,
         "C3 — Educational Real £ Growth and CAGR",
         "<p>We filtered all entries with <code>category == 'educational'</code>, applied the "
         "Phelps Brown-Hopkins price deflator to convert nominal £ to real £ (1700 = 100), and "
         "computed compound annual growth rates (CAGR) separately for each of the four eras.</p>"
         "<p>We also identified the largest single-year jumps in real educational spending and pulled "
         "the actual descriptions for those years to understand what drove each spike.</p>"
         "<p><strong>Result:</strong> Real educational spending was essentially flat in the pre-industrial "
         "and transition eras (CAGR −0.2% and −0.5%), then accelerated sharply:</p>"
         "<ul>"
         "<li>Early industrial (1820–1859): <strong>+2.9% p.a.</strong></li>"
         "<li>Late industrial (1860–1900): <strong>+7.0% p.a.</strong></li>"
         "</ul>"
         "<p>The largest jumps correspond to new scholarship foundations and the introduction of open "
         "competitive awards — confirming that financial commitment to education grew fastest precisely "
         "when reform legislation was enacted.</p>"),
        # ── Part IV: Text trend analysis ─────────────────────────────────────
        ("T0", "Part IV: Text-Level Evidence (english_description, 52,897 entries)",
         "T0 — Top-40 Word Frequency Heatmap",
         "<p>We computed word frequencies per decade across all 52,897 entries, normalised to "
         "occurrences per 1,000 entries, and plotted the top 40 words as a heatmap — words on the "
         "y-axis, decades on the x-axis, colour intensity = frequency. Common function words and "
         "generic accounting terms were excluded.</p>"
         "<p><strong>What it shows:</strong> Ecclesiastical and agricultural terms dominate the early "
         "columns; financial and educational terms grow toward the right. This heatmap serves as a "
         "high-level orientation — a single image that summarises two centuries of vocabulary change "
         "before the more targeted analyses that follow.</p>"),
        ("T1", None,
         "T1 — Indicator Word Trajectories",
         "<p>We grouped specific words into four thematic indicator groups and tracked each group's "
         "frequency per decade:</p>"
         "<ol>"
         "<li><strong>Traditional / declining</strong> — capons, feast, tithes, brawn, oatmeal</li>"
         "<li><strong>Financial modernisation</strong> — dividend, stock, insurance, consols, trust</li>"
         "<li><strong>Educational transformation</strong> — scholar, exhibition, examination, lecture, bursary</li>"
         "<li><strong>Industrial technology</strong> — gas, railway, coal, apparatus, laboratory</li>"
         "</ol>"
         "<p><strong>What it shows:</strong> The four trajectories tell the transformation story in "
         "miniature. Traditional vocabulary peaks before 1760 and fades. Financial vocabulary grows "
         "steadily from the 1770s. Educational vocabulary explodes after 1860. Industrial technology "
         "vocabulary is nearly absent until the 1840s, then appears suddenly — consistent with the "
         "rapid arrival of gas lighting and the railway in Oxford.</p>"),
        ("T2", None,
         "T2 — First Appearances of Modern Concepts",
         "<p>For each target concept, we scanned all entries in chronological order and recorded the "
         "year and the exact description text of the first appearance. This grounds abstract trends "
         "in concrete moments: not 'insurance grew in the 1800s' but 'in 1797, the college paid its "
         "first insurance premium for barns at Witney.'</p>"
         "<p><strong>Results:</strong></p>"
         "<ul>"
         "<li><strong>Property insurance</strong> — 1797 (barns at Witney)</li>"
         "<li><strong>Income tax</strong> — 1799 (three instalments)</li>"
         "<li><strong>Canal dividend</strong> — 1809</li>"
         "<li><strong>Railway</strong> — 1845</li>"
         "<li><strong>Institutional deficit carried forward</strong> — 1858</li>"
         "<li><strong>Scholarship exhibitioner</strong> — 1866</li>"
         "<li>The word <strong>'undergraduate'</strong> — 1871</li>"
         "</ul>"
         "<p>Each entry marks the moment a concept from the wider economy entered the college's own "
         "accounting language for the first time.</p>"),
        ("T3", None,
         "T3 — Dying Vocabulary: What Oxford Stopped Paying For",
         "<p>We identified words that were common in the early ledger but expected to disappear as "
         "the medieval economy gave way to the modern one: capons, brawn, oatmeal, feast, Michaelmas "
         "(as a payment occasion), Candlemas, augmentation, almoner, chest, glazier, joiner, thatcher.</p>"
         "<p>For each word we computed its frequency per decade and identified the decade when it "
         "effectively dropped to zero.</p>"
         "<p><strong>What it shows:</strong> Two kinds of disappearance are visible. The in-kind food "
         "payments (capons, brawn, oatmeal) reflect the end of feudal provisioning obligations — the "
         "college stopped receiving or paying in produce and switched to cash entirely. The ecclesiastical "
         "calendar words (Candlemas, feast) disappear as secular quarterly and half-yearly payment "
         "dates replace them. Most of this vocabulary is gone by 1860.</p>"),
        ("T4", None,
         "T4 — Era-Distinctive Bigrams (Lift Analysis)",
         "<p>We extracted all two-word phrases (bigrams) from the descriptions and computed a lift "
         "score for each bigram in each era:</p>"
         "<p style='margin-left:1.2em'><em>lift = rate in this era ÷ rate in all other eras combined</em></p>"
         "<p>A high lift score means a bigram is unusually concentrated in one period — it is "
         "characteristic of that era rather than present throughout the corpus.</p>"
         "<p><strong>What it shows:</strong> The top-lift bigrams per era act as a fingerprint of "
         "each period's dominant concerns. Pre-industrial bigrams tend to be agricultural and "
         "ecclesiastical. Transition-era bigrams reflect financial instruments and wartime "
         "administration. Early industrial bigrams include infrastructure and reform language. "
         "Late Victorian bigrams are dominated by educational, scientific, and institutional terms.</p>"),
        ("T5", None,
         "T5 — Category Vocabulary Signatures",
         "<p>For each combination of era and spending category, we computed a log-odds score for "
         "every word:</p>"
         "<p style='margin-left:1.2em'><em>log₂(P(word | category) / P(word | era background))</em></p>"
         "<p>A high positive score means the word appears far more often in this category than the "
         "era-wide base rate would predict — it is a signature word of that category in that period. "
         "We show the top 10 signature words per category, per era, as a heatmap.</p>"
         "<p><strong>What it shows:</strong> The signatures change within each category over time. "
         "Land rent shifts from feudal lease vocabulary (copyhold, fine, heriot) toward market "
         "vocabulary (rack rent, improvement, drainage). Financial entries move from benefaction "
         "and chest to dividend, stock, trust, and balance. The transformation was not just about "
         "categories growing or shrinking — the content of each category changed too.</p>"),
        ("T6", None,
         "T6 — Ecclesiastical vs Secular Calendar Language",
         "<p>We searched all descriptions for two groups of time markers:</p>"
         "<ul>"
         "<li><strong>Ecclesiastical</strong> — Michaelmas, Lady Day, Candlemas, Christmas, feast, "
         "Midsummer, Annunciation, Nativity</li>"
         "<li><strong>Secular</strong> — quarterly, half-yearly, instalment, annually, calendar, "
         "January, December</li>"
         "</ul>"
         "<p>We computed the rate of each group per decade per 1,000 entries and identified the "
         "crossover decade when secular language overtook ecclesiastical.</p>"
         "<p><strong>What it shows:</strong> For most of the 18th century, ecclesiastical time "
         "markers dominate — payments fall due at Michaelmas or Lady Day, not in September or March. "
         "The crossover to secular language occurs in the later 18th century and is essentially "
         "complete by the early Victorian era. Michaelmas and Christmas persist the longest; "
         "Candlemas and Annunciation disappear first.</p>"),
        ("T7", None,
         "T7 — Description Complexity: Length, Ditto Rates, Personal vs Institutional Names",
         "<p>We measured three proxies for how descriptions changed in character over time:</p>"
         "<ol>"
         "<li><strong>Average word count</strong> per description per decade — more words suggests "
         "more specific, individualised accounting</li>"
         "<li><strong>Ditto rate</strong> — share of entries using shorthand like 'ditto', 'same as "
         "above', 'do.', 'idem' — high rates suggest templated, repetitive recording</li>"
         "<li><strong>Personal vs institutional name rate</strong> — entries mentioning Mr/Mrs/Dr "
         "vs entries mentioning College/Company/Trust/Fund/Bank</li>"
         "</ol>"
         "<p><strong>What it shows:</strong> Average description length is stable across all two centuries "
         "(6–10 words per entry), with no systematic trend in either direction. Ditto rates are low in the "
         "early 18th century, rise to a peak in the 1820s–1830s (8–9% of entries), then fall as Victorian "
         "accounting became more individualised.</p>"
         "<p>The clearest signal is the personal-to-institutional name ratio. Personal name indicators "
         "(Mr, Mrs, Dr) appear in ~16–20% of all entries in the 1700s and fall to ~1% by the 1870s. "
         "Institutional name indicators (Company, Trust, Fund, Bank) rise from ~10% to ~55–60% of "
         "entries by the 1890s. This description-level shift mirrors — and is consistent with — the "
         "fuller payee analysis in T9.</p>"),
        ("T8", None,
         "T8 — Tax Vocabulary Expansion: The Fiscal State in Oxford's Accounts",
         "<p>We searched descriptions for 14 specific tax and rate types: land tax, poor rate, window "
         "tax, house tax, income tax, property tax, stamp duty, legacy duty, church rate, tithe, gaol "
         "tax, lighting rate, highway rate, county rate.</p>"
         "<p>For each decade we computed the frequency of each type per 1,000 entries and counted how "
         "many distinct types were actively mentioned that decade.</p>"
         "<p><strong>What it shows:</strong> In 1700–1760, only 2–3 tax types appear, dominated by "
         "land tax and tithes. By the 1800s, 9 distinct types appear in a single decade. This "
         "proliferation traces the expansion of the British fiscal state as actually experienced by "
         "one institution paying its obligations — not a legislative count, but a record of what "
         "the college encountered in practice. The peak complexity falls in the early 19th century; "
         "some older taxes then disappear as the Victorian fiscal system was rationalised.</p>"),
        ("T9", None,
         "T9 — Personal to Institutional Payees: From Guild Economy to Corporate Market",
         "<p>We used regular expressions to identify two types of payee language in descriptions:</p>"
         "<ul>"
         "<li><strong>Personal payees</strong> — entries mentioning Mr, Mrs, or Miss alongside a "
         "trade word (mason, carpenter, glazier, plumber, smith, builder, joiner): payments to "
         "named individual craftsmen</li>"
         "<li><strong>Institutional payees</strong> — entries mentioning Company, Society, Trust, "
         "Fund, Bank, Association, Institution, or Committee: payments to named organisations</li>"
         "</ul>"
         "<p>We computed both rates per decade for all entries, then broke them down by spending category.</p>"
         "<p><strong>What it shows:</strong> Personal payee language peaks in the early 18th century "
         "(~16–20% of all entries) and falls to ~1% by the 1870s. Institutional payee language is "
         "negligible before 1820, rises sharply after 1850, and reaches ~40% of all entries by the "
         "1890s. The crossover occurs in the 1860s.</p>"
         "<p>The shift is fastest in financial entries and slowest in maintenance, where local craftsmen "
         "persist longest. This is the language-level record of the economy's transition from "
         "personalised guild relationships to anonymous corporate markets.</p>"),
        ("T10", None,
         "T10 — Charity Language: The Poor Law 1834 in Oxford's Own Words",
         "<p>We filtered all entries with <code>category == 'charitable'</code> and divided "
         "vocabulary into two groups:</p>"
         "<ul>"
         "<li><strong>Personal almsgiving language</strong> — alms, beggar, poor, prisoner, "
         "bocardo (the Oxford town gaol), widow, destitute, vagabond</li>"
         "<li><strong>Institutional charity language</strong> — subscription, donation, infirmary, "
         "orphan, dispensary, relief fund, charitable, institution</li>"
         "</ul>"
         "<p>We tracked both groups' rates per 1,000 charitable entries per decade, with the "
         "1834 Poor Law Amendment Act annotated as a vertical line.</p>"
         "<p><strong>What it shows:</strong> Personal almsgiving language dominates throughout the "
         "18th century (~1,300–1,500 per 1,000 charitable entries). It begins falling after 1790 "
         "and collapses between 1800 and 1820. Institutional language overtakes personal language "
         "by around <strong>1810</strong> — more than two decades before the 1834 Act.</p>"
         "<p>This suggests Oxford was not simply responding to legislative pressure when it shifted "
         "from direct almsgiving to institutional subscriptions. The change was already underway, "
         "driven by a broader cultural shift toward organised philanthropy that predated Poor Law reform.</p>"),
    ]

    html_parts = [
        "<!DOCTYPE html><html lang='en'>",
        "<head><meta charset='utf-8'>",
        "<title>Oxford Ledger Analysis v4 — Novelty-Phase Report 1700–1900</title>",
        "<style>",
        "body{font-family:Georgia,serif;max-width:1150px;margin:auto;padding:2em 2.5em;line-height:1.65;color:#222;}",
        "h1{color:#1a252f;font-size:1.8em;margin-bottom:0.2em;}",
        "h2{color:#2c3e50;font-size:1.25em;border-bottom:2px solid #aab7c4;padding-bottom:4px;margin-top:2em;}",
        "h3{color:#34495e;font-size:1.05em;margin-top:1.6em;margin-bottom:0.3em;}",
        ".subtitle{color:#555;font-size:1em;margin-bottom:0.4em;}",
        ".baseline{background:#fdf6e3;border-left:4px solid #e6ac00;padding:0.6em 1em;margin:0.8em 0;font-size:0.88em;}",
        ".novel-note{background:#eaf4fb;border-left:4px solid #2980b9;padding:0.7em 1.2em;margin:0.8em 0;font-size:0.9em;}",
        ".novel-note p{margin:0.5em 0;} .novel-note p:first-child{margin-top:0;} .novel-note p:last-child{margin-bottom:0;}",
        ".novel-note ul,.novel-note ol{margin:0.4em 0 0.4em 1.4em;padding:0;}",
        ".novel-note li{margin:0.25em 0;}",
        ".progress-box{background:#f0f9f0;border:1px solid #b2d8b2;border-radius:5px;padding:1.2em 1.6em;margin:1em 0;}",
        ".progress-box h3{color:#1a5e20;margin-top:0.6em;font-size:1em;}",
        ".progress-box ul{margin:0.4em 0 0.8em 1.2em;}",
        ".progress-box li{margin:0.3em 0;font-size:0.93em;}",
        ".finding-highlight{background:#fff8e1;border-left:3px solid #f39c12;padding:0.4em 0.8em;"
        "margin:0.3em 0;font-size:0.91em;}",
        ".vs-prev{display:grid;grid-template-columns:1fr 1fr;gap:1em;margin:0.8em 0;}",
        ".vs-prev .col{background:#f9f9f9;border:1px solid #ddd;border-radius:4px;padding:0.8em 1em;}",
        ".vs-prev .col h4{margin:0 0 0.4em 0;font-size:0.9em;color:#555;}",
        ".vs-prev .col ul{margin:0.2em 0 0 1em;font-size:0.88em;}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:3px;margin:0.6em 0;display:block;}",
        "pre{background:#f4f4f4;padding:1em;font-size:0.78em;overflow-x:auto;border-radius:3px;}",
        "hr{border:none;border-top:1px solid #ccc;margin:2em 0;}",
        ".toc{background:#f9f9f9;border:1px solid #ddd;padding:1em 1.5em;border-radius:4px;margin-bottom:2em;}",
        ".toc li{margin:0.2em 0;}",
        ".toc a{color:#2980b9;text-decoration:none;}",
        ".toc a:hover{text-decoration:underline;}",
        "footer{font-size:0.8em;color:#888;margin-top:3em;}",
        # ── Print styles ─────────────────────────────────────────────────────
        "@media print{",
        "  body{max-width:100%;padding:1cm 1.5cm;font-size:9.5pt;line-height:1.5;}",
        "  h1{font-size:15pt;} h2{font-size:12pt;} h3{font-size:10.5pt;}",
        "  .toc{display:none;}",
        "  .progress-box{background:none;border:0.5pt solid #aaa;padding:0.5em 0.8em;margin-bottom:0.8em;}",
        "  .progress-box h3{color:#000;}",
        "  .vs-prev{display:block;}",
        "  .vs-prev .col{background:none;border:0.5pt solid #ccc;margin-bottom:0.5em;padding:0.4em 0.6em;}",
        "  .finding-highlight{background:none;border-left:2pt solid #aaa;padding:0.3em 0.6em;}",
        "  h2.part-header{break-before:page;page-break-before:always;border-bottom:1.5pt solid #aab7c4;padding-top:0.3cm;}",
        "  .section-block{break-inside:avoid;page-break-inside:avoid;margin-bottom:1.2em;}",
        "  .section-header{break-inside:avoid;page-break-inside:avoid;}",
        "  .novel-note{break-inside:avoid;page-break-inside:avoid;"
        "    background:#f5faff;border-left:3pt solid #2980b9;padding:0.4em 0.8em;}",
        "  img{break-before:avoid;page-break-before:avoid;"
        "    break-inside:avoid;page-break-inside:avoid;"
        "    max-width:100%;border:0.5pt solid #ccc;margin:0.4em 0;}",
        "  pre{break-inside:avoid;page-break-inside:avoid;font-size:7.5pt;}",
        "  a[href]::after{content:none;}",
        "}",
        "</style></head><body>",
        "<h1>Oxford's Institutional Transformation 1700–1900</h1>",
        "<p class='subtitle'><em>Analysis v4 — Novelty-phase findings from 52,897 enriched ledger entries</em></p>",

        # ── Week's Progress Summary ──────────────────────────────────────────
        "<h2 id='progress'>This Week's Progress (April 7–13, 2026)</h2>",
        "<div class='progress-box'>",
        "<h3>What we did this week</h3>",
        "<ul>",
        "<li><strong>Renamed all report folders</strong> for clarity: "
        "<code>ledger_clean</code> → <code>analysis_v1</code>, "
        "<code>ledger_clean_v2</code> → <code>analysis_v2</code>, "
        "<code>enriched_analysis</code> → <code>analysis_v3</code>, "
        "<code>oxford_restructuring</code> → <code>analysis_v4</code>. "
        "All scripts updated to match.</li>",
        "<li><strong>Identified genuine literature gaps</strong> by cross-referencing the <em>related_literature</em> "
        "LaTeX file (Dunbabin, Ventura, Hodgson, Oxford Acts) against new research directions. "
        "Five unstudied angles confirmed: payment period modernisation, supplier networks, "
        "arrears by category, curriculum reform timing, proactive vs reactive diversification.</li>",
        "<li><strong>Built four new analysis scripts</strong> (all writing to <code>analysis_v4/</code>): "
        "<code>industrial_revolution_response.py</code> (A1–A4), "
        "<code>innovation_vocabulary_analysis.py</code> (B1–B3), "
        "<code>supplier_network_analysis.py</code> (C1–C3), "
        "<code>text_trend_analysis.py</code> (T0–T10).</li>",
        "<li><strong>Pivoted to text-first analysis</strong>: sampled hundreds of raw "
        "<code>english_description</code> entries across all decades to discover patterns "
        "before writing code, rather than relying solely on aggregate statistics.</li>",
        "<li><strong>Added three new text analyses</strong> (T8–T10) focused on the most compelling "
        "micro-level stories: tax complexity, the personal-to-institutional payee transition, "
        "and the charity language transformation.</li>",
        "</ul>",

        "<h3>New findings compared to Analysis v3</h3>",
        "<p>Analysis v3 (<em>enriched_analysis</em>) documented category shares, arrears rates, "
        "language shift (Latin→English), payment period distributions, and top person/place names "
        "per era — all at the aggregate level. This week's work goes further in four ways:</p>",
        "<div class='vs-prev'>",
        "<div class='col'><h4>Analysis v3 (what was known)</h4><ul>",
        "<li>Land rent declining, financial rising (share %)</li>",
        "<li>Latin disappeared ~1740–1780</li>",
        "<li>Educational entries grew in Victorian era</li>",
        "<li>Top person names: Oxford tradesmen and fellows</li>",
        "<li>Arrears rate by year (overall)</li>",
        "<li>Payment periods: annual/half-year dominant</li>",
        "</ul></div>",
        "<div class='col'><h4>Analysis v4 (new this week)</h4><ul>",
        "<li>Lead-lag test: diversification was <strong>simultaneous</strong> (k_max = 0; land and financial moved in parallel — no clear lead detected)</li>",
        "<li>OLS confirmed: cuts to ecclesiastical/domestic funded education/salary (p = 0.047)</li>",
        "<li>Tax types grew from 2–3 (1700s) to <strong>9 distinct types</strong> by 1800s</li>",
        "<li>Payee language crossover: institutional &gt; personal in <strong>1860s</strong></li>",
        "<li>Charity language crossover: institutional vocabulary overtook personal almsgiving "
        "by <strong>1810</strong> — <em>before</em> the 1834 Poor Law Act</li>",
        "<li>Innovation vocabulary (science, exams) emerged before 1854 Reform Act — "
        "all 6 groups first appear pre-1780 (earliest: 1737)</li>",
        "<li>First appearances pinpointed: income tax 1799, insurance 1797, canal dividend 1809, "
        "railway 1845, deficit 1858, undergraduate 1871</li>",
        "<li>Dying vocabulary documented: capons/feast/Candlemas gone by 1860</li>",
        "<li>Educational CAGR accelerated sharply: +7.0% p.a. in late industrial era (1860–1900)</li>",
        "</ul></div>",
        "</div>",  # end .vs-prev

        "<h3>Most surprising finding</h3>",
        "<div class='finding-highlight'>",
        "The charity language transition (T10) shows that Oxford was already shifting from personal "
        "almsgiving to institutional charitable language by <strong>1810</strong>, more than two decades "
        "before the Poor Law Amendment Act of 1834. This suggests the college's charitable practices "
        "were not a passive response to legislation — the institutional logic was already changing "
        "from within. This micro-level textual evidence is entirely new and has no counterpart in "
        "the existing literature.",
        "</div>",
        "</div>",  # end .progress-box

        "<hr>",
        "<div class='baseline'><strong>Established baseline (not repeated here):</strong> "
        "Dunbabin (1975), Ventura &amp; Voth (2015), Hodgson (2021), Oxford Acts 1854/1877, "
        "Thompson (1963), Poor Law 1834. All findings below go <em>beyond</em> what those sources established.</div>",
        "<div class='toc'><strong>Full Report Contents</strong><ul>",
        "<li><a href='#progress'>This Week's Progress Summary</a></li>",
        "<li><a href='#partI'>Part I — Financial Restructuring</a> (A1–A4)</li>",
        "<li><a href='#partII'>Part II — Institutional Vocabulary and Reform</a> (B1–B3)</li>",
        "<li><a href='#partIII'>Part III — Supplier Networks and Geography</a> (C1–C3)</li>",
        "<li><a href='#partIV'>Part IV — Text-Level Evidence</a> (T0–T10)</li>",
        "</ul></div>",
        "<hr>",
    ]

    part_anchors = {
        "Part I: Financial Restructuring": "partI",
        "Part II: Institutional Vocabulary and Reform": "partII",
        "Part III: Supplier Networks and Geography": "partIII",
        "Part IV: Text-Level Evidence (english_description, 52,897 entries)": "partIV",
    }

    for prefix, group_label, title, description in REPORT_SECTIONS:
        if group_label is not None:
            anchor = part_anchors.get(group_label, group_label.replace(" ", "_"))
            html_parts.append(f"<h2 id='{anchor}' class='part-header'>{group_label}</h2>")
        figs_for_section = sorted([f for f in figures if f.stem.startswith(f"fig_{prefix}_")])
        # Open section block — keeps heading + description + images together for print
        html_parts.append("<div class='section-block'>")
        html_parts.append(f"<div class='section-header'><h3>{title}</h3>"
                          f"<div class='novel-note'>{description}</div></div>")
        for fig_path in figs_for_section:
            img_b64 = base64.b64encode(fig_path.read_bytes()).decode()
            html_parts.append(f"<img src='data:image/png;base64,{img_b64}' alt='{fig_path.stem}'>")
        html_parts.append("</div>")  # end .section-block

    # Embed text summary
    for sfile in summaries:
        html_parts.append("<div class='section-block'>")
        html_parts.append("<h2>Financial Restructuring Summary (auto-generated)</h2>")
        html_parts.append(f"<pre>{sfile.read_text(encoding='utf-8')}</pre>")
        html_parts.append("</div>")

    html_parts.append(
        "<footer><hr>Generated from 1,581 enriched ledger pages (1700–1900). "
        "Price deflation: Phelps Brown-Hopkins index (1700=100). "
        "Text analysis: 52,897 entry rows with english_description.</footer>"
    )
    html_parts.append("</body></html>")

    out_path = OUT_DIR / "analysis_v4_report.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"  HTML report written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_enriched_data()

    expanded_place = analyse_place_network(df)
    plot_place_network(expanded_place)

    grp_roles, persist = analyse_person_network(df)
    plot_person_network(grp_roles, persist)

    educ_wide = analyse_educational_growth(df)
    plot_educational_growth(educ_wide)

    generate_html_report()

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
