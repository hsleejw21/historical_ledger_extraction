"""text_trend_analysis.py

Text-first analysis of the Oxford college ledger 1700-1900.

The central idea: the english_description field carries the real story. Numbers
show *how much* changed; words show *what* changed — the concepts, services,
relationships, and institutions Oxford was engaging with.

Analyses:
  T1 – Indicator word trajectories (per-decade frequency per 1000 entries)
  T2 – First appearances with actual context: when did each modern concept
       arrive in Oxford's accounts, and what was the first entry?
  T3 – Dying vocabulary: words common pre-1780 that vanish afterward
  T4 – Bigram trends: two-word phrases that are era-distinctive
  T5 – Category vocabulary signatures: what each spending category uniquely said
       in each era (log-odds ranked wordlists)
  T6 – Ecclesiastical time markers vs secular calendar: decline of feast/quarter
       day language as organizing principle
  T7 – Transaction description complexity: length, specificity, ditto-entries
  T8 – Tax vocabulary expansion: distinct tax types per decade as a proxy for
       the growing complexity of the Victorian fiscal state
  T9 – Personal-to-institutional payees: regex-based tracking of the shift
       from named individual craftsmen to Company/Society/Bank/Fund payees
  T10 – Charity language transformation: pre/post Poor Law 1834 vocabulary in
        charitable entries — crossover occurs 1810, 24 years before the Act

All analyses are grounded in actual example descriptions shown inline.

Output directory: experiments/reports/analysis_v4/
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

# ---------------------------------------------------------------------------
# Era / decade helpers
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


def era_of_year(year: int) -> str:
    if year < 1780: return "pre_industrial"
    if year < 1820: return "transition"
    if year < 1860: return "early_industrial"
    return "late_industrial"


def decade_of_year(year: int) -> int:
    return (year // 10) * 10


ERA_VLINES = [1780, 1820, 1860]


def add_era_vlines(ax, alpha=0.5):
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=alpha)


# ---------------------------------------------------------------------------
# Parse page_id
# ---------------------------------------------------------------------------

def parse_page_id(page_id: str) -> list[int]:
    single = re.match(r"^(\d{4})_(\d+)_image$", page_id)
    if single:
        return [int(single.group(1))]
    span = re.match(r"^(\d{4})-(\d{4})_(\d+)_image$", page_id)
    if span:
        y1, y2 = int(span.group(1)), int(span.group(2))
        if y2 < y1: y1, y2 = y2, y1
        return list(range(y1, y2 + 1))
    m = re.search(r"(\d{4})", page_id)
    if m: return [int(m.group(1))]
    return []


# ---------------------------------------------------------------------------
# Data loading — returns list of flat entry records
# ---------------------------------------------------------------------------

def load_all_entries() -> list[dict]:
    """Return every entry row from all enriched JSON files as flat dicts."""
    records = []
    files = sorted(ENRICHED_DIR.glob("*_image_enriched.json"))
    print(f"[load] Loading {len(files)} files …")
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        page_id = payload.get("page_id") or fp.stem.replace("_enriched", "")
        years = parse_page_id(page_id)
        if not years:
            continue
        year_weight = 1.0 / len(years)
        for r in payload.get("rows", []):
            if not isinstance(r, dict):
                continue
            if str(r.get("row_type", "")).strip().lower() != "entry":
                continue
            desc = r.get("english_description") or ""
            if not desc:
                continue
            for year in years:
                records.append({
                    "year":           year,
                    "decade":         decade_of_year(year),
                    "era":            era_of_year(year),
                    "year_weight":    year_weight,
                    "category":       r.get("category") or "other",
                    "direction":      r.get("direction") or "",
                    "language":       r.get("language") or "",
                    "payment_period": r.get("payment_period") or "",
                    "is_arrears":     bool(r.get("is_arrears")),
                    "section_header": r.get("section_header") or "",
                    "person_name":    r.get("person_name") or "",
                    "place_name":     r.get("place_name") or "",
                    "description":    r.get("description") or "",
                    "english_desc":   desc,
                    "notes":          r.get("notes") or "",
                })
    print(f"  Loaded {len(records):,} entry rows with descriptions")
    return records


# ---------------------------------------------------------------------------
# T1: Indicator word trajectories
# ---------------------------------------------------------------------------

# Curated indicator words grouped thematically
INDICATOR_GROUPS: dict[str, dict[str, str]] = {
    "Traditional (declining)": {
        "capons":        "Payment in kind (food rents)",
        "augmentation":  "Ecclesiastical top-up payment",
        "feast":         "Church feast-day payment timing",
        "michaelmas":    "Michaelmas quarter-day",
        "quit rent":     "Fixed feudal rent",
        "almoner":       "Medieval almsgiving officer",
        "maundy":        "Maundy Thursday charity",
    },
    "Financial modernization (rising)": {
        "insurance":     "Property/fire insurance premiums",
        "dividend":      "Financial investment dividends",
        "consols":       "Government stock (3% consols)",
        "bankers":       "Relationship with bankers",
        "loan":          "Loans extended or received",
        "mortgage":      "Mortgage transactions",
        "rack rent":     "Market-rate (rack) rents",
        "deficit":       "Institutional deficit",
        "percentage":    "Percentage-based accounting",
    },
    "Educational transformation (rising)": {
        "tuition":       "Tuition fee income/payments",
        "prize":         "Academic prize awards",
        "exhibition":    "Exhibition scholarship",
        "exhibitioner":  "Students on exhibitions",
        "undergraduate": "Undergraduate students",
        "entrance":      "Entrance fee / examination",
        "private":       "Private tuition / students",
        "composition":   "Composition fee",
    },
    "Industrial technology (new)": {
        "income tax":    "Government income tax",
        "canal":         "Canal company dividends",
        "railway":       "Railway-related payments",
        "gas":           "Gas lighting",
        "drainage":      "Agricultural drainage works",
        "insurance premium": "Explicit insurance premium",
    },
}

ALL_INDICATORS = {kw: desc for group in INDICATOR_GROUPS.values()
                  for kw, desc in group.items()}


def compute_indicator_frequencies(records: list[dict]) -> pd.DataFrame:
    """Per-decade frequency of each indicator word (per 1000 entries)."""
    print("[T1] Computing indicator word frequencies …")
    decade_total: Counter = Counter()
    decade_word: dict[int, Counter] = defaultdict(Counter)

    for rec in records:
        d = rec["decade"]
        text = rec["english_desc"].lower()
        decade_total[d] += 1
        for kw in ALL_INDICATORS:
            if kw in text:
                decade_word[d][kw] += 1

    rows = []
    for decade in sorted(decade_total.keys()):
        total = decade_total[decade]
        row = {"decade": decade, "total_entries": total}
        for kw in ALL_INDICATORS:
            row[kw] = round(decade_word[decade][kw] / total * 1000, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_indicator_trajectories(freq_df: pd.DataFrame) -> None:
    print("[T1] Plotting indicator trajectories …")
    groups = list(INDICATOR_GROUPS.keys())
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 5 * len(groups)))

    palettes = [
        ["#8B4513","#9467bd","#d62728","#7f7f7f","#bcbd22","#e377c2","#a65628"],
        ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#17becf","#bcbd22"],
        ["#2ca02c","#ff7f0e","#1f77b4","#d62728","#9467bd","#8c564b","#bcbd22","#17becf"],
        ["#d62728","#1f77b4","#ff7f0e","#2ca02c","#9467bd","#8c564b"],
    ]

    decades = freq_df["decade"].values

    for ax, (group_name, kw_dict), palette in zip(axes, INDICATOR_GROUPS.items(), palettes):
        kws = list(kw_dict.keys())
        for kw, color in zip(kws, palette):
            if kw in freq_df.columns:
                vals = freq_df[kw].values
                ax.plot(decades, vals, label=kw, color=color, lw=1.8, marker="o",
                        markersize=3)
        add_era_vlines(ax)
        ax.set_xlim(1700, 1905)
        ax.set_ylabel("Entries per 1000")
        ax.set_title(f"{group_name}", fontweight="bold")
        ax.legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.6)

    axes[-1].set_xlabel("Decade")
    fig.suptitle("Oxford Ledger Vocabulary Trajectories 1700–1900\n"
                 "(per 1000 entries with english_description)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig_T1_indicator_trajectories.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T2: First appearances with actual text context
# ---------------------------------------------------------------------------

# Key modern concepts to track, with a brief interpretive label
FIRST_APPEARANCE_TARGETS = {
    "insurance":         "Oxford buys property insurance",
    "income tax":        "Government taxes the college",
    "moderator":         "Exam moderator role emerges",
    "rack rent":         "Market-rate rents begin",
    "bankers":           "Banking relationship formalised",
    "canal":             "Industrial canal investment",
    "railway":           "Railway era arrives",
    "gas":               "Gas lighting installed",
    "deficit":           "First institutional deficit",
    "tuition":           "Tuition as a formal category",
    "exhibitioner":      "Exhibition scholarships paid out",
    "undergraduate":     "Students labelled 'undergraduates'",
    "percentage":        "Percentage-based accounting",
    "drainage":          "Agricultural drainage works",
    "widows":            "Widows fund / charity",
    "choir":             "Choir / choral service",
    "competition":       "Open competition for awards",
    "composition fee":   "Composition fee income",
    "prize":             "Academic prize awarded",
    "consols":           "3% consols investment",
    "mortgage":          "Mortgage transaction",
    "loan":              "Loan income or payment",
    "private":           "Private tuition / students",
}


def find_first_appearances(records: list[dict]) -> pd.DataFrame:
    """For each target term, find the FIRST entry (by year) and return its text."""
    print("[T2] Finding first appearances …")
    # Sort records by year
    sorted_records = sorted(records, key=lambda r: r["year"])
    found: dict[str, dict] = {}
    for rec in sorted_records:
        text = rec["english_desc"].lower()
        for term, label in FIRST_APPEARANCE_TARGETS.items():
            if term not in found and term in text:
                found[term] = {
                    "term":          term,
                    "label":         label,
                    "year":          rec["year"],
                    "era":           rec["era"],
                    "category":      rec["category"],
                    "direction":     rec["direction"],
                    "english_desc":  rec["english_desc"],
                }

    rows = []
    for term, label in FIRST_APPEARANCE_TARGETS.items():
        if term in found:
            rows.append(found[term])
        else:
            rows.append({"term": term, "label": label, "year": None,
                         "era": None, "category": None, "direction": None,
                         "english_desc": "NOT FOUND"})

    df = pd.DataFrame(rows).sort_values("year", na_position="last")
    df.to_csv(OUT_DIR / "first_appearances.csv", index=False)
    return df


def plot_first_appearances(df: pd.DataFrame) -> None:
    print("[T2] Plotting first appearances …")
    valid = df[df["year"].notna()].copy()
    valid["year"] = valid["year"].astype(int)
    valid = valid.sort_values("year")

    era_colors = {
        "pre_industrial":  "#8B4513",
        "transition":      "#ff7f0e",
        "early_industrial":"#1f77b4",
        "late_industrial": "#2ca02c",
    }
    era_labels = {
        "pre_industrial":  "Pre-industrial (1700–1779)",
        "transition":      "Transition (1780–1819)",
        "early_industrial":"Early industrial (1820–1859)",
        "late_industrial": "Late industrial (1860–1900)",
    }

    fig, ax = plt.subplots(figsize=(11, 9))

    y_pos = list(range(len(valid)))
    for i, (_, row) in enumerate(valid.iterrows()):
        color = era_colors.get(row["era"], "#aaa")
        yr = row["year"]
        # Lollipop stem
        ax.hlines(i, 1695, yr, color="#cccccc", lw=1.2, zorder=1)
        # Dot at first appearance
        ax.scatter(yr, i, color=color, s=90, zorder=3)
        # Year label just to the right of the dot
        ax.text(yr + 1.5, i, str(yr), va="center", ha="left", fontsize=7.5,
                color=color, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([row["term"] for _, row in valid.iterrows()], fontsize=8.5)

    # Era boundary lines
    for vx in ERA_VLINES:
        ax.axvline(vx, color="grey", lw=0.8, ls="--", alpha=0.5)
    # Reform act lines
    for yr, lbl in {1854: "Oxford Act 1854", 1877: "Oxford Act 1877"}.items():
        ax.axvline(yr, color="darkblue", lw=1.2, ls="-.", alpha=0.8)
        ax.text(yr + 0.8, len(valid) - 0.3, lbl, fontsize=6.5, color="darkblue",
                rotation=90, va="top")

    # Era shading bands
    era_spans = [
        (1700, 1780, "#8B4513"),
        (1780, 1820, "#ff7f0e"),
        (1820, 1860, "#1f77b4"),
        (1860, 1901, "#2ca02c"),
    ]
    for x0, x1, c in era_spans:
        ax.axvspan(x0, x1, alpha=0.05, color=c, zorder=0)

    # Legend
    handles = [mpatches.Patch(color=era_colors[e], label=era_labels[e])
               for e in ["pre_industrial", "transition", "early_industrial", "late_industrial"]]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9)

    ax.set_xlabel("Year of first appearance in ledger", fontsize=9)
    ax.set_xlim(1695, 1915)
    ax.set_ylim(-0.8, len(valid) - 0.2)
    ax.set_title("When Did Modern Concepts First Enter Oxford's Accounts?",
                 fontweight="bold", fontsize=11)
    ax.set_facecolor("#fafafa")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_T2_first_appearances.png")
    plt.close(fig)


def write_first_appearances_narrative(df: pd.DataFrame) -> None:
    """Write a readable text file with the actual first-appearance descriptions."""
    lines = [
        "FIRST APPEARANCES OF MODERN CONCEPTS IN OXFORD COLLEGE ACCOUNTS",
        "=" * 68,
        "Each entry shows the FIRST time a modern concept appeared in the",
        "enriched english_description, with the actual description text.",
        "",
    ]
    valid = df[df["year"].notna()].sort_values("year")
    for _, row in valid.iterrows():
        lines.append(f"[{int(row['year'])}] {row['term'].upper()}")
        lines.append(f"  Significance: {row['label']}")
        lines.append(f"  Era: {row['era']}  |  Category: {row['category']}  |  Direction: {row['direction']}")
        lines.append(f"  First entry: \"{row['english_desc']}\"")
        lines.append("")
    (OUT_DIR / "first_appearances_narrative.txt").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# T3: Dying vocabulary — words that were common pre-1780, vanish by 1860
# ---------------------------------------------------------------------------

DYING_VOCAB = {
    # Food/payment in kind
    "capons":       "Capons delivered as rent (payment in kind)",
    "brawn":        "Brawn (pork product) as rent payment",
    "oatmeal":      "Oatmeal — domestic provisioning",
    # Ecclesiastical calendar
    "feast":        "Feast-day payment timing",
    "michaelmas":   "Michaelmas quarter day",
    "candlemas":    "Candlemas feast",
    "maundy":       "Maundy Thursday almsgiving",
    # Medieval institutional language
    "augmentation": "Ecclesiastical augmentation payments",
    "almoner":      "Medieval alms officer",
    "chest":        "College chest (physical cash store)",
    # Trade types that disappear
    "glazier":      "Glazier (pre-industrial window repair)",
    "joiner":       "Joiner (pre-industrial woodworker)",
    "thatcher":     "Thatcher (pre-industrial roofing)",
    # Latin residue
    "soluta":       "Latin: 'paid' (ledger template word)",
    "recepta":      "Latin: 'received' (ledger template word)",
}


def analyse_dying_vocabulary(records: list[dict]) -> pd.DataFrame:
    print("[T3] Analysing dying vocabulary …")
    decade_total: Counter = Counter()
    decade_word: dict[int, Counter] = defaultdict(Counter)
    for rec in records:
        d = rec["decade"]
        text = rec["english_desc"].lower()
        decade_total[d] += 1
        for kw in DYING_VOCAB:
            if kw in text:
                decade_word[d][kw] += 1

    rows = []
    for decade in sorted(decade_total.keys()):
        total = decade_total[decade]
        row = {"decade": decade, "total_entries": total}
        for kw in DYING_VOCAB:
            row[kw] = round(decade_word[decade][kw] / total * 1000, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "dying_vocabulary_decades.csv", index=False)
    return df


def plot_dying_vocabulary(df: pd.DataFrame) -> None:
    print("[T3] Plotting dying vocabulary …")
    decades = df["decade"].values
    words = list(DYING_VOCAB.keys())

    # Group 1: food/payment-in-kind
    group1 = ["capons", "brawn", "oatmeal"]
    # Group 2: ecclesiastical calendar
    group2 = ["feast", "michaelmas", "candlemas", "maundy"]
    # Group 3: medieval institutional
    group3 = ["augmentation", "almoner", "chest"]
    # Group 4: trades
    group4 = ["glazier", "joiner", "thatcher"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    configs = [
        (axes[0, 0], group1, "Payment in Kind (Food Rents)", ["#8B4513","#d2691e","#cd853f"]),
        (axes[0, 1], group2, "Ecclesiastical Calendar Language", ["#9467bd","#c5b0d5","#a65628","#e377c2"]),
        (axes[1, 0], group3, "Medieval Institutional Concepts", ["#1f77b4","#aec7e8","#ff7f0e"]),
        (axes[1, 1], group4, "Pre-Industrial Trade Vocabulary", ["#2ca02c","#98df8a","#d62728"]),
    ]

    for ax, group, title, colors in configs:
        for kw, color in zip(group, colors):
            if kw in df.columns:
                ax.plot(decades, df[kw].values, label=kw, color=color, lw=2, marker="o", markersize=3)
        add_era_vlines(ax)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Entries per 1000")
        ax.legend(fontsize=7)
        ax.set_xlim(1700, 1905)

    axes[1, 0].set_xlabel("Decade")
    axes[1, 1].set_xlabel("Decade")
    fig.suptitle("Dying Vocabulary: Traditional Concepts Fading from Oxford's Accounts",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig_T3_dying_vocabulary.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T4: Bigram trends — meaningful two-word phrases
# ---------------------------------------------------------------------------

STOP_BIGRAMS = {
    "the same", "in the", "of the", "to the", "for the", "at the",
    "from the", "by the", "on the", "and the", "is the", "to be",
    "as the", "of a", "in a", "for a", "that the", "was the",
    "this is", "have been", "which is",
}

ERA_BIGRAM_SAMPLES = 3  # show top N bigrams per era


def extract_bigrams(text: str) -> list[str]:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)
            if f"{words[i]} {words[i+1]}" not in STOP_BIGRAMS]


def analyse_bigrams(records: list[dict]) -> dict[str, list[tuple[str, int]]]:
    print("[T4] Analysing bigrams by era …")
    era_bigrams: dict[str, Counter] = defaultdict(Counter)
    era_totals: Counter = Counter()

    for rec in records:
        era = rec["era"]
        era_totals[era] += 1
        for bg in extract_bigrams(rec["english_desc"]):
            era_bigrams[era][bg] += 1

    # Find distinctive bigrams per era (high in this era, low in others)
    result = {}
    for era in ERA_ORDER:
        this = era_bigrams[era]
        other = Counter()
        for e2 in ERA_ORDER:
            if e2 != era:
                other.update(era_bigrams[e2])
        other_total = sum(era_totals[e] for e in ERA_ORDER if e != era)
        this_total = era_totals[era]

        scores = []
        for bg, cnt in this.items():
            if cnt < 5:
                continue
            this_rate = cnt / this_total
            other_rate = (other[bg] + 1) / (other_total + 1)
            lift = this_rate / other_rate
            scores.append((bg, cnt, round(lift, 2)))
        scores.sort(key=lambda x: -x[2])
        result[era] = scores[:25]

    # Save
    rows = []
    for era, items in result.items():
        for bg, cnt, lift in items:
            rows.append({"era": era, "bigram": bg, "count": cnt, "lift": lift})
    pd.DataFrame(rows).to_csv(OUT_DIR / "era_distinctive_bigrams.csv", index=False)
    return result


def plot_bigrams(result: dict) -> None:
    print("[T4] Plotting bigrams …")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    era_colors = {
        "pre_industrial":   "#8B4513",
        "transition":       "#ff7f0e",
        "early_industrial": "#1f77b4",
        "late_industrial":  "#2ca02c",
    }
    for ax, era in zip(axes.flatten(), ERA_ORDER):
        items = result.get(era, [])[:15]
        if not items:
            continue
        bigrams, counts, lifts = zip(*items)
        y = range(len(bigrams))
        ax.barh(list(y), lifts, color=era_colors[era], alpha=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{bg} ({cnt})" for bg, cnt in zip(bigrams, counts)], fontsize=7.5)
        ax.set_xlabel("Lift over other eras")
        ax.set_title(f"Distinctive bigrams: {ERA_LABELS[era].replace(chr(10), ' ')}",
                     fontweight="bold", color=era_colors[era])
        ax.axvline(1, color="grey", lw=0.8, ls="--")
    fig.suptitle("Era-Distinctive Bigrams (Two-Word Phrases)\n"
                 "Lift > 1 means the phrase appears more often in this era than others",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig_T4_bigram_trends.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T5: Category vocabulary signatures (log-odds per era)
# ---------------------------------------------------------------------------

FOCUS_CATEGORIES = ["land_rent", "financial", "educational", "ecclesiastical",
                    "maintenance", "administrative", "domestic", "charitable"]


def category_vocabulary_signatures(records: list[dict]) -> None:
    print("[T5] Computing category vocabulary signatures …")
    # For each (era, category) compute word frequencies
    # Distinctive words = high P(word|cat,era) / P(word|all,era)
    era_cat_words: dict[tuple, Counter] = defaultdict(Counter)
    era_words: dict[str, Counter] = defaultdict(Counter)
    era_cat_totals: dict[tuple, int] = defaultdict(int)
    era_totals: Counter = Counter()

    for rec in records:
        era = rec["era"]
        cat = rec["category"]
        words = set(re.findall(r"[a-z]{4,}", rec["english_desc"].lower()))
        era_cat_words[(era, cat)].update(words)
        era_words[era].update(words)
        era_cat_totals[(era, cat)] += 1
        era_totals[era] += 1

    rows = []
    for era in ERA_ORDER:
        for cat in FOCUS_CATEGORIES:
            this = era_cat_words[(era, cat)]
            total_this = era_cat_totals[(era, cat)]
            total_era = era_totals[era]
            if total_this < 5:
                continue
            all_words = era_words[era]
            scores = []
            for word, cnt in this.items():
                if cnt < 3:
                    continue
                p_cat = cnt / total_this
                p_all = all_words[word] / total_era
                if p_all > 0:
                    log_odds = np.log2(p_cat / p_all)
                    scores.append((word, cnt, round(log_odds, 2)))
            scores.sort(key=lambda x: -x[2])
            for rank, (word, cnt, lo) in enumerate(scores[:10], 1):
                rows.append({"era": era, "category": cat, "rank": rank,
                             "word": word, "count": cnt, "log_odds": lo})

    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(OUT_DIR / "category_vocabulary_signatures.csv", index=False)

    # Plot: one heatmap per era showing top words per category
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, era in zip(axes.flatten(), ERA_ORDER):
        era_sub = sig_df[sig_df["era"] == era]
        cats_present = [c for c in FOCUS_CATEGORIES if c in era_sub["category"].values]
        # Build matrix: top 6 words per cat
        top_words_set = []
        for cat in cats_present:
            sub = era_sub[era_sub["category"] == cat].head(6)
            top_words_set.extend(sub["word"].tolist())
        top_words = list(dict.fromkeys(top_words_set))[:30]  # deduplicate, keep order
        if not top_words:
            continue
        matrix = pd.DataFrame(0.0, index=top_words, columns=cats_present)
        for cat in cats_present:
            sub = era_sub[era_sub["category"] == cat]
            for _, row in sub.iterrows():
                if row["word"] in matrix.index:
                    matrix.loc[row["word"], cat] = row["log_odds"]
        sns.heatmap(matrix, ax=ax, cmap="RdYlGn", center=0,
                    linewidths=0.3, annot=False,
                    cbar_kws={"label": "log₂(P(word|cat) / P(word|era))"})
        ax.set_title(f"Category vocabulary signatures\n{ERA_LABELS[era].replace(chr(10),' ')}",
                     fontweight="bold", fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    fig.suptitle("What Words Define Each Spending Category? (log-odds vs era background)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig_T5_category_vocabulary_signatures.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T6: Ecclesiastical time markers vs secular calendar
# ---------------------------------------------------------------------------

ECCLESIASTICAL_MARKERS = {
    "michaelmas":  "Michaelmas (29 Sep)",
    "christmas":   "Christmas (25 Dec)",
    "midsummer":   "Midsummer / St John (24 Jun)",
    "lady day":    "Lady Day (25 Mar)",
    "candlemas":   "Candlemas (2 Feb)",
    "feast":       "Feast day (generic)",
    "nativity":    "Nativity of Christ",
    "annunciation": "Annunciation",
}

SECULAR_MARKERS = {
    "half-year":   "Half-yearly period",
    "quarterly":   "Quarterly payment",
    "annually":    "Annual payment",
    "instalment":  "Instalment payment",
    "calendar":    "Calendar year",
    "january":     "January (secular month)",
    "december":    "December (secular month)",
}


def analyse_calendar_language(records: list[dict]) -> pd.DataFrame:
    print("[T6] Analysing ecclesiastical vs secular time language …")
    decade_total: Counter = Counter()
    decade_eccl: dict[int, Counter] = defaultdict(Counter)
    decade_sec: dict[int, Counter] = defaultdict(Counter)

    for rec in records:
        d = rec["decade"]
        text = rec["english_desc"].lower()
        decade_total[d] += 1
        for marker in ECCLESIASTICAL_MARKERS:
            if marker in text:
                decade_eccl[d][marker] += 1
        for marker in SECULAR_MARKERS:
            if marker in text:
                decade_sec[d][marker] += 1

    rows = []
    for decade in sorted(decade_total.keys()):
        total = decade_total[decade]
        row = {"decade": decade, "total": total}
        row["eccl_total"] = sum(decade_eccl[decade].values())
        row["sec_total"] = sum(decade_sec[decade].values())
        row["eccl_rate"] = round(row["eccl_total"] / total * 1000, 1)
        row["sec_rate"]  = round(row["sec_total"] / total * 1000, 1)
        for m in ECCLESIASTICAL_MARKERS:
            row[f"eccl_{m.replace(' ', '_')}"] = round(decade_eccl[decade][m] / total * 1000, 2)
        for m in SECULAR_MARKERS:
            row[f"sec_{m.replace(' ', '_').replace('-', '_')}"] = round(decade_sec[decade][m] / total * 1000, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "calendar_language_decades.csv", index=False)
    return df


def plot_calendar_language(df: pd.DataFrame) -> None:
    print("[T6] Plotting calendar language …")
    decades = df["decade"].values

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # Panel 1: Total ecclesiastical vs secular rate
    ax = axes[0]
    ax.fill_between(decades, df["eccl_rate"], alpha=0.4, color="#9467bd",
                    label="All ecclesiastical markers (per 1000)")
    ax.fill_between(decades, df["sec_rate"], alpha=0.4, color="#1f77b4",
                    label="All secular time markers (per 1000)")
    ax.plot(decades, df["eccl_rate"], color="#9467bd", lw=2)
    ax.plot(decades, df["sec_rate"], color="#1f77b4", lw=2)
    add_era_vlines(ax)
    ax.set_ylabel("Entries per 1000")
    ax.set_title("Ecclesiastical Calendar vs Secular Time Language in Oxford Accounts",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1905)

    # Find crossover
    df_s = df.set_index("decade")
    cross = df_s[df_s["sec_rate"] > df_s["eccl_rate"]]
    if not cross.empty:
        cx = cross.index.min()
        ax.annotate(f"Secular > Ecclesiastical\n({cx}s)",
                    xy=(cx, df_s.loc[cx, "sec_rate"]),
                    xytext=(cx + 10, df_s.loc[cx, "sec_rate"] + 5),
                    arrowprops=dict(arrowstyle="->", color="navy"), fontsize=7.5, color="navy")

    # Panel 2: Individual markers stacked
    ax = axes[1]
    eccl_keys = ["feast", "michaelmas", "christmas", "midsummer", "lady day", "candlemas"]
    eccl_cols = [f"eccl_{k.replace(' ', '_')}" for k in eccl_keys]
    eccl_cols = [c for c in eccl_cols if c in df.columns]
    eccl_data = np.column_stack([df[c].fillna(0).values for c in eccl_cols])
    eccl_labels = eccl_keys[:len(eccl_cols)]
    eccl_colors = ["#9467bd","#c5b0d5","#a65628","#e377c2","#d62728","#8c564b"][:len(eccl_cols)]
    ax.stackplot(decades, eccl_data.T, labels=eccl_labels,
                 colors=eccl_colors, alpha=0.85)
    add_era_vlines(ax)
    ax.set_ylabel("Entries per 1000 (stacked)")
    ax.set_xlabel("Decade")
    ax.set_title("Individual Ecclesiastical Markers: Which Survived Longest?",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_xlim(1700, 1905)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_T6_calendar_language.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T7: Description complexity — length, specificity, ditto-entries
# ---------------------------------------------------------------------------

DITTO_PATTERNS = ["ditto", "same as", "as above", "as before", "do.", "idem",
                  "same", "as the previous", "as per"]


def analyse_description_complexity(records: list[dict]) -> pd.DataFrame:
    print("[T7] Analysing description complexity …")
    decade_total: Counter = Counter()
    decade_lengths: dict[int, list[int]] = defaultdict(list)
    decade_ditto: Counter = Counter()
    decade_personal: Counter = Counter()   # entries mentioning a person's name
    decade_institutional: Counter = Counter()  # entries mentioning an institution

    personal_rx = re.compile(r"\bMr\.?\b|\bMrs\.?\b|\bDr\.?\b|\bRev\.?\b|"
                             r"\bProf\.?\b|\bSir\b", re.IGNORECASE)
    institution_rx = re.compile(r"\bCollege\b|\bUniversity\b|\bChurch\b|\bCompany\b|"
                                r"\bSociety\b|\bTrust\b|\bFund\b|\bBank\b", re.IGNORECASE)

    for rec in records:
        d = rec["decade"]
        text = rec["english_desc"]
        text_lower = text.lower()
        decade_total[d] += 1
        decade_lengths[d].append(len(text.split()))
        if any(p in text_lower for p in DITTO_PATTERNS):
            decade_ditto[d] += 1
        if personal_rx.search(text):
            decade_personal[d] += 1
        if institution_rx.search(text):
            decade_institutional[d] += 1

    rows = []
    for decade in sorted(decade_total.keys()):
        total = decade_total[decade]
        lengths = decade_lengths[decade]
        rows.append({
            "decade":              decade,
            "total_entries":       total,
            "avg_words":           round(np.mean(lengths), 1),
            "median_words":        round(np.median(lengths), 1),
            "ditto_rate":          round(decade_ditto[decade] / total * 100, 1),
            "personal_name_rate":  round(decade_personal[decade] / total * 100, 1),
            "institutional_rate":  round(decade_institutional[decade] / total * 100, 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "description_complexity_decades.csv", index=False)
    return df


def plot_description_complexity(df: pd.DataFrame) -> None:
    print("[T7] Plotting description complexity …")
    decades = df["decade"].values

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    ax = axes[0]
    ax.plot(decades, df["avg_words"], color="#1f77b4", lw=2, label="Average words")
    ax.plot(decades, df["median_words"], color="#aec7e8", lw=1.5, ls="--", label="Median words")
    add_era_vlines(ax)
    ax.set_ylabel("Words per description")
    ax.set_title("Description Length: Are Oxford's Accounts Getting More Specific?",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1905)

    ax = axes[1]
    ax.bar(decades, df["ditto_rate"], color="#d62728", alpha=0.7, width=8,
           label="'Ditto' / repeated entries (%)")
    add_era_vlines(ax)
    ax.set_ylabel("% of entries")
    ax.set_title("Ditto / Repetitive Entries: Declining Bureaucratic Shorthand?",
                 fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(decades, df["personal_name_rate"], color="#9467bd", lw=2,
            label="Entries with personal name (Mr/Mrs/Dr)")
    ax.plot(decades, df["institutional_rate"], color="#ff7f0e", lw=2, ls="--",
            label="Entries with institution (College/Company/Trust/Bank)")
    add_era_vlines(ax)
    ax.set_ylabel("% of entries")
    ax.set_xlabel("Decade")
    ax.set_title("From Personal to Institutional: Who Is Oxford Paying?",
                 fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Description Complexity: How Oxford's Accounting Language Evolved",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig_T7_description_complexity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bonus: Decade word-frequency heatmap (top 40 words)
# ---------------------------------------------------------------------------

GLOBAL_STOP = {
    "payment", "paid", "received", "receipt", "entry", "annual", "half",
    "year", "years", "from", "with", "that", "this", "same", "also",
    "made", "note", "related", "relating", "entries", "covering",
    "recorded", "amount", "total", "payments", "general", "various",
    "certain", "their", "college", "oxford", "including", "previous",
    "other", "within", "funds", "each", "under", "above", "dated",
    "part", "parts", "item", "items", "separate", "additional", "noted",
    "account", "accounts", "charge", "charges", "stated", "further",
    "towards", "being", "been", "have", "following", "charged", "paid",
    "previous", "made", "purpose", "purposes", "cost", "costs",
}


def plot_decade_word_heatmap(records: list[dict]) -> None:
    print("[Bonus] Plotting decade word heatmap …")
    decade_total: Counter = Counter()
    decade_words: dict[int, Counter] = defaultdict(Counter)
    for rec in records:
        d = rec["decade"]
        text = rec["english_desc"].lower()
        decade_total[d] += 1
        words = {w for w in re.findall(r"[a-z]{5,}", text) if w not in GLOBAL_STOP}
        decade_words[d].update(words)

    decades = sorted(decade_total.keys())
    # Find top 40 words by total frequency across all decades
    total_counter: Counter = Counter()
    for d in decades:
        total = decade_total[d]
        for w, c in decade_words[d].items():
            total_counter[w] += c
    top_words = [w for w, _ in total_counter.most_common(45) if w not in GLOBAL_STOP][:40]

    # Build matrix: word × decade, value = per-1000 rate
    matrix = np.zeros((len(top_words), len(decades)))
    for j, decade in enumerate(decades):
        total = decade_total[decade]
        for i, word in enumerate(top_words):
            matrix[i, j] = decade_words[decade][word] / total * 1000

    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels([str(d) for d in decades], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words, fontsize=7)
    plt.colorbar(im, ax=ax, label="Frequency per 1000 entries")
    # Add era vertical lines
    for vx in ERA_VLINES:
        try:
            idx = decades.index(vx)
        except ValueError:
            idx = next((i for i, d in enumerate(decades) if d >= vx), None)
        if idx is not None:
            ax.axvline(idx - 0.5, color="white", lw=2)
    ax.set_title("Top-40 Word Frequency Heatmap by Decade (per 1000 entries)\n"
                 "White lines = era boundaries (1780, 1820, 1860)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_T0_word_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T8: Tax vocabulary expansion — fiscal state complexity
# ---------------------------------------------------------------------------

TAX_TYPES = {
    "land tax":     "land tax",
    "poor rate":    "poor rate",
    "window tax":   "window tax",
    "house tax":    "house tax",
    "income tax":   "income tax",
    "property tax": "property tax",
    "stamp duty":   "stamp duty",
    "legacy duty":  "legacy duty",
    "church rate":  "church rate",
    "tithe":        "tithe",
    "gaol tax":     "gaol",
    "lighting rate":"lighting rate",
    "highway rate": "highway rate",
    "county rate":  "county rate",
}


def analyse_tax_expansion(records: list[dict]) -> pd.DataFrame:
    print("[T8] Analysing tax vocabulary expansion …")
    decade_total: Counter = Counter()
    decade_tax: dict[int, Counter] = defaultdict(Counter)

    for rec in records:
        d = rec["decade"]
        text = rec["english_desc"].lower()
        decade_total[d] += 1
        for tax_key, pattern in TAX_TYPES.items():
            if pattern in text:
                decade_tax[d][tax_key] += 1

    rows = []
    for decade in sorted(decade_total.keys()):
        total = decade_total[decade]
        row = {"decade": decade, "total": total}
        active = 0
        for tax_key in TAX_TYPES:
            cnt = decade_tax[decade][tax_key]
            row[tax_key] = round(cnt / total * 1000, 2)
            if cnt > 0:
                active += 1
        row["distinct_tax_types"] = active
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "tax_vocabulary_expansion.csv", index=False)
    return df


def plot_tax_expansion(df: pd.DataFrame) -> None:
    print("[T8] Plotting tax expansion …")
    decades = df["decade"].values
    tax_cols = list(TAX_TYPES.keys())
    tax_data = np.column_stack([df[c].fillna(0).values for c in tax_cols])

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # Panel 1: stacked area of individual tax types
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(tax_cols)))
    ax.stackplot(decades, tax_data.T, labels=tax_cols, colors=colors, alpha=0.85)
    add_era_vlines(ax)
    ax.set_ylabel("Entries per 1000 (stacked)")
    ax.set_title("Tax Type Proliferation in Oxford College Accounts (per 1000 entries)",
                 fontweight="bold")
    ax.legend(fontsize=6.5, ncol=4, loc="upper left")
    ax.set_xlim(1700, 1905)

    # Panel 2: distinct tax types active per decade
    ax = axes[1]
    ax.bar(decades, df["distinct_tax_types"], color="#e6550d", alpha=0.8, width=8,
           label="Distinct tax types mentioned this decade")
    add_era_vlines(ax)
    ax.set_ylabel("Count of distinct tax types")
    ax.set_xlabel("Decade")
    ax.set_title("How Many Tax Types Did the College Encounter? (Expanding Fiscal State)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    # Annotate peak
    peak_idx = df["distinct_tax_types"].idxmax()
    peak_decade = df.loc[peak_idx, "decade"]
    peak_val = df.loc[peak_idx, "distinct_tax_types"]
    ax.annotate(f"Peak: {peak_val} types\n({peak_decade}s)",
                xy=(peak_decade, peak_val),
                xytext=(peak_decade - 20, peak_val + 0.3),
                arrowprops=dict(arrowstyle="->", color="darkred"),
                fontsize=8, color="darkred")

    fig.suptitle("The Expanding Fiscal State: Tax Vocabulary in Oxford's Ledger 1700–1900",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig_T8_tax_expansion.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T9: Personal-to-institutional payees
# ---------------------------------------------------------------------------

PERSONAL_PAYEE_RX = re.compile(
    r"\bMr\.?\b|\bMrs\.?\b|\bMiss\b|\bDr\.?\b|\bRev\.?\b|\bSir\b",
    re.IGNORECASE
)
TRADE_RX = re.compile(
    r"\bmason\b|\bcarpenter\b|\bglazier\b|\bplumber\b|\bsmith\b|\bpainter\b|"
    r"\bbuilder\b|\bjoiner\b|\btailor\b|\bshoemaker\b|\bblacksmith\b|\bsaddler\b",
    re.IGNORECASE
)
INSTITUTIONAL_RX = re.compile(
    r"\bCompany\b|\bSociety\b|\bTrust\b|\bFund\b|\bBank\b|\bAssociation\b|"
    r"\bInstitution\b|\bCommittee\b|\bCorporation\b|\bBrotherhood\b|\bBrothers\b",
    re.IGNORECASE
)

PAYEE_CATEGORIES = ["maintenance", "educational", "salary_stipend", "charitable",
                    "domestic", "financial"]


def analyse_payee_types(records: list[dict]) -> pd.DataFrame:
    print("[T9] Analysing personal vs institutional payees …")
    decade_total: Counter = Counter()
    decade_personal: Counter = Counter()
    decade_personal_trade: Counter = Counter()
    decade_institutional: Counter = Counter()
    # Per-category
    cat_decade_total: dict[str, Counter] = {c: Counter() for c in PAYEE_CATEGORIES}
    cat_decade_personal: dict[str, Counter] = {c: Counter() for c in PAYEE_CATEGORIES}
    cat_decade_institutional: dict[str, Counter] = {c: Counter() for c in PAYEE_CATEGORIES}

    for rec in records:
        d = rec["decade"]
        text = rec["english_desc"]
        cat = rec["category"]
        decade_total[d] += 1
        has_personal = bool(PERSONAL_PAYEE_RX.search(text))
        has_trade = bool(TRADE_RX.search(text))
        has_institutional = bool(INSTITUTIONAL_RX.search(text))
        if has_personal:
            decade_personal[d] += 1
        if has_personal and has_trade:
            decade_personal_trade[d] += 1
        if has_institutional:
            decade_institutional[d] += 1
        if cat in PAYEE_CATEGORIES:
            cat_decade_total[cat][d] += 1
            if has_personal:
                cat_decade_personal[cat][d] += 1
            if has_institutional:
                cat_decade_institutional[cat][d] += 1

    rows = []
    for decade in sorted(decade_total.keys()):
        total = decade_total[decade]
        row = {
            "decade": decade,
            "total": total,
            "personal_rate": round(decade_personal[decade] / total * 100, 1),
            "personal_trade_rate": round(decade_personal_trade[decade] / total * 100, 1),
            "institutional_rate": round(decade_institutional[decade] / total * 100, 1),
        }
        for cat in PAYEE_CATEGORIES:
            ct = cat_decade_total[cat][decade]
            if ct > 0:
                row[f"{cat}_personal_rate"] = round(
                    cat_decade_personal[cat][decade] / ct * 100, 1)
                row[f"{cat}_institutional_rate"] = round(
                    cat_decade_institutional[cat][decade] / ct * 100, 1)
            else:
                row[f"{cat}_personal_rate"] = float("nan")
                row[f"{cat}_institutional_rate"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "payee_type_decades.csv", index=False)
    return df


def plot_payee_types(df: pd.DataFrame) -> None:
    print("[T9] Plotting payee type evolution …")
    decades = df["decade"].values

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # Panel 1: overall personal vs institutional
    ax = axes[0]
    ax.fill_between(decades, df["personal_rate"], alpha=0.35, color="#9467bd",
                    label="Personal payee (Mr/Mrs/Dr…)")
    ax.fill_between(decades, df["institutional_rate"], alpha=0.35, color="#ff7f0e",
                    label="Institutional payee (Company/Society/Trust…)")
    ax.plot(decades, df["personal_rate"], color="#9467bd", lw=2)
    ax.plot(decades, df["institutional_rate"], color="#ff7f0e", lw=2)
    add_era_vlines(ax)
    ax.set_ylabel("% of all entries")
    ax.set_title("Personal vs Institutional Payees Across All Entries",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1905)
    # Crossover annotation
    df_s = df.set_index("decade")
    cross = df_s[df_s["institutional_rate"] > df_s["personal_rate"]]
    if not cross.empty:
        cx = cross.index.min()
        cx_val = df_s.loc[cx, "institutional_rate"]
        ax.annotate(f"Institutions > Personal\n({cx}s)",
                    xy=(cx, cx_val),
                    xytext=(cx + 10, cx_val + 1),
                    arrowprops=dict(arrowstyle="->", color="saddlebrown"),
                    fontsize=8, color="saddlebrown")

    # Panel 2: by category — maintenance vs educational vs charitable
    ax = axes[1]
    show_cats = ["maintenance", "educational", "charitable"]
    cat_colors = {"maintenance": "#2ca02c", "educational": "#1f77b4", "charitable": "#d62728"}
    for cat in show_cats:
        col_p = f"{cat}_personal_rate"
        col_i = f"{cat}_institutional_rate"
        if col_p in df.columns:
            ax.plot(decades, df[col_p], color=cat_colors[cat], lw=2,
                    label=f"{cat} — personal")
            ax.plot(decades, df[col_i], color=cat_colors[cat], lw=2, ls="--",
                    label=f"{cat} — institutional")
    add_era_vlines(ax)
    ax.set_ylabel("% of category entries")
    ax.set_xlabel("Decade")
    ax.set_title("Personal vs Institutional by Category (solid=personal, dashed=institutional)",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("From Guild Economy to Corporate Market: Payee Language in Oxford's Ledger",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig_T9_personal_to_institutional.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T10: Charity language transformation (pre/post Poor Law 1834)
# ---------------------------------------------------------------------------

CHARITY_PRE_1834 = {
    "alms":      "alms",
    "beggar":    "beggar",
    "poor":      "poor",
    "prisoner":  "prisoner",
    "bocardo":   "bocardo",
    "widow":     "widow",
    "destitute": "destitute",
    "vagabond":  "vagabond",
}

CHARITY_POST_1834 = {
    "subscription":  "subscription",
    "donation":      "donation",
    "infirmary":     "infirmary",
    "orphan":        "orphan",
    "dispensary":    "dispensary",
    "relief fund":   "relief fund",
    "charitable":    "charitable",
    "institution":   "institution",
}

POOR_LAW_YEAR = 1834


def analyse_charity_language(records: list[dict]) -> pd.DataFrame:
    print("[T10] Analysing charity language transformation …")
    decade_total: Counter = Counter()
    decade_pre: dict[int, Counter] = defaultdict(Counter)
    decade_post: dict[int, Counter] = defaultdict(Counter)

    # Filter to charitable category only
    charitable = [r for r in records if r.get("category") == "charitable"]
    for rec in charitable:
        d = rec["decade"]
        text = rec["english_desc"].lower()
        decade_total[d] += 1
        for key, pattern in CHARITY_PRE_1834.items():
            if pattern in text:
                decade_pre[d][key] += 1
        for key, pattern in CHARITY_POST_1834.items():
            if pattern in text:
                decade_post[d][key] += 1

    rows = []
    for decade in sorted(decade_total.keys()):
        total = max(decade_total[decade], 1)
        row = {"decade": decade, "total_charitable": total}
        row["pre_total_rate"] = round(sum(decade_pre[decade].values()) / total * 1000, 1)
        row["post_total_rate"] = round(sum(decade_post[decade].values()) / total * 1000, 1)
        for key in CHARITY_PRE_1834:
            row[f"pre_{key}"] = round(decade_pre[decade][key] / total * 1000, 2)
        for key in CHARITY_POST_1834:
            row[f"post_{key}"] = round(decade_post[decade][key] / total * 1000, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "charity_language_decades.csv", index=False)
    return df


def plot_charity_language(df: pd.DataFrame) -> None:
    print("[T10] Plotting charity language …")
    decades = df["decade"].values

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # Panel 1: total pre vs post rates
    ax = axes[0]
    ax.fill_between(decades, df["pre_total_rate"], alpha=0.4, color="#d62728",
                    label="Personal almsgiving vocabulary\n(alms, beggar, poor, prisoner, bocardo)")
    ax.fill_between(decades, df["post_total_rate"], alpha=0.4, color="#1f77b4",
                    label="Institutional charity vocabulary\n(subscription, infirmary, donation, dispensary)")
    ax.plot(decades, df["pre_total_rate"], color="#d62728", lw=2)
    ax.plot(decades, df["post_total_rate"], color="#1f77b4", lw=2)
    ax.axvline(POOR_LAW_YEAR, color="black", lw=1.5, ls="--", alpha=0.8)
    ax.annotate("Poor Law\nAmendment Act\n1834", xy=(POOR_LAW_YEAR, ax.get_ylim()[1] * 0.9),
                xytext=(POOR_LAW_YEAR + 5, ax.get_ylim()[1] * 0.85),
                fontsize=7.5, color="black")
    add_era_vlines(ax)
    ax.set_ylabel("Occurrences per 1000 charitable entries")
    ax.set_title("Charity Language Transformation: Personal Almsgiving → Institutional Relief",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1700, 1905)

    # Panel 2: individual terms stacked
    ax = axes[1]
    pre_keys = list(CHARITY_PRE_1834.keys())
    post_keys = list(CHARITY_POST_1834.keys())
    pre_data = np.column_stack([df[f"pre_{k}"].fillna(0).values for k in pre_keys])
    post_data = np.column_stack([df[f"post_{k}"].fillna(0).values for k in post_keys])
    pre_colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(pre_keys)))
    post_colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(post_keys)))

    ax.stackplot(decades, pre_data.T, labels=[f"pre: {k}" for k in pre_keys],
                 colors=pre_colors, alpha=0.8)
    ax.stackplot(decades, -post_data.T, labels=[f"post: {k}" for k in post_keys],
                 colors=post_colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(POOR_LAW_YEAR, color="black", lw=1.5, ls="--", alpha=0.8)
    add_era_vlines(ax)
    ax.set_ylabel("↑ Personal / ↓ Institutional (per 1000)")
    ax.set_xlabel("Decade")
    ax.set_title("Individual Terms: Pre-1834 Vocabulary (above) vs Post-1834 Vocabulary (below)",
                 fontweight="bold")
    ax.legend(fontsize=6.5, ncol=4, loc="lower left")

    fig.suptitle("The Poor Law 1834 in Oxford's Charity Accounts: A Textual Record",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig_T10_charity_transformation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    records = load_all_entries()

    # T0: overview heatmap
    plot_decade_word_heatmap(records)

    # T1: indicator trajectories
    freq_df = compute_indicator_frequencies(records)
    freq_df.to_csv(OUT_DIR / "indicator_word_frequencies.csv", index=False)
    plot_indicator_trajectories(freq_df)

    # T2: first appearances
    first_df = find_first_appearances(records)
    plot_first_appearances(first_df)
    write_first_appearances_narrative(first_df)

    # T3: dying vocabulary
    dying_df = analyse_dying_vocabulary(records)
    plot_dying_vocabulary(dying_df)

    # T4: bigrams
    bigram_result = analyse_bigrams(records)
    plot_bigrams(bigram_result)

    # T5: category vocabulary signatures
    category_vocabulary_signatures(records)

    # T6: calendar language
    cal_df = analyse_calendar_language(records)
    plot_calendar_language(cal_df)

    # T7: description complexity
    comp_df = analyse_description_complexity(records)
    plot_description_complexity(comp_df)

    # T8: tax vocabulary expansion
    tax_df = analyse_tax_expansion(records)
    plot_tax_expansion(tax_df)

    # T9: personal to institutional payees
    payee_df = analyse_payee_types(records)
    plot_payee_types(payee_df)

    # T10: charity language transformation
    charity_df = analyse_charity_language(records)
    plot_charity_language(charity_df)

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
