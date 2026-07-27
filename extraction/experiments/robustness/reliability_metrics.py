"""
experiments/robustness/reliability_metrics.py

Quantifies how consistent the LLM-assigned labels are across three independent
model runs on the same ~100-page sample:
  - original     : gpt-5-mini (the full production run)
  - haiku45      : Anthropic Claude Haiku 4.5 (different provider / architecture)
  - gemini25flash: Google Gemini 2.5 Flash (third independent provider / architecture)

Metrics computed
----------------
1. Cohen's κ (pairwise)
   Standard pairwise inter-annotator agreement for each pair of raters on each
   categorical field.  Widely used in NLP annotation studies (Gilardi et al. 2023;
   Møller et al. 2023).  Landis & Koch (1977) benchmarks: >0.8 almost-perfect,
   0.6–0.8 substantial, 0.4–0.6 moderate.

2. Fleiss' κ (multi-rater)
   Generalises Cohen's κ to three or more raters simultaneously.  One number per
   field summarises the overall cross-model reliability.

3. % exact agreement
   Proportion of entry rows where all three models assign the exact same label.
   Simple and intuitive complement to κ.

4. Bootstrap 95% confidence intervals on aggregate statistics
   For each model run, resample pages with replacement (B=1000 bootstrap draws),
   recompute key aggregate statistics per draw, and report the 2.5–97.5th
   percentile band.  Overlapping CIs across models means the downstream analysis
   conclusions are stable regardless of which model is used.
   Statistics bootstrapped:
     - income share (entry count) by decade
     - land_rent category share by era
     - financial category share by era
     - educational direction share by era (% income)
     - english language share by decade

Outputs
-------
experiments/reports/robustness/reliability/
  kappa_pairwise.csv          — Cohen's κ per (field, model-pair)
  kappa_fleiss.csv            — Fleiss' κ per field
  agreement_rate.csv          — % exact match per field, plus per-class breakdown
  bootstrap_income_share.csv  — decade × model bootstrap CIs
  bootstrap_category_share.csv— era × category × model bootstrap CIs
  bootstrap_language_share.csv— decade × language × model bootstrap CIs
  reliability_summary.txt     — human-readable summary

Usage
-----
    .venv/bin/python -m experiments.robustness.reliability_metrics
    .venv/bin/python -m experiments.robustness.reliability_metrics --n-bootstrap 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ENRICHED_DIR   = ROOT / "experiments" / "results" / "enriched"
ROBUSTNESS_DIR = ROOT / "experiments" / "results" / "robustness"
SAMPLE_FILE    = ROBUSTNESS_DIR / "sample_pages.json"
OUT_DIR        = ROOT / "experiments" / "reports" / "robustness" / "reliability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Phelps Brown-Hopkins price index (same as analysis_enriched.py)
# ---------------------------------------------------------------------------
_PBH: dict[int, float] = {
    1700: 100.0, 1710: 103.7, 1720: 101.3, 1730:  93.6,
    1740: 100.0, 1750: 104.7, 1760: 115.7, 1770: 125.4,
    1780: 138.3, 1790: 145.2, 1800: 203.4, 1810: 269.0,
    1820: 213.7, 1830: 175.4, 1840: 170.3, 1850: 161.2,
    1860: 175.4, 1870: 193.0, 1880: 182.4, 1890: 160.0,
    1900: 169.7,
}
_PBH_YRS = sorted(_PBH); _PBH_VALS = [_PBH[y] for y in _PBH_YRS]


def price_index(year: int) -> float:
    if year <= _PBH_YRS[0]:  return _PBH_VALS[0]
    if year >= _PBH_YRS[-1]: return _PBH_VALS[-1]
    return float(np.interp(year, _PBH_YRS, _PBH_VALS))


def era_of_year(year: int) -> str:
    if year < 1760:            return "pre_1760"
    if 1760 <= year <= 1840:   return "industrial_1760_1840"
    return "post_1840"


def decade_of_year(year: int) -> str:
    d = (year // 10) * 10
    return f"{d}s"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_money(v: Any) -> float:
    if v is None or v == "": return 0.0
    if isinstance(v, (int, float)): return 0.0 if np.isnan(float(v)) else float(v)
    s = re.sub(r"[^0-9.\-]", "", str(v).strip())
    try: return float(s)
    except ValueError: return 0.0


def parse_fraction(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)): return 0.0
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().lower()
    for frac, val in [("¼","0.25"),("1/4","0.25"),("½","0.5"),("1/2","0.5"),
                      ("¾","0.75"),("3/4","0.75")]:
        if s == frac: return float(val)
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
# Data loading
# ---------------------------------------------------------------------------
CATEGORICAL_FIELDS = ["direction", "category", "language", "payment_period"]
BOOLEAN_FIELDS     = ["is_arrears", "is_signature"]


def load_enriched_file(path: Path) -> list[dict]:
    """Load entry rows from one enriched JSON, returning flat dicts with year info."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    page_id = payload.get("page_id") or path.stem.replace("_enriched", "")
    try:
        years, _ = parse_page_id(page_id)
    except ValueError:
        return []

    year_weight = 1.0 / len(years)
    records = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict): continue
        if row.get("row_type") != "entry": continue
        for year in years:
            records.append({
                "page_id":    page_id,
                "row_index":  row.get("row_index"),
                "year":       year,
                "year_weight": year_weight,
                "amount":     amount_decimal(row),
                **{f: row.get(f) for f in CATEGORICAL_FIELDS + BOOLEAN_FIELDS},
            })
    return records


def load_run(run_dir: Path, page_ids: list[str]) -> pd.DataFrame:
    """Load enriched data for the given page_ids from a directory."""
    records = []
    missing = []
    for pid in page_ids:
        fp = run_dir / f"{pid}_enriched.json"
        if not fp.exists():
            missing.append(pid)
            continue
        records.extend(load_enriched_file(fp))
    if missing:
        print(f"  [warn] {run_dir.name}: {len(missing)} pages missing: {missing[:3]}{'...' if len(missing)>3 else ''}")
    df = pd.DataFrame(records)
    if not df.empty:
        df["era"]    = df["year"].map(era_of_year)
        df["decade"] = df["year"].map(decade_of_year)
        df["amount_weighted"] = df["amount"] * df["year_weight"]
    return df


# ---------------------------------------------------------------------------
# Kappa helpers
# ---------------------------------------------------------------------------
def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's κ, treating None/NaN as a separate category 'unknown'."""
    a = [x if x is not None and (not isinstance(x, float) or not np.isnan(x)) else "unknown"
         for x in labels_a]
    b = [x if x is not None and (not isinstance(x, float) or not np.isnan(x)) else "unknown"
         for x in labels_b]
    try:
        return float(cohen_kappa_score(a, b))
    except Exception:
        return float("nan")


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' κ from a ratings matrix of shape (N_items, N_categories).
    ratings_matrix[i, j] = number of raters who assigned category j to item i.
    n = total number of raters per item (assumed constant).
    """
    N, k = ratings_matrix.shape
    n = int(ratings_matrix[0].sum())

    if n < 2:
        return float("nan")

    # P_i = proportion of agreeing pairs for item i
    P_i = (1.0 / (n * (n - 1))) * (
        np.sum(ratings_matrix ** 2, axis=1) - n
    )
    P_bar = float(np.mean(P_i))

    # p_j = marginal proportion for category j
    p_j = ratings_matrix.sum(axis=0) / (N * n)
    P_e_bar = float(np.sum(p_j ** 2))

    if abs(1.0 - P_e_bar) < 1e-10:
        return float("nan")

    return (P_bar - P_e_bar) / (1.0 - P_e_bar)


def build_ratings_matrix(
    labels_by_rater: list[list],
    categories: list[str],
) -> np.ndarray:
    """
    Build a (N_items, N_categories) ratings matrix for Fleiss' κ.
    labels_by_rater[r][i] = label assigned by rater r to item i.
    Items where any rater assigned None are excluded.
    """
    n_items = len(labels_by_rater[0])
    cat_idx = {c: i for i, c in enumerate(categories)}
    matrix_rows = []
    for i in range(n_items):
        row_labels = [labels_by_rater[r][i] for r in range(len(labels_by_rater))]
        # Replace None with "unknown"
        row_labels = [
            x if x is not None and (not isinstance(x, float) or not np.isnan(x)) else "unknown"
            for x in row_labels
        ]
        counts = np.zeros(len(categories), dtype=float)
        for lab in row_labels:
            idx = cat_idx.get(lab)
            if idx is not None:
                counts[idx] += 1
        matrix_rows.append(counts)
    return np.array(matrix_rows)


# ---------------------------------------------------------------------------
# Agreement rate per field
# ---------------------------------------------------------------------------
def agreement_rate(
    labels_by_rater: list[list],
) -> tuple[float, pd.DataFrame]:
    """
    Returns overall % exact agreement and a per-class breakdown.
    """
    n_items = len(labels_by_rater[0])
    n_agree = 0
    for i in range(n_items):
        vals = {
            x if x is not None and (not isinstance(x, float) or not np.isnan(x)) else "unknown"
            for x in [labels_by_rater[r][i] for r in range(len(labels_by_rater))]
        }
        if len(vals) == 1:
            n_agree += 1

    overall = n_agree / n_items if n_items > 0 else float("nan")

    # Per-class: among items where original=class_c, what fraction agree across all?
    class_rows = []
    orig = [
        x if x is not None and (not isinstance(x, float) or not np.isnan(x)) else "unknown"
        for x in labels_by_rater[0]
    ]
    for cls in sorted(set(orig)):
        idxs = [i for i, v in enumerate(orig) if v == cls]
        n_cls = len(idxs)
        n_cls_agree = sum(
            1 for i in idxs
            if len({
                x if x is not None and (not isinstance(x, float) or not np.isnan(x)) else "unknown"
                for x in [labels_by_rater[r][i] for r in range(len(labels_by_rater))]
            }) == 1
        )
        class_rows.append({
            "label": cls,
            "n_items": n_cls,
            "n_agree": n_cls_agree,
            "agree_rate": n_cls_agree / n_cls if n_cls > 0 else float("nan"),
        })

    return overall, pd.DataFrame(class_rows)


# ---------------------------------------------------------------------------
# Bootstrap aggregate statistics
# ---------------------------------------------------------------------------
def compute_aggregate_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute key aggregate statistics from a flat enriched DataFrame."""
    stats = {}

    # 1. Income share by decade (entry count)
    dir_df = df[df["direction"].isin(["income", "expenditure"])].copy()
    if not dir_df.empty:
        grp = dir_df.groupby(["decade", "direction"])["year_weight"].sum().unstack(fill_value=0)
        for col in ["income", "expenditure"]:
            if col not in grp.columns: grp[col] = 0.0
        grp["income_share"] = grp["income"] / (grp["income"] + grp["expenditure"]).replace(0, np.nan)
        stats["income_share_by_decade"] = grp[["income_share"]].reset_index()

    # 2. Land-rent category share by era
    cat_df = df[df["category"].notna()].copy()
    if not cat_df.empty:
        era_total = cat_df.groupby("era")["year_weight"].sum()
        era_land  = cat_df[cat_df["category"] == "land_rent"].groupby("era")["year_weight"].sum()
        land_share = (era_land / era_total).rename("land_rent_share").reset_index()
        stats["land_rent_share_by_era"] = land_share

        # 2b. Financial category share by era
        era_fin = cat_df[cat_df["category"] == "financial"].groupby("era")["year_weight"].sum()
        fin_share = (era_fin / era_total).rename("financial_share").reset_index()
        stats["financial_share_by_era"] = fin_share

    # 3. English language share by decade
    lang_df = df[df["language"].isin(["latin", "english", "mixed"])].copy()
    if not lang_df.empty:
        grp = lang_df.groupby(["decade", "language"])["year_weight"].sum().unstack(fill_value=0)
        total = grp.sum(axis=1)
        for lang in ["english", "latin", "mixed"]:
            if lang not in grp.columns: grp[lang] = 0.0
        grp["english_share"] = grp["english"] / total.replace(0, np.nan)
        stats["english_share_by_decade"] = grp[["english_share"]].reset_index()

    # 4. Educational direction share by era (% income)
    edu_df = df[
        (df["category"] == "educational") &
        (df["direction"].isin(["income", "expenditure"]))
    ].copy()
    if not edu_df.empty:
        grp = edu_df.groupby(["era", "direction"])["year_weight"].sum().unstack(fill_value=0)
        for col in ["income", "expenditure"]:
            if col not in grp.columns: grp[col] = 0.0
        grp["edu_income_share"] = grp["income"] / (grp["income"] + grp["expenditure"]).replace(0, np.nan)
        stats["educational_direction_by_era"] = grp[["edu_income_share"]].reset_index()

    return stats


def bootstrap_ci(
    df: pd.DataFrame,
    page_ids: list[str],
    stat_fn,
    stat_key: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Resample pages with replacement and compute a statistic's distribution.
    Returns a DataFrame with columns matching stat_fn output + lower/upper CI columns.
    """
    rng = np.random.default_rng(seed)
    pages = np.array(page_ids)
    bootstrap_results: list[pd.DataFrame] = []

    for _ in range(n_bootstrap):
        sampled_pages = rng.choice(pages, size=len(pages), replace=True)
        boot_df = pd.concat(
            [df[df["page_id"].str.split("_").apply(lambda p: "_".join(p[:2])) == pid.rsplit("_image", 1)[0]]
             if False else  # can't do exact page_id match easily; use direct filter
             df[df["page_id"] == pid]
             for pid in sampled_pages],
            ignore_index=True,
        )
        stats = stat_fn(boot_df)
        if stat_key in stats:
            bootstrap_results.append(stats[stat_key])

    if not bootstrap_results:
        return pd.DataFrame()

    # Get reference (non-bootstrap) stat
    ref_stats = stat_fn(df)
    if stat_key not in ref_stats:
        return pd.DataFrame()
    ref = ref_stats[stat_key]

    # For each row in ref, collect all bootstrap estimates and compute CI
    _known_stat_cols = {"income_share", "land_rent_share", "english_share",
                        "financial_share", "edu_income_share"}
    index_cols = [c for c in ref.columns if c not in _known_stat_cols]
    stat_col   = [c for c in ref.columns if c not in index_cols][0]

    # Stack bootstrap estimates
    boot_stack = pd.concat(bootstrap_results, ignore_index=True)
    merged = ref.copy()
    merged["estimate"] = merged[stat_col]

    ci_rows = []
    for _, row in ref.iterrows():
        mask = pd.Series([True] * len(boot_stack))
        for ic in index_cols:
            mask &= boot_stack[ic] == row[ic]
        vals = boot_stack.loc[mask, stat_col].dropna()
        ci_rows.append({
            **{ic: row[ic] for ic in index_cols},
            "estimate":  float(row[stat_col]) if not pd.isna(row[stat_col]) else float("nan"),
            "ci_lower":  float(np.percentile(vals, 2.5))  if len(vals) >= 10 else float("nan"),
            "ci_upper":  float(np.percentile(vals, 97.5)) if len(vals) >= 10 else float("nan"),
        })

    return pd.DataFrame(ci_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute inter-model reliability metrics")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap iterations for CI estimation (default: 1000)")
    args = parser.parse_args()

    if not SAMPLE_FILE.exists():
        sys.exit(f"Sample file not found: {SAMPLE_FILE}\nRun sample_pages.py first.")

    with open(SAMPLE_FILE, encoding="utf-8") as fh:
        sample = json.load(fh)
    page_ids: list[str] = sample["page_ids"]

    # --- Load three runs ---
    runs: dict[str, pd.DataFrame] = {}

    print("Loading original run (gpt-5-mini)...")
    runs["original"] = load_run(ENRICHED_DIR, page_ids)

    for model_name in ["haiku45", "gemini25flash"]:
        model_dir = ROBUSTNESS_DIR / model_name
        if not model_dir.exists():
            print(f"[warn] {model_name} output not found at {model_dir} — skipping")
            continue
        print(f"Loading {model_name}...")
        runs[model_name] = load_run(model_dir, page_ids)

    if len(runs) < 2:
        sys.exit("Need at least 2 model runs to compute reliability. Run rerun_enrichment.py first.")

    run_names = list(runs.keys())
    print(f"\nRuns available: {run_names}")
    for name, df in runs.items():
        print(f"  {name}: {len(df):,} entry-rows from {df['page_id'].nunique()} pages")

    # --- Align entries across runs ---
    # Build a multi-index key: (page_id, row_index, year)
    # Only keep rows present in ALL runs
    key_cols = ["page_id", "row_index", "year"]

    aligned: dict[str, pd.DataFrame] = {}
    common_keys = None
    for name, df in runs.items():
        if df.empty:
            continue
        df = df.set_index(key_cols)
        if common_keys is None:
            common_keys = df.index
        else:
            common_keys = common_keys.intersection(df.index)
        aligned[name] = df

    if common_keys is None or len(common_keys) == 0:
        sys.exit("No overlapping entries found across runs.")

    print(f"\nCommon entries across all runs: {len(common_keys):,}")

    for name in aligned:
        aligned[name] = aligned[name].loc[common_keys].reset_index()

    # --- 1. Cohen's κ (pairwise) ---
    print("\nComputing Cohen's κ (pairwise)...")
    kappa_rows = []
    all_fields = CATEGORICAL_FIELDS  # booleans don't need kappa

    for field in all_fields:
        for (n1, df1), (n2, df2) in combinations(aligned.items(), 2):
            lab1 = df1[field].tolist()
            lab2 = df2[field].tolist()
            kappa = cohen_kappa(lab1, lab2)
            kappa_rows.append({
                "field":   field,
                "rater_a": n1,
                "rater_b": n2,
                "cohen_kappa": round(kappa, 4),
                "n_items": len(lab1),
            })

    kappa_df = pd.DataFrame(kappa_rows)
    kappa_df.to_csv(OUT_DIR / "kappa_pairwise.csv", index=False)
    print(kappa_df.to_string(index=False))

    # --- 2. Fleiss' κ (all models) ---
    print("\nComputing Fleiss' κ (all raters)...")
    fleiss_rows = []
    for field in all_fields:
        # Determine category set (union across all runs)
        all_vals: set[str] = set()
        for df in aligned.values():
            all_vals |= set(
                x if x is not None and (not isinstance(x, float) or not np.isnan(x)) else "unknown"
                for x in df[field].tolist()
            )
        categories = sorted(all_vals)

        labels_by_rater = [
            aligned[name][field].tolist() for name in run_names if name in aligned
        ]
        if len(labels_by_rater) < 2:
            continue

        matrix = build_ratings_matrix(labels_by_rater, categories)
        fk = fleiss_kappa(matrix)
        fleiss_rows.append({
            "field":        field,
            "n_raters":     len(labels_by_rater),
            "n_items":      len(matrix),
            "n_categories": len(categories),
            "fleiss_kappa": round(fk, 4),
        })

    fleiss_df = pd.DataFrame(fleiss_rows)
    fleiss_df.to_csv(OUT_DIR / "kappa_fleiss.csv", index=False)
    print(fleiss_df.to_string(index=False))

    # --- 3. % exact agreement (all fields) ---
    print("\nComputing exact agreement rates...")
    agree_summary_rows = []
    for field in all_fields + BOOLEAN_FIELDS:
        labels_by_rater = [
            aligned[name][field].tolist() for name in run_names if name in aligned
        ]
        if len(labels_by_rater) < 2:
            continue
        overall, per_class = agreement_rate(labels_by_rater)
        agree_summary_rows.append({"field": field, "exact_agree_rate": round(overall, 4),
                                    "n_items": len(labels_by_rater[0])})
        per_class["field"] = field
        per_class.to_csv(OUT_DIR / f"agreement_per_class_{field}.csv", index=False)

    agree_df = pd.DataFrame(agree_summary_rows)
    agree_df.to_csv(OUT_DIR / "agreement_rate.csv", index=False)
    print(agree_df.to_string(index=False))

    # --- 4. Bootstrap CIs ---
    print(f"\nBootstrap CIs (B={args.n_bootstrap})...")
    for stat_key, label in [
        ("income_share_by_decade",        "income_share"),
        ("land_rent_share_by_era",        "land_rent_share"),
        ("english_share_by_decade",       "english_share"),
        ("financial_share_by_era",        "financial_share"),
        ("educational_direction_by_era",  "edu_income_share"),
    ]:
        ci_frames = []
        for run_name, df in runs.items():
            if df.empty: continue
            ci = bootstrap_ci(
                df, page_ids, compute_aggregate_stats,
                stat_key, n_bootstrap=args.n_bootstrap,
            )
            if ci.empty: continue
            ci["model"] = run_name
            ci_frames.append(ci)

        if ci_frames:
            combined = pd.concat(ci_frames, ignore_index=True)
            combined.to_csv(OUT_DIR / f"bootstrap_{stat_key}.csv", index=False)
            print(f"  {stat_key}: saved {len(combined)} rows")

    # --- 5. Human-readable summary ---
    print("\nWriting reliability_summary.txt...")
    with open(OUT_DIR / "reliability_summary.txt", "w") as fh:
        fh.write("INTER-MODEL RELIABILITY SUMMARY\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Models compared: {run_names}\n")
        fh.write(f"Pages sampled: {len(page_ids)}\n")
        fh.write(f"Common entries: {len(common_keys):,}\n\n")

        fh.write("Interpretation guide (Landis & Koch 1977):\n")
        fh.write("  κ > 0.80 = Almost perfect\n")
        fh.write("  κ 0.60–0.80 = Substantial\n")
        fh.write("  κ 0.40–0.60 = Moderate\n")
        fh.write("  κ < 0.40 = Weak\n\n")

        fh.write("Fleiss' κ (all models):\n")
        if not fleiss_df.empty:
            fh.write(fleiss_df[["field", "fleiss_kappa"]].to_string(index=False))
        fh.write("\n\nCohen's κ (pairwise):\n")
        if not kappa_df.empty:
            fh.write(kappa_df[["field", "rater_a", "rater_b", "cohen_kappa"]].to_string(index=False))
        fh.write("\n\nExact agreement rates:\n")
        if not agree_df.empty:
            fh.write(agree_df.to_string(index=False))
        fh.write("\n")

    print(f"\nAll reliability outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
