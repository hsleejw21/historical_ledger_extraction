from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "experiments" / "reports" / "analysis_v1"
SRC_FILE = SRC_DIR / "all_rows_with_amounts.csv"
OUT_DIR = ROOT / "experiments" / "reports" / "analysis_v1"
EMB_DIR = OUT_DIR / "embeddings"


def start_year_from_sheet(sheet: str) -> int:
    m1 = re.match(r"^(\d{4})_\d+_image$", sheet)
    if m1:
        return int(m1.group(1))
    m2 = re.match(r"^(\d{4})-\d{4}_\d+_image$", sheet)
    if m2:
        return int(m2.group(1))
    return -1


def choose_k(X: np.ndarray, max_k: int = 12) -> tuple[int, float]:
    best_k, best_s = 2, -1.0
    upper = min(max_k, len(X) - 1)
    for k in range(2, upper + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=40)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s


def build_docs(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    d = df[df["row_type"].isin(["entry", "total"])].copy()
    d = d[d["description_norm"].fillna("") != ""]

    if unit == "year":
        g = (
            d.groupby("year", as_index=False)["description_norm"]
            .apply(lambda s: " ".join(s.astype(str).tolist()))
            .rename(columns={"year": "unit_id", "description_norm": "doc"})
            .sort_values("unit_id")
        )
        g["year_ref"] = g["unit_id"].astype(int)
        g["unit"] = "year"
        return g

    # page-level: de-duplicate rows duplicated by year-range expansion
    p = d[["sheet", "row_idx", "description_norm"]].drop_duplicates().copy()
    g = (
        p.groupby("sheet", as_index=False)["description_norm"]
        .apply(lambda s: " ".join(s.astype(str).tolist()))
        .rename(columns={"sheet": "unit_id", "description_norm": "doc"})
    )
    g["year_ref"] = g["unit_id"].map(start_year_from_sheet)
    g["unit"] = "page"
    g = g.sort_values(["year_ref", "unit_id"])
    return g


def cluster_docs(docs_df: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=6000)
    Xs = vec.fit_transform(docs_df["doc"].tolist())
    n_comp = max(2, min(150, Xs.shape[0] - 1, Xs.shape[1] - 1))
    X = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(Xs)
    X = StandardScaler().fit_transform(X)

    k, sil = choose_k(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=60)
    labels = km.fit_predict(X)

    out = docs_df[["unit", "unit_id", "year_ref"]].copy()
    out["cluster"] = labels
    out.to_csv(EMB_DIR / f"{unit}_embedding_clusters.csv", index=False)

    profile = (
        out.groupby("cluster", as_index=False)
        .agg(
            n_units=("unit_id", "count"),
            min_year=("year_ref", "min"),
            max_year=("year_ref", "max"),
        )
        .sort_values("cluster")
    )
    profile.to_csv(EMB_DIR / f"{unit}_cluster_profile.csv", index=False)

    meta = pd.DataFrame(
        [{"unit": unit, "n_units": len(out), "k": k, "silhouette": sil, "embedding": "tfidf_svd"}]
    )
    meta.to_csv(EMB_DIR / f"{unit}_embedding_meta.csv", index=False)

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X)
    plot = out.copy()
    plot["pc1"] = z[:, 0]
    plot["pc2"] = z[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=plot, x="pc1", y="pc2", hue="cluster", palette="tab10", s=52, ax=ax)
    ax.set_title(f"{unit.capitalize()} text embeddings (PCA)")
    fig.tight_layout()
    fig.savefig(EMB_DIR / f"{unit}_embedding_pca.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    sns.scatterplot(data=plot, x="year_ref", y="cluster", hue="cluster", palette="tab10", s=36, ax=ax)
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title(f"{unit.capitalize()} clusters over time")
    ax.set_xlabel("Reference year")
    fig.tight_layout()
    fig.savefig(EMB_DIR / f"{unit}_cluster_timeline.png", dpi=180)
    plt.close(fig)

    return out, meta


def main() -> None:
    if not SRC_FILE.exists():
        raise FileNotFoundError(f"Missing required source file: {SRC_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(SRC_FILE)

    year_docs = build_docs(df, unit="year")
    page_docs = build_docs(df, unit="page")
    year_docs.to_csv(EMB_DIR / "year_docs.csv", index=False)
    page_docs.to_csv(EMB_DIR / "page_docs.csv", index=False)

    _, year_meta = cluster_docs(year_docs, unit="year")
    _, page_meta = cluster_docs(page_docs, unit="page")

    comp = pd.concat([year_meta, page_meta], ignore_index=True)
    comp.to_csv(EMB_DIR / "unit_comparison.csv", index=False)

    # bring key files into clean root for easy navigation
    for file_name in [
        "REANALYSIS_SUMMARY.md",
        "yearly_numeric_summary.csv",
        "yearly_change_points.csv",
        "header_account_first_appearance.csv",
        "header_account_yearly_counts.csv",
        "yearly_amount_and_volume.png",
        "yearly_median_p90.png",
        "change_points_amount_sum.png",
    ]:
        src = SRC_DIR / file_name
        if src.exists():
            (OUT_DIR / file_name).write_bytes(src.read_bytes())

    summary = f"""# Clean analysis package: ledger

## What is included
- Core numeric trend outputs (year-level)
- Header/account evolution outputs
- Embedding clustering comparison by unit:
  - year-level document clustering
  - page-level document clustering

## Embedding setup for comparison
- Embedding: TF-IDF + SVD
- Clustering: KMeans with silhouette-based k selection

## Quick compare
{comp.to_string(index=False)}

## Files to look at first
- embeddings/unit_comparison.csv
- embeddings/year_cluster_timeline.png
- embeddings/page_cluster_timeline.png
- yearly_amount_and_volume.png
- change_points_amount_sum.png
"""
    (OUT_DIR / "COMPARISON_SUMMARY.md").write_text(summary, encoding="utf-8")

    print(f"Done. Clean package created at: {OUT_DIR}")


if __name__ == "__main__":
    main()
