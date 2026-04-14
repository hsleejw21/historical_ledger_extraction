"""embedding_analysis_enriched.py

Embedding-based analysis of enriched ledger entries.

Key improvements over the previous embedding work
(hybrid_year_text_embeddings_ledger.py / compare_unit_text_clustering_ledger.py):

  1. Uses `english_description` (clean, LLM-normalised English) rather than
     raw OCR text.  TF-IDF on standardised descriptions produces much sharper
     vocabulary clusters.

  2. Entry-level clustering — discovers semantic transaction groups directly,
     independent of the labelled categories, and measures how well the unsupervised
     clusters recover those labels (cluster purity / category alignment).

  3. Year-level cosine-similarity matrix — shows which years are semantically
     closest to each other (temporal neighbourhood), not just which cluster they
     fall into.

  4. Category semantic drift — tracks how within-category vocabulary shifts
     across decades, revealing changing institutional language.

  5. Direction vocabulary contrast — log-odds TF-IDF separating income from
     expenditure entries to surface the key distinguishing words.

  6. Sentence-transformer dense embeddings (optional) — automatically used
     if `sentence-transformers` is installed; otherwise falls back to TF-IDF/SVD.

Output directory: experiments/reports/enriched_analysis/embeddings/
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_CSV = ROOT / "experiments" / "reports" / "enriched_analysis" / "all_enriched_entries.csv"
ENRICHED_DIR = ROOT / "experiments" / "results" / "enriched"
OUT_DIR = ROOT / "experiments" / "reports" / "analysis_v3" / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "land_rent", "ecclesiastical", "maintenance", "salary_stipend",
    "administrative", "educational", "financial", "domestic",
    "charitable", "other",
]
ERA_ORDER = ["pre_1760", "industrial_1760_1840", "post_1840"]
ERA_LABELS = {
    "pre_1760": "Pre-1760",
    "industrial_1760_1840": "1760–1840",
    "post_1840": "Post-1840",
}
ERA_VLINES = [1760, 1840]

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def era_of_year(year: int) -> str:
    if year < 1760:
        return "pre_1760"
    if 1760 <= year <= 1840:
        return "industrial_1760_1840"
    return "post_1840"


def _add_era_shading(ax: plt.Axes) -> None:
    ax.axvspan(1760, 1840, alpha=0.06, color="steelblue", zorder=0)
    for yr in ERA_VLINES:
        ax.axvline(yr, color="steelblue", lw=0.8, ls="--", alpha=0.5)


def choose_k(X: np.ndarray, k_min: int = 2, k_max: int = 12) -> tuple[int, float]:
    """Return (best_k, best_silhouette) by exhaustive search."""
    best_k, best_s = k_min, -1.0
    upper = min(k_max, len(X) - 1)
    for k in range(k_min, upper + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=40)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        if s > best_s:
            best_s = s
            best_k = k
    return best_k, best_s


def cluster_purity(labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Unsupervised cluster purity against a ground-truth label array."""
    total = 0
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        if mask.sum() == 0:
            continue
        counts = np.bincount(true_labels[mask])
        total += counts.max()
    return total / len(labels)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the flat enriched entries CSV produced by analysis_enriched.py."""
    if not ENRICHED_CSV.exists():
        raise FileNotFoundError(
            f"Master CSV not found at {ENRICHED_CSV}.\n"
            "Run analysis_enriched.py first."
        )
    df = pd.read_csv(ENRICHED_CSV)
    # Keep only rows with a usable english description
    df = df[df["english_desc"].notna() & (df["english_desc"].str.strip() != "")]
    df["era"] = df["year"].map(era_of_year)
    print(f"Loaded {len(df):,} entry rows with english_desc "
          f"({df['year'].min()}–{df['year'].max()})")
    return df


# ---------------------------------------------------------------------------
# Embedding builders
# ---------------------------------------------------------------------------

def build_tfidf_matrix(
    texts: list[str],
    max_features: int = 8000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """Return (dense_matrix, fitted_vectorizer).  The dense matrix uses
    TruncatedSVD so it works even for very large corpora."""
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )
    Xs = vec.fit_transform(texts)
    n_comp = max(2, min(200, Xs.shape[0] - 1, Xs.shape[1] - 1))
    X = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(Xs)
    return X, vec


def build_raw_tfidf(
    texts: list[str],
    max_features: int = 8000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> tuple[Any, TfidfVectorizer]:
    """Return (sparse_matrix, fitted_vectorizer) without SVD compression —
    used for interpretability (top-terms extraction)."""
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )
    Xs = vec.fit_transform(texts)
    return Xs, vec


def try_sentence_transformer(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray | None:
    """Attempt dense embedding; return None if the package is unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  Using sentence-transformer: {model_name}")
        model = SentenceTransformer(model_name)
        emb = model.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        return np.asarray(emb, dtype=np.float32)
    except ImportError:
        return None
    except Exception as exc:
        print(f"  [WARN] sentence-transformer failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Analysis 1: Entry-level semantic clustering
# ---------------------------------------------------------------------------

def analyse_entry_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """K-means on entry-level TF-IDF of english_desc.

    Outputs:
      - entry cluster assignments
      - category purity per cluster
      - cluster top-terms
    """
    print("\n[1/6] Entry-level semantic clustering …")
    texts = df["english_desc"].tolist()

    # Try dense embedding first; fall back to TF-IDF/SVD
    X_dense = try_sentence_transformer(texts)
    if X_dense is not None:
        X = StandardScaler().fit_transform(X_dense)
        method = "sentence_transformer"
    else:
        X_svd, vec = build_tfidf_matrix(texts)
        X = StandardScaler().fit_transform(X_svd)
        method = "tfidf_svd"

    print(f"  Embedding method: {method}, matrix shape: {X.shape}")

    best_k, best_sil = choose_k(X, k_min=2, k_max=14)
    print(f"  Best k={best_k}, silhouette={best_sil:.4f}")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=60)
    labels = km.fit_predict(X)

    result = df[["year", "era", "category", "direction", "language"]].copy()
    result["entry_cluster"] = labels

    # Category alignment: what category dominates each cluster?
    cat_valid = result[result["category"].notna()]
    cluster_cat = (
        cat_valid.groupby(["entry_cluster", "category"])
        .size()
        .reset_index(name="count")
    )
    dominant = (
        cluster_cat.sort_values("count", ascending=False)
        .groupby("entry_cluster")
        .first()
        .reset_index()[["entry_cluster", "category"]]
        .rename(columns={"category": "dominant_category"})
    )
    result = result.merge(dominant, on="entry_cluster", how="left")

    # Purity against labeled categories
    cat_mask = result["category"].notna()
    if cat_mask.sum() > 0:
        le = LabelEncoder()
        true_cats = le.fit_transform(result.loc[cat_mask, "category"])
        pred_clusters = result.loc[cat_mask, "entry_cluster"].values
        purity = cluster_purity(pred_clusters, true_cats)
        print(f"  Category purity of clusters: {purity:.4f}")
    else:
        purity = 0.0

    # PCA visualization
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X)
    result["pc1"] = z[:, 0]
    result["pc2"] = z[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: colour by cluster
    ax = axes[0]
    palette = sns.color_palette("tab10", best_k)
    for cl in range(best_k):
        mask = result["entry_cluster"] == cl
        ax.scatter(result.loc[mask, "pc1"], result.loc[mask, "pc2"],
                   s=3, alpha=0.3, color=palette[cl],
                   label=f"C{cl} ({dominant.loc[dominant['entry_cluster'] == cl, 'dominant_category'].values[0] if cl in dominant['entry_cluster'].values else '?'})")
    ax.set_title(f"Entry clusters (k={best_k}, sil={best_sil:.3f}, {method})")
    ax.legend(markerscale=4, fontsize=7, ncol=2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Right: colour by labeled category
    ax = axes[1]
    cats = [c for c in CATEGORY_ORDER if c in result["category"].values]
    cat_palette = dict(zip(cats, sns.color_palette("tab10", len(cats))))
    for cat in cats:
        mask = result["category"] == cat
        ax.scatter(result.loc[mask, "pc1"], result.loc[mask, "pc2"],
                   s=3, alpha=0.3, color=cat_palette[cat], label=cat)
    ax.set_title("Entry PCA coloured by labeled category")
    ax.legend(markerscale=4, fontsize=7, ncol=2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "entry_clusters_pca.png")
    plt.close(fig)

    # Cluster profile
    profile = (
        result.groupby("entry_cluster")
        .agg(
            n_entries       = ("year", "count"),
            dominant_cat    = ("dominant_category", "first"),
            min_year        = ("year", "min"),
            max_year        = ("year", "max"),
            pct_income      = ("direction", lambda s: (s == "income").mean()),
            pct_expenditure = ("direction", lambda s: (s == "expenditure").mean()),
            pct_latin       = ("language", lambda s: (s == "latin").mean()),
            pct_english     = ("language", lambda s: (s == "english").mean()),
        )
        .reset_index()
    )
    profile["method"] = method
    profile["silhouette"] = best_sil
    profile["purity"] = purity
    profile.to_csv(OUT_DIR / "entry_cluster_profile.csv", index=False)
    print("  Saved entry_cluster_profile.csv + entry_clusters_pca.png")

    return result


# ---------------------------------------------------------------------------
# Analysis 2: Year-level document clustering on english_description
# ---------------------------------------------------------------------------

def analyse_year_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster years by the aggregate english_description vocabulary.
    Uses english_desc instead of raw OCR → much cleaner signal."""
    print("\n[2/6] Year-level document clustering …")

    year_docs = (
        df.groupby("year")["english_desc"]
        .apply(lambda s: " ".join(s.dropna().astype(str).tolist()))
        .reset_index()
        .rename(columns={"english_desc": "doc"})
        .sort_values("year")
    )
    year_docs = year_docs[year_docs["doc"].str.strip() != ""]

    texts = year_docs["doc"].tolist()
    X_svd, vec = build_tfidf_matrix(texts, max_features=6000)
    X = StandardScaler().fit_transform(X_svd)

    # Also try sentence-transformer on concatenated year docs
    X_dense = try_sentence_transformer(texts)
    if X_dense is not None:
        X_dense = StandardScaler().fit_transform(X_dense)
        # Hybrid: blend TF-IDF and dense
        X_hybrid = np.hstack([X * 0.4, X_dense * 0.6])
        X_final = X_hybrid
        method = "hybrid_tfidf_dense"
    else:
        X_final = X
        method = "tfidf_svd"

    best_k, best_sil = choose_k(X_final, k_min=2, k_max=10)
    print(f"  Best k={best_k}, silhouette={best_sil:.4f} ({method})")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=60)
    labels = km.fit_predict(X_final)

    year_docs["cluster"] = labels
    year_docs["era"] = year_docs["year"].map(era_of_year)
    year_docs["method"] = method
    year_docs["silhouette"] = best_sil

    # Cluster profile
    profile = (
        year_docs.groupby("cluster")
        .agg(
            n_years  = ("year", "count"),
            min_year = ("year", "min"),
            max_year = ("year", "max"),
        )
        .reset_index()
    )
    profile["era_majority"] = profile.apply(
        lambda r: year_docs.loc[
            (year_docs["cluster"] == r["cluster"]), "era"
        ].mode().iloc[0],
        axis=1,
    )

    # Top terms per cluster (re-fit raw TF-IDF on cluster docs)
    cluster_docs = (
        year_docs.groupby("cluster")["doc"]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index()
    )
    Xs_raw, vec_raw = build_raw_tfidf(cluster_docs["doc"].tolist(),
                                       max_features=4000, min_df=1)
    feature_names = vec_raw.get_feature_names_out()
    top_n = 10
    top_terms_list = []
    for i, row in cluster_docs.iterrows():
        idx = list(cluster_docs.index).index(i)
        scores = np.asarray(Xs_raw[idx].todense()).flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        terms = ", ".join(feature_names[top_idx])
        top_terms_list.append(terms)
    cluster_docs["top_terms"] = top_terms_list
    profile = profile.merge(cluster_docs[["cluster", "top_terms"]], on="cluster")
    profile.to_csv(OUT_DIR / "year_cluster_profile.csv", index=False)

    year_docs_out = year_docs[["year", "era", "cluster", "method", "silhouette"]].copy()
    year_docs_out.to_csv(OUT_DIR / "year_cluster_assignments.csv", index=False)

    # PCA scatter
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X_final)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: cluster colours
    ax = axes[0]
    palette = sns.color_palette("tab10", best_k)
    for cl in range(best_k):
        mask = year_docs["cluster"] == cl
        sub = year_docs[mask]
        ax.scatter(z[mask, 0], z[mask, 1], color=palette[cl],
                   s=40, alpha=0.85, label=f"C{cl}")
        for _, r in sub.iterrows():
            idx = year_docs.index.get_loc(r.name)
            ax.text(z[idx, 0], z[idx, 1], str(int(r["year"])),
                    fontsize=5.5, alpha=0.7)
    ax.set_title(f"Year clusters (k={best_k}, sil={best_sil:.3f})")
    ax.legend(fontsize=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Right: cluster timeline
    ax = axes[1]
    for cl in range(best_k):
        mask = year_docs["cluster"] == cl
        sub = year_docs[mask]
        ax.scatter(sub["year"], [cl] * mask.sum(), color=palette[cl], s=20, alpha=0.8)
    _add_era_shading(ax)
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"C{i}" for i in range(best_k)])
    ax.set_title(f"Cluster assignment over time ({method})")
    ax.set_xlabel("Year")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "year_clusters.png")
    plt.close(fig)
    print("  Saved year_cluster_profile.csv + year_clusters.png")

    return year_docs


# ---------------------------------------------------------------------------
# Analysis 3: Year-to-year cosine similarity matrix (semantic temporal drift)
# ---------------------------------------------------------------------------

def analyse_temporal_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Build a year × year cosine-similarity matrix from year-level
    english_description TF-IDF vectors.  Reveals which years are
    semantically nearest neighbours, independently of any clustering."""
    print("\n[3/6] Year-to-year cosine similarity …")

    year_docs = (
        df.groupby("year")["english_desc"]
        .apply(lambda s: " ".join(s.dropna().astype(str).tolist()))
        .reset_index()
        .sort_values("year")
    )
    years = year_docs["year"].values
    texts = year_docs["english_desc"].tolist()

    # Raw TF-IDF (sparse) for cosine similarity (no SVD — preserve angles)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=2, max_features=8000)
    Xs = vec.fit_transform(texts)
    sim_matrix = cosine_similarity(Xs)

    sim_df = pd.DataFrame(sim_matrix, index=years, columns=years)
    sim_df.to_csv(OUT_DIR / "year_cosine_similarity.csv")

    # For each year, its top-5 most similar other years
    neighbours = []
    for i, y in enumerate(years):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        top5_idx = sims.argsort()[::-1][:5]
        neighbours.append({
            "year": int(y),
            "era": era_of_year(int(y)),
            "top5_similar_years": ";".join(str(int(years[j])) for j in top5_idx),
            "top5_similarities": ";".join(f"{sims[j]:.4f}" for j in top5_idx),
            "mean_sim_to_prev10": float(
                np.mean([sim_matrix[i][j] for j in range(max(0, i - 10), i)])
            ) if i > 0 else np.nan,
        })
    nbr_df = pd.DataFrame(neighbours)
    nbr_df.to_csv(OUT_DIR / "year_similarity_neighbours.csv", index=False)

    # Heatmap (every 5th year label to avoid crowding)
    tick_years = years[::5]
    tick_idx = [np.where(years == y)[0][0] for y in tick_years]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=sim_matrix.max())
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_years, rotation=90, fontsize=6)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_years, fontsize=6)
    ax.set_title("Year-to-Year Cosine Similarity (english_description TF-IDF)")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    # Add era dividers
    for yr_boundary in [1760, 1840]:
        idx_b = np.searchsorted(years, yr_boundary)
        ax.axvline(idx_b, color="white", lw=1.0, ls="--")
        ax.axhline(idx_b, color="white", lw=1.0, ls="--")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "year_cosine_heatmap.png")
    plt.close(fig)

    # Rolling mean self-similarity (how much vocabulary changes year-to-year)
    lag1_sim = [sim_matrix[i, i - 1] for i in range(1, len(years))]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(years[1:], lag1_sim, color="steelblue", lw=1.0, alpha=0.7)
    roll = pd.Series(lag1_sim, index=years[1:]).rolling(10, center=True).mean()
    ax.plot(roll.index, roll.values, color="darkblue", lw=2, label="10-yr rolling mean")
    _add_era_shading(ax)
    ax.set_ylabel("Cosine similarity (year vs year-1)")
    ax.set_title("Year-on-Year Vocabulary Continuity")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "year_on_year_continuity.png")
    plt.close(fig)

    print("  Saved year_cosine_similarity.csv, year_cosine_heatmap.png, "
          "year_on_year_continuity.png")
    return nbr_df


# ---------------------------------------------------------------------------
# Analysis 4: Category semantic drift by decade
# ---------------------------------------------------------------------------

def analyse_category_drift(df: pd.DataFrame) -> pd.DataFrame:
    """For each category, build decade-level TF-IDF vectors from english_desc
    and measure how much within-category vocabulary changes over time."""
    print("\n[4/6] Category semantic drift by decade …")

    cat_df = df[df["category"].notna() & df["english_desc"].notna()].copy()
    cat_df["decade"] = (cat_df["year"] // 10) * 10

    records: list[dict] = []
    drift_rows: list[dict] = []

    for cat in CATEGORY_ORDER:
        sub = cat_df[cat_df["category"] == cat]
        if len(sub) < 20:
            continue

        dec_docs = (
            sub.groupby("decade")["english_desc"]
            .apply(lambda s: " ".join(s.tolist()))
            .reset_index()
            .sort_values("decade")
        )
        if len(dec_docs) < 3:
            continue

        decades = dec_docs["decade"].values
        texts = dec_docs["english_desc"].tolist()

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              min_df=1, max_features=2000)
        try:
            Xs = vec.fit_transform(texts)
        except ValueError:
            continue
        sim = cosine_similarity(Xs)

        # consecutive-decade similarity
        for i in range(1, len(decades)):
            drift_rows.append({
                "category": cat,
                "decade_from": int(decades[i - 1]),
                "decade_to":   int(decades[i]),
                "cosine_sim":  float(sim[i, i - 1]),
            })

        # Overall within-category cohesion (mean pairwise similarity)
        n = len(decades)
        pairwise = [sim[i, j] for i in range(n) for j in range(n) if i != j]
        records.append({
            "category": cat,
            "n_decades": n,
            "mean_pairwise_sim": float(np.mean(pairwise)),
            "min_pairwise_sim":  float(np.min(pairwise)),
        })

    drift_df = pd.DataFrame(drift_rows)
    cohesion_df = pd.DataFrame(records).sort_values("mean_pairwise_sim")
    drift_df.to_csv(OUT_DIR / "category_decade_drift.csv", index=False)
    cohesion_df.to_csv(OUT_DIR / "category_cohesion.csv", index=False)

    # Plot: consecutive-decade similarity per category
    if not drift_df.empty:
        cats_present = [c for c in CATEGORY_ORDER if c in drift_df["category"].unique()]
        fig, ax = plt.subplots(figsize=(13, 5))
        palette = dict(zip(cats_present, sns.color_palette("tab10", len(cats_present))))
        for cat in cats_present:
            sub = drift_df[drift_df["category"] == cat]
            ax.plot(sub["decade_to"], sub["cosine_sim"],
                    label=cat, color=palette[cat], lw=1.3, alpha=0.85)
        _add_era_shading(ax)
        ax.set_ylabel("Cosine similarity (consecutive decades)")
        ax.set_title("Within-Category Vocabulary Continuity by Decade")
        ax.set_xlabel("Decade")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "category_decade_drift.png", bbox_inches="tight")
        plt.close(fig)

    # Bar: mean cohesion
    fig, ax = plt.subplots(figsize=(8, 4))
    cohesion_df_plot = cohesion_df.sort_values("mean_pairwise_sim")
    colors = ["#d62728" if v < cohesion_df["mean_pairwise_sim"].median() else "#1f77b4"
              for v in cohesion_df_plot["mean_pairwise_sim"]]
    ax.barh(cohesion_df_plot["category"], cohesion_df_plot["mean_pairwise_sim"],
            color=colors, alpha=0.8)
    ax.axvline(cohesion_df["mean_pairwise_sim"].median(), color="black",
               lw=1, ls="--", label="Median")
    ax.set_xlabel("Mean pairwise cosine similarity (within category, across decades)")
    ax.set_title("Category Vocabulary Cohesion Over Time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "category_cohesion.png")
    plt.close(fig)

    print("  Saved category_decade_drift.csv/png, category_cohesion.csv/png")
    return drift_df


# ---------------------------------------------------------------------------
# Analysis 5: Direction vocabulary contrast (log-odds TF-IDF)
# ---------------------------------------------------------------------------

def analyse_direction_vocabulary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-odds ratio to find words that strongly distinguish
    income from expenditure, overall and per era."""
    print("\n[5/6] Direction vocabulary contrast …")

    dir_df = df[df["direction"].isin(["income", "expenditure"])
                & df["english_desc"].notna()].copy()

    records: list[dict] = []
    top_n = 20

    def _log_odds(pos_counts: np.ndarray, neg_counts: np.ndarray,
                  total_pos: int, total_neg: int,
                  smoothing: float = 0.5) -> np.ndarray:
        p = (pos_counts + smoothing) / (total_pos + smoothing * len(pos_counts))
        q = (neg_counts + smoothing) / (total_neg + smoothing * len(neg_counts))
        return np.log(p / q)

    # Build one vectoriser over all text
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=5, max_features=6000, use_idf=False,
                          norm=None)  # raw counts for log-odds
    vec.fit(dir_df["english_desc"].tolist())
    feat = vec.get_feature_names_out()

    results_rows: list[dict] = []

    def _contrast(sub: pd.DataFrame, scope_label: str) -> pd.DataFrame:
        income_texts = sub.loc[sub["direction"] == "income", "english_desc"].tolist()
        exp_texts    = sub.loc[sub["direction"] == "expenditure", "english_desc"].tolist()
        if not income_texts or not exp_texts:
            return pd.DataFrame()
        Xi = vec.transform(income_texts).toarray().sum(axis=0)
        Xe = vec.transform(exp_texts).toarray().sum(axis=0)
        lo = _log_odds(Xi, Xe, int(Xi.sum()), int(Xe.sum()))
        df_out = pd.DataFrame({"term": feat, "log_odds": lo, "scope": scope_label})
        return df_out

    # Overall
    overall = _contrast(dir_df, "overall")
    if not overall.empty:
        results_rows.append(overall)

    # Per era
    for era in ERA_ORDER:
        sub = dir_df[dir_df["era"] == era]
        if len(sub) < 50:
            continue
        era_df = _contrast(sub, era)
        if not era_df.empty:
            results_rows.append(era_df)

    if not results_rows:
        print("  No contrast data available.")
        return pd.DataFrame()

    all_results = pd.concat(results_rows, ignore_index=True)
    all_results.to_csv(OUT_DIR / "direction_vocabulary_logodds.csv", index=False)

    # Plot top income / expenditure words for each scope
    scopes = all_results["scope"].unique()
    fig, axes = plt.subplots(len(scopes), 2,
                             figsize=(14, 3.5 * len(scopes)),
                             squeeze=False)

    for row_idx, scope in enumerate(scopes):
        sub = all_results[all_results["scope"] == scope]
        income_top  = sub.nlargest(top_n, "log_odds")
        exp_top     = sub.nsmallest(top_n, "log_odds")

        ax = axes[row_idx][0]
        ax.barh(income_top["term"].tolist()[::-1],
                income_top["log_odds"].tolist()[::-1],
                color="#4daf4a", alpha=0.8)
        ax.set_title(f"Income-distinctive words ({scope})", fontsize=8)
        ax.set_xlabel("Log-odds")
        ax.tick_params(labelsize=7)

        ax = axes[row_idx][1]
        ax.barh(exp_top["term"].tolist()[::-1],
                abs(exp_top["log_odds"]).tolist()[::-1],
                color="#e41a1c", alpha=0.8)
        ax.set_title(f"Expenditure-distinctive words ({scope})", fontsize=8)
        ax.set_xlabel("|Log-odds|")
        ax.tick_params(labelsize=7)

    fig.suptitle("Income vs Expenditure Vocabulary Contrast (Log-Odds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "direction_vocabulary_contrast.png", bbox_inches="tight")
    plt.close(fig)

    print("  Saved direction_vocabulary_logodds.csv, direction_vocabulary_contrast.png")
    return all_results


# ---------------------------------------------------------------------------
# Analysis 6: Category top terms by era
# ---------------------------------------------------------------------------

def analyse_category_top_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the most distinctive TF-IDF terms per (era, category) pair.
    Treats each (era, category) as a single document."""
    print("\n[6/6] Category top terms by era …")

    cat_df = df[df["category"].notna() & df["english_desc"].notna()].copy()

    # Build (era, category) aggregate documents
    agg = (
        cat_df.groupby(["era", "category"])["english_desc"]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index()
        .rename(columns={"english_desc": "doc"})
    )
    agg = agg[agg["doc"].str.strip() != ""]

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=1, max_features=5000)
    Xs = vec.fit_transform(agg["doc"].tolist())
    feat = vec.get_feature_names_out()

    records: list[dict] = []
    top_n = 8
    for i, row in agg.iterrows():
        idx = list(agg.index).index(i)
        scores = np.asarray(Xs[idx].todense()).flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        for rank, tidx in enumerate(top_idx):
            records.append({
                "era":      row["era"],
                "category": row["category"],
                "rank":     rank + 1,
                "term":     feat[tidx],
                "tfidf":    float(scores[tidx]),
            })

    terms_df = pd.DataFrame(records)
    terms_df.to_csv(OUT_DIR / "category_era_top_terms.csv", index=False)

    # Visualise as a grid of horizontal bar charts
    eras = ERA_ORDER
    cats = [c for c in CATEGORY_ORDER if c in terms_df["category"].unique()]
    n_cats = len(cats)
    n_eras = len(eras)

    fig, axes = plt.subplots(n_cats, n_eras,
                              figsize=(n_eras * 5, n_cats * 1.8),
                              squeeze=False)
    for r, cat in enumerate(cats):
        for c, era in enumerate(eras):
            ax = axes[r][c]
            sub = terms_df[(terms_df["category"] == cat) & (terms_df["era"] == era)]
            if sub.empty:
                ax.axis("off")
                continue
            ax.barh(sub["term"].tolist()[::-1], sub["tfidf"].tolist()[::-1],
                    color="#4878cf", alpha=0.75)
            if r == 0:
                ax.set_title(ERA_LABELS.get(era, era), fontsize=8)
            if c == 0:
                ax.set_ylabel(cat, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("")

    fig.suptitle("Top TF-IDF Terms per Category × Era (english_description)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "category_era_top_terms.png", bbox_inches="tight")
    plt.close(fig)

    print("  Saved category_era_top_terms.csv + category_era_top_terms.png")
    return terms_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Enriched Ledger — Embedding Analysis")
    print("=" * 60)

    df = load_data()

    # 1. Entry-level clustering
    entry_result = analyse_entry_clusters(df)
    entry_result[["year", "era", "category", "direction", "language",
                  "entry_cluster", "dominant_category"]].to_csv(
        OUT_DIR / "entry_cluster_assignments.csv", index=False
    )

    # 2. Year-level document clustering
    year_docs = analyse_year_clusters(df)

    # 3. Temporal cosine similarity
    nbr_df = analyse_temporal_similarity(df)

    # 4. Category semantic drift
    drift_df = analyse_category_drift(df)

    # 5. Direction vocabulary contrast
    logodds_df = analyse_direction_vocabulary(df)

    # 6. Category top terms by era
    terms_df = analyse_category_top_terms(df)

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS SUMMARY")
    print("=" * 60)

    # Year cluster summary
    if not year_docs.empty:
        print("\nYear cluster assignments:")
        yd = year_docs.sort_values("year")
        sil = float(yd["silhouette"].iloc[0]) if "silhouette" in yd else 0
        method = yd["method"].iloc[0] if "method" in yd else "?"
        print(f"  Method: {method},  silhouette: {sil:.4f}")
        for cl in sorted(yd["cluster"].unique()):
            sub = yd[yd["cluster"] == cl]
            era_dist = sub["era"].value_counts().to_dict()
            print(f"  Cluster {cl}: {len(sub)} years "
                  f"({sub['year'].min()}–{sub['year'].max()})  "
                  f"era mix: {era_dist}")

    # Category cohesion ranking
    cohesion_path = OUT_DIR / "category_cohesion.csv"
    if cohesion_path.exists():
        coh = pd.read_csv(cohesion_path).sort_values("mean_pairwise_sim", ascending=False)
        print("\nCategory vocabulary cohesion (high = stable vocabulary over decades):")
        for _, r in coh.iterrows():
            print(f"  {r['category']:20s}: {r['mean_pairwise_sim']:.4f}")

    # Semantic drift — most discontinuous decade transitions
    if not drift_df.empty:
        low_sim = drift_df.nsmallest(10, "cosine_sim")
        print("\nTop 10 most discontinuous decade transitions (lowest cosine sim):")
        for _, r in low_sim.iterrows():
            print(f"  {r['category']:20s}  {int(r['decade_from'])}s→{int(r['decade_to'])}s "
                  f"sim={r['cosine_sim']:.4f}")

    # Temporal neighbours of key years
    for key_year in [1760, 1800, 1840, 1880]:
        row = nbr_df[nbr_df["year"] == key_year]
        if not row.empty:
            print(f"\n  Most similar years to {key_year}: "
                  f"{row.iloc[0]['top5_similar_years']}")

    print(f"\nAll outputs written to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
