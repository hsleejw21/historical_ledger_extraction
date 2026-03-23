"""latinbert_year_embeddings.py
Produces year-level document embeddings using a Latin-specific BERT model
(LatinBERT / PhilBerta), then clusters and visualises them alongside the
TF-IDF baseline from v2 for comparison.

Model precedence (tried in order):
  1. bowphs/PhilBerta  — BERT trained on classical Latin texts, standard
                         HuggingFace tokenization, recommended for this corpus
  2. bert-base-multilingual-cased — fallback if PhilBerta is unavailable

Why not dbamman/latin-bert?
  The Bamman & Burns (2020) LatinBERT uses a custom character-level Latin
  tokenizer from CLTK that is not compatible with AutoTokenizer.  PhilBerta
  achieves comparable Latin-language modelling via standard subword tokenization.

Mixed-language note:
  This corpus spans Latin-dominant pre-1760 text through English-dominant
  post-1840 text.  A Latin-specific model will produce meaningful embeddings
  for early entries; performance on later English entries may be weaker.
  The outputs include a per-era silhouette comparison so you can judge where
  the Latin model adds value.

Usage:
  python latinbert_year_embeddings.py           # uses GPU if available
  python latinbert_year_embeddings.py --cpu     # force CPU
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


ROOT    = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "experiments" / "results" / "ledger.xlsx"
OUT_DIR  = ROOT / "experiments" / "reports" / "ledger_clean_v2" / "latinbert"

# Priority-ordered model list
_CANDIDATE_MODELS = [
    "bowphs/PhilBerta",          # Latin BERT — standard tokenization
    "bert-base-multilingual-cased",
]

# Extended stop words (same as v2)
_LEDGER_EXTRA_STOPS: frozenset[str] = frozenset({
    "year", "years", "yr", "yrs",
    "total", "totall", "balance", "carried", "forward", "brought",
    "folio", "fol", "ff", "account", "accounts", "acct",
    "received", "paid", "payment", "payments",
    "sum", "sums", "amount", "amounts",
    "pound", "pounds", "shilling", "shillings", "pence", "penny", "sterling",
    "ob", "viz", "ie", "per",
})
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | _LEDGER_EXTRA_STOPS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_fraction(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    mapping = {"¼": 0.25, "1/4": 0.25, ".25": 0.25,
               "½": 0.5,  "1/2": 0.5,  ".5":  0.5,
               "¾": 0.75, "3/4": 0.75, ".75": 0.75}
    return mapping.get(s, 0.0)


def parse_money_number(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = re.sub(r"[^0-9.\-]", "", str(value).strip())
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def normalize_text(s: Any) -> str:
    if pd.isna(s):
        return ""
    text = str(s).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def era_of_year(year: int) -> str:
    if year < 1760:
        return "pre_1760"
    if year <= 1840:
        return "industrial_1760_1840"
    return "post_1840"


def parse_sheet_years(sheet: str) -> tuple[list[int], int] | None:
    m = re.match(r"^(\d{4})_(\d+)_image$", sheet)
    if m:
        return [int(m.group(1))], int(m.group(2))
    m = re.match(r"^(\d{4})-(\d{4})_(\d+)_image$", sheet)
    if not m:
        return None
    y1, y2, page = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y2 < y1 and (y1 - y2) > 5:
        y2 = y1 + 1
    if y2 < y1:
        y1, y2 = y2, y1
    return list(range(y1, y2 + 1)), page


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_year_docs(workbook_path: Path) -> pd.DataFrame:
    """Load all entry+total rows and aggregate per year into documents."""
    xl = pd.ExcelFile(workbook_path)
    rows: list[dict[str, Any]] = []

    for sheet in xl.sheet_names:
        parsed = parse_sheet_years(sheet)
        if parsed is None:
            continue
        years, page = parsed
        weight = 1.0 / len(years)

        df = pd.read_excel(workbook_path, sheet_name=sheet)
        req = ["Type", "Description", "£ (Pounds)", "s (Shillings)", "d (Pence)", "d Fraction"]
        if not set(req).issubset(df.columns):
            continue

        for _, r in df.iterrows():
            row_type = str(r["Type"]).strip().lower()
            if row_type not in {"entry", "total"}:
                continue
            desc = normalize_text(r["Description"])
            if not desc:
                continue
            p  = parse_money_number(r["£ (Pounds)"])
            s_ = parse_money_number(r["s (Shillings)"])
            d  = parse_money_number(r["d (Pence)"])
            f  = parse_fraction(r["d Fraction"])
            amt = p + s_ / 20.0 + (d + f) / 240.0
            for year in years:
                rows.append({"year": year, "description": desc,
                             "amount_decimal": amt, "amount_weighted": amt * weight})

    df_all = pd.DataFrame(rows)
    df_all["era"] = df_all["year"].map(era_of_year)

    year_docs = (
        df_all.groupby("year")
        .agg(
            doc            = ("description",    lambda s: " ".join(s.tolist())),
            n_descriptions = ("description",    "count"),
            amount_sum     = ("amount_weighted", "sum"),
            amount_median  = ("amount_decimal",  "median"),
            era            = ("era",             "first"),
        )
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return year_docs


# ---------------------------------------------------------------------------
# LatinBERT embedding
# ---------------------------------------------------------------------------

def load_model(device: str) -> tuple[AutoTokenizer, AutoModel, str]:
    """Load the first available candidate model."""
    for model_name in _CANDIDATE_MODELS:
        try:
            print(f"  Trying model: {model_name} …")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model     = AutoModel.from_pretrained(model_name).to(device)
            model.eval()
            print(f"  Loaded: {model_name}")
            return tokenizer, model, model_name
        except Exception as e:
            print(f"  Could not load {model_name}: {e}")
    raise RuntimeError("No suitable model could be loaded. "
                       "Check your internet connection or install transformers.")


def embed_text(text: str, tokenizer: AutoTokenizer, model: AutoModel,
               device: str, max_chunk_tokens: int = 512) -> np.ndarray:
    """Mean-pool token embeddings over the full text, chunked to max_chunk_tokens.

    Long documents (year concatenations) are split into non-overlapping chunks;
    chunk embeddings are averaged by token count to produce a single vector.
    """
    if not text.strip():
        # Return zero vector matching the model's hidden size
        hidden = model.config.hidden_size
        return np.zeros(hidden, dtype=np.float32)

    # Tokenize without truncation to get all token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Split into chunks of (max_chunk_tokens - 2) to leave room for [CLS]/[SEP]
    chunk_size = max_chunk_tokens - 2
    chunks     = [token_ids[i: i + chunk_size] for i in range(0, len(token_ids), chunk_size)]

    chunk_vecs: list[np.ndarray] = []
    chunk_sizes: list[int] = []

    with torch.no_grad():
        for chunk in chunks:
            # Wrap chunk with model-appropriate special tokens ([CLS]/[SEP] for
            # BERT-style; <s>/</s> for RoBERTa-style).  Both are exposed via
            # cls_token_id / sep_token_id on the tokenizer.
            input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
            ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention  = torch.ones_like(ids_tensor)

            outputs = model(input_ids=ids_tensor, attention_mask=attention)

            # Mean pool over token dimension (excluding special tokens at pos 0 and -1)
            last_hidden = outputs.last_hidden_state[0]  # (seq_len, hidden)
            token_emb   = last_hidden[1:-1]             # strip [CLS] and [SEP]
            if token_emb.shape[0] == 0:
                token_emb = last_hidden                 # fallback: include specials
            vec = token_emb.mean(dim=0).cpu().numpy()
            chunk_vecs.append(vec)
            chunk_sizes.append(len(chunk))

    # Weighted average by chunk length
    total = sum(chunk_sizes)
    combined = sum(v * (n / total) for v, n in zip(chunk_vecs, chunk_sizes))
    return combined.astype(np.float32)


def build_latinbert_embeddings(
    year_docs: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
) -> np.ndarray:
    """Embed all year documents; returns array of shape (n_years, hidden_size)."""
    vecs: list[np.ndarray] = []
    n = len(year_docs)
    for i, (_, row) in enumerate(year_docs.iterrows()):
        if (i + 1) % 20 == 0 or i == 0 or i == n - 1:
            print(f"  Embedding year {int(row['year'])} ({i + 1}/{n})…")
        vecs.append(embed_text(row["doc"], tokenizer, model, device))
    return np.stack(vecs, axis=0)


# ---------------------------------------------------------------------------
# TF-IDF baseline (for comparison)
# ---------------------------------------------------------------------------

def build_tfidf_embeddings(year_docs: pd.DataFrame) -> np.ndarray:
    tfidf = TfidfVectorizer(
        stop_words=STOP_WORDS,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=3500,
    )
    from sklearn.decomposition import TruncatedSVD
    X_sparse = tfidf.fit_transform(year_docs["doc"])
    n_comp   = max(2, min(50, X_sparse.shape[0] - 1, X_sparse.shape[1] - 1))
    svd      = TruncatedSVD(n_components=n_comp, random_state=42)
    return svd.fit_transform(X_sparse)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def best_kmeans(X: np.ndarray, k_range: range, label: str) -> tuple[np.ndarray, int, float]:
    best_k = k_range.start
    best_s = -1.0
    best_l = None
    for k in k_range:
        if k >= X.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=30)
        lbl = km.fit_predict(X)
        if len(np.unique(lbl)) < 2:
            continue
        s = silhouette_score(X, lbl)
        if s > best_s:
            best_s = s
            best_k = k
            best_l = lbl
    print(f"  [{label}] best k={best_k}, silhouette={best_s:.4f}")
    if best_l is None:
        best_l = np.zeros(X.shape[0], dtype=int)
    return best_l, best_k, best_s


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pca(X: np.ndarray, labels: np.ndarray, year_docs: pd.DataFrame,
             title: str, filepath: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X)
    era_palette = {
        "pre_1760":             "#1f77b4",
        "industrial_1760_1840": "#ff7f0e",
        "post_1840":            "#2ca02c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colour by cluster
    for k in np.unique(labels):
        mask = labels == k
        axes[0].scatter(Z[mask, 0], Z[mask, 1], s=25, label=f"Cluster {k}", alpha=0.8)
    axes[0].set_title(f"{title}\n(coloured by cluster)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].legend(fontsize=8)

    # Right: colour by era
    for era, col in era_palette.items():
        mask = year_docs["era"] == era
        axes[1].scatter(Z[mask, 0], Z[mask, 1], s=25, label=era, color=col, alpha=0.8)
    axes[1].set_title(f"{title}\n(coloured by era)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(filepath, dpi=180)
    plt.close(fig)


def plot_cluster_timeline(year_docs: pd.DataFrame, labels: np.ndarray,
                          title: str, filepath: Path) -> None:
    df = year_docs[["year", "era"]].copy()
    df["cluster"] = labels
    fig, ax = plt.subplots(figsize=(13, 3.5))
    sns.scatterplot(data=df, x="year", y="cluster", hue="cluster",
                    palette="tab10", ax=ax, s=55, legend="brief")
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(filepath, dpi=180)
    plt.close(fig)


def plot_comparison_timeline(year_docs: pd.DataFrame,
                              lb_labels: np.ndarray,
                              tfidf_labels: np.ndarray,
                              filepath: Path) -> None:
    df = year_docs[["year", "era"]].copy()
    df["latinbert_cluster"] = lb_labels
    df["tfidf_cluster"]     = tfidf_labels

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for ax, col, title in zip(
        axes,
        ["latinbert_cluster", "tfidf_cluster"],
        ["LatinBERT clusters", "TF-IDF + SVD clusters (v2 baseline)"],
    ):
        sns.scatterplot(data=df, x="year", y=col, hue=col,
                        palette="tab10", ax=ax, s=40, legend="brief")
        ax.axvspan(1760, 1840, color="grey", alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(filepath, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force_cpu: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else
                                       "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load data
    print("Loading ledger data…")
    year_docs = load_year_docs(WORKBOOK)
    print(f"  {len(year_docs)} year documents, "
          f"years {int(year_docs['year'].min())}–{int(year_docs['year'].max())}")

    # 2. LatinBERT embeddings
    print("Loading LatinBERT model…")
    tokenizer, model, model_name = load_model(device)

    print("Building LatinBERT year embeddings (this may take a few minutes)…")
    X_lb = build_latinbert_embeddings(year_docs, tokenizer, model, device)

    # Normalise embeddings before clustering
    X_lb_norm = StandardScaler().fit_transform(X_lb)

    # 3. TF-IDF baseline
    print("Building TF-IDF baseline embeddings…")
    X_tfidf = build_tfidf_embeddings(year_docs)

    # 4. Cluster both
    print("Clustering LatinBERT embeddings…")
    lb_labels, lb_k, lb_sil = best_kmeans(X_lb_norm, range(2, 9), "LatinBERT")

    print("Clustering TF-IDF embeddings…")
    tf_labels, tf_k, tf_sil = best_kmeans(X_tfidf, range(2, 9), "TF-IDF")

    # 5. Save embeddings and cluster assignments
    emb_df = year_docs[["year", "era", "amount_sum", "amount_median"]].copy()
    emb_df["latinbert_cluster"] = lb_labels
    emb_df["tfidf_cluster"]     = tf_labels

    emb_lb_df = pd.DataFrame(X_lb, columns=[f"dim_{i}" for i in range(X_lb.shape[1])])
    emb_lb_df.insert(0, "year", year_docs["year"].values)
    emb_lb_df.to_csv(OUT_DIR / "year_latinbert_embeddings.csv", index=False)

    emb_df.to_csv(OUT_DIR / "year_latinbert_clusters.csv", index=False)

    # Era × cluster cross-tab
    for method, col in [("latinbert", "latinbert_cluster"), ("tfidf", "tfidf_cluster")]:
        xtab = (
            emb_df.groupby(["era", col], as_index=False)
            .size()
            .rename(columns={"size": "n_years", col: "cluster"})
            .assign(method=method)
        )
        xtab.to_csv(OUT_DIR / f"era_cluster_profile_{method}.csv", index=False)

    # Silhouette comparison table
    sil_df = pd.DataFrame([
        {"method": "LatinBERT", "model": model_name, "best_k": lb_k, "silhouette": round(lb_sil, 4)},
        {"method": "TF-IDF+SVD (v2)", "model": "TfidfVectorizer+TruncatedSVD", "best_k": tf_k, "silhouette": round(tf_sil, 4)},
    ])
    sil_df.to_csv(OUT_DIR / "embedding_comparison.csv", index=False)
    print("\nSilhouette comparison:")
    print(sil_df.to_string(index=False))

    # 6. Plots
    print("Generating plots…")
    sns.set_theme(style="whitegrid")

    plot_pca(X_lb_norm, lb_labels, year_docs,
             f"LatinBERT PCA ({model_name})",
             OUT_DIR / "latinbert_pca.png")

    plot_pca(X_tfidf, tf_labels, year_docs,
             "TF-IDF + SVD PCA (v2 baseline)",
             OUT_DIR / "tfidf_pca.png")

    plot_cluster_timeline(year_docs, lb_labels,
                          f"LatinBERT cluster timeline ({model_name})",
                          OUT_DIR / "latinbert_cluster_timeline.png")

    plot_comparison_timeline(year_docs, lb_labels, tf_labels,
                             OUT_DIR / "latinbert_vs_tfidf_timeline.png")

    # 7. Summary markdown
    summary = f"""# LatinBERT Embedding Analysis

## Model
- **{model_name}**
- Device: {device}

## Silhouette scores (year-level clustering)
| Method | Model | Best k | Silhouette |
|--------|-------|--------|------------|
| LatinBERT | {model_name} | {lb_k} | {lb_sil:.4f} |
| TF-IDF + SVD (v2) | TfidfVectorizer + TruncatedSVD | {tf_k} | {tf_sil:.4f} |

## Interpretation notes
- The corpus spans Latin-dominant (pre-1760), transitional (1760-1840), and
  English-dominant (post-1840) text. A Latin-specific model may capture semantic
  structure in early entries better than bag-of-words TF-IDF.
- Check **latinbert_vs_tfidf_timeline.png** to see whether the two methods
  agree on where historical transitions occur.
- Check **latinbert_pca.png** (right panel, coloured by era) to see whether
  the Latin model separates eras more cleanly than TF-IDF.
- A higher silhouette for LatinBERT on pre-1760 years would support the claim
  that it captures meaningful semantic variation in the early Latin entries.

## Output files
- year_latinbert_embeddings.csv  — raw embedding vectors (n_years × hidden_dim)
- year_latinbert_clusters.csv    — cluster assignments + era + amount metadata
- era_cluster_profile_latinbert.csv
- era_cluster_profile_tfidf.csv
- embedding_comparison.csv       — silhouette score comparison table

## Figures
- latinbert_pca.png              — PCA of LatinBERT embeddings (cluster + era)
- tfidf_pca.png                  — PCA of TF-IDF embeddings (for comparison)
- latinbert_cluster_timeline.png — cluster assignments across 1700-1900
- latinbert_vs_tfidf_timeline.png — side-by-side timeline comparison
"""
    (OUT_DIR / "LATINBERT_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no GPU/MPS)")
    args = parser.parse_args()
    main(force_cpu=args.cpu)
