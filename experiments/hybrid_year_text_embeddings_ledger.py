from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "experiments" / "reports" / "ledger_clean"
OUT_DIR = REPORT_DIR / "hybrid_embeddings"


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None]
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def build_year_docs() -> pd.DataFrame:
    src = REPORT_DIR / "all_rows_with_amounts.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    df = pd.read_csv(src)
    df = df[df["row_type"].isin(["entry", "total"])].copy()
    df = df[df["description_norm"].fillna("") != ""]

    docs = (
        df.groupby("year", as_index=False)["description_norm"]
        .apply(lambda s: " ".join(s.astype(str).tolist()))
        .rename(columns={"description_norm": "doc"})
        .sort_values("year")
    )
    return docs


def embed_tfidf(docs: Iterable[str], max_features: int = 5000) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=max_features)
    X = vec.fit_transform(list(docs))
    n_comp = max(2, min(128, X.shape[0] - 1, X.shape[1] - 1))
    return TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)


def embed_english_sentence_transformer(docs: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(docs, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb)


def embed_latin_transformer(docs: list[str], model_name_or_path: str, batch_size: int = 8, max_length: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state.cpu().numpy(), enc["attention_mask"].cpu().numpy())
            all_emb.append(pooled)
    emb = np.vstack(all_emb)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-9, None)
    return emb


def choose_k(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[int, float]:
    best_k, best_score = 2, -1.0
    upper = min(k_max, len(X) - 1)
    for k in range(k_min, upper + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=40)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if s > best_score:
            best_score = s
            best_k = k
    return best_k, best_score


def run(mode: str, latin_model: str | None, english_model: str, latin_weight: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs_df = build_year_docs()
    docs = docs_df["doc"].tolist()

    if mode == "tfidf":
        X = embed_tfidf(docs)
        method_used = "tfidf_svd"
    else:
        lat_ok = False
        eng_ok = False
        lat_emb = None
        eng_emb = None

        if latin_model:
            try:
                lat_emb = embed_latin_transformer(docs, latin_model)
                lat_ok = True
            except Exception as e:
                print(f"[WARN] Latin embedding unavailable ({latin_model}): {e}")

        try:
            eng_emb = embed_english_sentence_transformer(docs, english_model)
            eng_ok = True
        except Exception as e:
            print(f"[WARN] English embedding unavailable ({english_model}): {e}")

        if lat_ok and eng_ok:
            lw = float(np.clip(latin_weight, 0.0, 1.0))
            ew = 1.0 - lw
            X = np.hstack([lat_emb * lw, eng_emb * ew])
            method_used = "hybrid_latin_plus_english"
        elif lat_ok:
            X = lat_emb
            method_used = "latin_only"
        elif eng_ok:
            X = eng_emb
            method_used = "english_only"
        else:
            X = embed_tfidf(docs)
            method_used = "tfidf_svd_fallback"

    X = StandardScaler().fit_transform(X)
    best_k, sil = choose_k(X)
    km = KMeans(n_clusters=best_k, random_state=42, n_init=60)
    labels = km.fit_predict(X)

    out = docs_df[["year"]].copy()
    out["cluster"] = labels
    out["embedding_method"] = method_used
    out.to_csv(OUT_DIR / "year_text_embedding_clusters.csv", index=False)

    profile = out.groupby("cluster", as_index=False).agg(n_years=("year", "count"), min_year=("year", "min"), max_year=("year", "max"))
    profile.to_csv(OUT_DIR / "year_text_cluster_profile.csv", index=False)

    meta = pd.DataFrame(
        [
            {
                "embedding_method": method_used,
                "n_years": len(out),
                "k": best_k,
                "silhouette": sil,
                "latin_model": latin_model or "",
                "english_model": english_model,
                "latin_weight": latin_weight,
            }
        ]
    )
    meta.to_csv(OUT_DIR / "year_text_embedding_meta.csv", index=False)

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X)
    plot_df = out.copy()
    plot_df["pc1"] = z[:, 0]
    plot_df["pc2"] = z[:, 1]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="cluster", palette="tab10", ax=ax, s=60)
    for _, r in plot_df.iterrows():
        ax.text(r["pc1"], r["pc2"], str(int(r["year"])), fontsize=7, alpha=0.7)
    ax.set_title(f"Year text embedding clusters ({method_used}, k={best_k}, sil={sil:.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "year_text_embedding_clusters_pca.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.scatterplot(data=out, x="year", y="cluster", hue="cluster", palette="tab10", ax=ax, s=55)
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title(f"Year cluster timeline ({method_used})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "year_text_embedding_clusters_timeline.png", dpi=180)
    plt.close(fig)

    print(f"Done. Method={method_used}, k={best_k}, silhouette={sil:.4f}")
    print(f"Outputs: {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Latin+English year text embedding clustering")
    parser.add_argument("--mode", choices=["tfidf", "hybrid"], default="hybrid")
    parser.add_argument(
        "--latin-model",
        default="",
        help="Local/HF Latin model path or id. For dbamman/latin-bert workflows, pass the converted HuggingFace model path.",
    )
    parser.add_argument("--english-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--latin-weight", type=float, default=0.6)
    args = parser.parse_args()

    run(mode=args.mode, latin_model=(args.latin_model or None), english_model=args.english_model, latin_weight=args.latin_weight)


if __name__ == "__main__":
    main()
