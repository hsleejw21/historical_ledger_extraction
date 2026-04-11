from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "experiments" / "results" / "ledger.xlsx"
OUT_DIR = ROOT / "experiments" / "reports" / "ledger_clean"


def parse_fraction(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    mapping = {
        "¼": 0.25,
        "1/4": 0.25,
        ".25": 0.25,
        "½": 0.5,
        "1/2": 0.5,
        ".5": 0.5,
        "¾": 0.75,
        "3/4": 0.75,
        ".75": 0.75,
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
    s = str(value).strip()
    if not s:
        return 0.0
    s = re.sub(r"[^0-9.\-]", "", s)
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

    # Handle likely OCR typo like 1769-1700 => treat as next year span.
    if y2 < y1 and (y1 - y2) > 5:
        y2 = y1 + 1

    if y2 < y1:
        y1, y2 = y2, y1

    years = list(range(y1, y2 + 1))
    return years, page


def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


def load_data(workbook_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(workbook_path)
    rows: list[dict[str, Any]] = []

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

            pounds = parse_money_number(r["£ (Pounds)"])
            shillings = parse_money_number(r["s (Shillings)"])
            pence = parse_money_number(r["d (Pence)"])
            frac = parse_fraction(r["d Fraction"])
            amount_decimal = pounds + (shillings / 20.0) + ((pence + frac) / 240.0)

            desc_raw = "" if pd.isna(r["Description"]) else str(r["Description"])
            desc_norm = normalize_text(desc_raw)

            for year in years:
                rows.append(
                    {
                        "sheet": sheet,
                        "sheet_year_span": "-".join([str(years[0]), str(years[-1])]) if len(years) > 1 else str(year),
                        "year": year,
                        "page": page,
                        "row_idx": int(i) + 1,
                        "row_type": row_type,
                        "description_raw": desc_raw,
                        "description_norm": desc_norm,
                        "amount_decimal": amount_decimal,
                        "year_weight": year_weight,
                        "amount_weighted": amount_decimal * year_weight,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No rows loaded from workbook.")
    out["era"] = out["year"].map(era_of_year)
    return out


def build_yearly_numeric(df: pd.DataFrame) -> pd.DataFrame:
    data_rows = df[df["row_type"].isin(["entry", "total"])].copy()

    yearly = (
        data_rows.groupby("year", as_index=False)
        .agg(
            n_pages=("page", "nunique"),
            n_data_rows=("year_weight", "sum"),
            n_entry_rows=("year_weight", lambda s: float(s[data_rows.loc[s.index, "row_type"] == "entry"].sum())),
            n_total_rows=("year_weight", lambda s: float(s[data_rows.loc[s.index, "row_type"] == "total"].sum())),
            amount_sum=("amount_weighted", "sum"),
            amount_mean=("amount_decimal", "mean"),
            amount_median=("amount_decimal", "median"),
            amount_p90=("amount_decimal", lambda s: float(np.quantile(s, 0.90))),
            amount_max=("amount_decimal", "max"),
        )
        .sort_values("year")
    )
    yearly["n_data_rows"] = yearly["n_data_rows"].round(3)
    yearly["n_entry_rows"] = yearly["n_entry_rows"].round(3)
    yearly["n_total_rows"] = yearly["n_total_rows"].round(3)
    yearly["era"] = yearly["year"].map(era_of_year)
    return yearly


def detect_change_points(yearly: pd.DataFrame) -> pd.DataFrame:
    metrics = ["amount_sum", "n_data_rows", "amount_median"]
    records: list[dict[str, Any]] = []

    for metric in metrics:
        diff = yearly[metric].diff().fillna(0.0)
        rz = robust_zscore(diff.abs())
        flag = rz >= 2.5
        for i in range(len(yearly)):
            records.append(
                {
                    "year": int(yearly.iloc[i]["year"]),
                    "metric": metric,
                    "delta": float(diff.iloc[i]),
                    "abs_delta_robust_z": float(rz.iloc[i]),
                    "is_change_point": bool(flag.iloc[i]),
                }
            )

    cp = pd.DataFrame(records)
    return cp.sort_values(["is_change_point", "abs_delta_robust_z"], ascending=[False, False])


def yearly_text_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, Any]:
    data_rows = df[df["row_type"].isin(["entry", "total"])].copy()
    data_rows = data_rows[data_rows["description_norm"] != ""]

    year_docs = (
        data_rows.groupby("year")["description_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index(name="doc")
        .sort_values("year")
    )

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=3500)
    X = tfidf.fit_transform(year_docs["doc"])
    return year_docs, tfidf, X


def cluster_years(
    year_docs: pd.DataFrame,
    X_text: Any,
    yearly_numeric: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    num = yearly_numeric.set_index("year").loc[year_docs["year"], ["amount_sum", "n_data_rows", "amount_median"]]
    X_num = StandardScaler().fit_transform(num)
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


def top_terms_by_year(year_docs: pd.DataFrame, tfidf: TfidfVectorizer, X_text: Any, top_n: int = 15) -> pd.DataFrame:
    terms = np.array(tfidf.get_feature_names_out())
    records: list[dict[str, Any]] = []
    for i, y in enumerate(year_docs["year"].tolist()):
        row = X_text[i].toarray().ravel()
        idx = np.argsort(row)[-top_n:][::-1]
        for rank, j in enumerate(idx, start=1):
            if row[j] <= 0:
                continue
            records.append(
                {
                    "year": int(y),
                    "rank": rank,
                    "term": terms[j],
                    "tfidf": float(row[j]),
                }
            )
    return pd.DataFrame(records)


def era_term_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_rows = df[df["row_type"].isin(["entry", "total"])].copy()
    data_rows = data_rows[data_rows["description_norm"] != ""]

    era_docs = (
        data_rows.groupby("era")["description_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .reindex(["pre_1760", "industrial_1760_1840", "post_1840"])
        .fillna("")
    )

    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=5000)
    X = vec.fit_transform(era_docs.tolist())
    terms = np.array(vec.get_feature_names_out())
    cnt = pd.DataFrame(X.toarray(), index=era_docs.index, columns=terms)

    freq = cnt.div(cnt.sum(axis=1).replace(0, 1), axis=0)

    top_records: list[dict[str, Any]] = []
    for era in freq.index:
        top = freq.loc[era].sort_values(ascending=False).head(30)
        for rank, (term, val) in enumerate(top.items(), start=1):
            top_records.append({"era": era, "rank": rank, "term": term, "relative_freq": float(val)})
    era_top = pd.DataFrame(top_records)

    comp = pd.DataFrame({"term": terms})
    comp["pre"] = freq.loc["pre_1760", terms].values
    comp["ind"] = freq.loc["industrial_1760_1840", terms].values
    comp["post"] = freq.loc["post_1840", terms].values
    comp["delta_post_vs_pre"] = comp["post"] - comp["pre"]
    comp["delta_ind_vs_pre"] = comp["ind"] - comp["pre"]

    emerging_post = comp.sort_values("delta_post_vs_pre", ascending=False).head(80)
    emerging_ind = comp.sort_values("delta_ind_vs_pre", ascending=False).head(80)
    emerging = pd.concat(
        [
            emerging_ind.assign(change="industrial_vs_pre"),
            emerging_post.assign(change="post_vs_pre"),
        ],
        ignore_index=True,
    )
    era_term_long = (
        freq.reset_index()
        .rename(columns={"index": "era"})
        .melt(id_vars="era", var_name="term", value_name="relative_freq")
    )
    return era_top, emerging, era_term_long


def recurring_accounts(df: pd.DataFrame) -> pd.DataFrame:
    data_rows = df[df["row_type"].isin(["entry", "total"])].copy()
    data_rows = data_rows[data_rows["description_norm"] != ""]

    g = (
        data_rows.groupby("description_norm", as_index=False)
        .agg(
            n_occurrences=("description_norm", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            sample_text=("description_raw", "first"),
        )
        .sort_values("n_occurrences", ascending=False)
    )
    g["first_era"] = g["first_year"].map(era_of_year)
    return g


def header_account_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    headers = df[df["row_type"] == "header"].copy()
    headers = headers[headers["description_norm"] != ""]

    first_app = (
        headers.groupby("description_norm", as_index=False)
        .agg(
            n_occurrences=("description_norm", "count"),
            n_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            sample_text=("description_raw", "first"),
        )
        .sort_values(["n_years", "n_occurrences"], ascending=[False, False])
    )
    first_app["first_era"] = first_app["first_year"].map(era_of_year)

    yearly_counts = (
        headers.groupby("year", as_index=False)
        .agg(
            n_header_rows=("row_type", "count"),
            n_unique_header_texts=("description_norm", "nunique"),
        )
        .sort_values("year")
    )

    era_docs = (
        headers.groupby("era")["description_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .reindex(["pre_1760", "industrial_1760_1840", "post_1840"])
        .fillna("")
    )
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=3000)
    X = vec.fit_transform(era_docs.tolist())
    terms = np.array(vec.get_feature_names_out())
    mat = pd.DataFrame(X.toarray(), index=era_docs.index, columns=terms)
    freq = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)

    era_top_records: list[dict[str, Any]] = []
    for era in freq.index:
        top = freq.loc[era].sort_values(ascending=False).head(40)
        for rank, (term, val) in enumerate(top.items(), start=1):
            era_top_records.append({"era": era, "rank": rank, "term": term, "relative_freq": float(val)})
    era_top = pd.DataFrame(era_top_records)

    return first_app, yearly_counts, era_top


def make_plots(yearly: pd.DataFrame, cp: pd.DataFrame, year_clusters: pd.DataFrame, era_top: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(yearly["year"], yearly["amount_sum"], color="#1f77b4", marker="o", linewidth=2, label="amount_sum")
    ax1.set_ylabel("Total amount (decimal pounds)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["n_data_rows"], color="#ff7f0e", marker=".", alpha=0.7, label="n_data_rows")
    ax2.set_ylabel("Number of data rows", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax1.set_title("Year-level total amount and row volume")
    ax1.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yearly_amount_and_volume.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(yearly["year"], yearly["amount_median"], marker="o", label="median")
    ax.plot(yearly["year"], yearly["amount_p90"], marker="o", label="p90")
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title("Year-level amount distribution summary")
    ax.set_xlabel("Year")
    ax.set_ylabel("Decimal pounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yearly_median_p90.png", dpi=180)
    plt.close(fig)

    cp_amount = cp[(cp["metric"] == "amount_sum") & (cp["is_change_point"])].copy()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(yearly["year"], yearly["amount_sum"], marker="o", color="#1f77b4")
    for _, r in cp_amount.iterrows():
        y = int(r["year"])
        val = float(yearly.loc[yearly["year"] == y, "amount_sum"].iloc[0])
        ax.axvline(y, color="red", linestyle="--", alpha=0.45)
        ax.scatter([y], [val], color="red", zorder=3)
    ax.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax.set_title("Detected change points on yearly total amount")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total amount")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "change_points_amount_sum.png", dpi=180)
    plt.close(fig)

    if not year_clusters.empty:
        fig, ax = plt.subplots(figsize=(13, 3.5))
        sns.scatterplot(data=year_clusters, x="year", y="cluster", hue="cluster", palette="tab10", ax=ax, s=55)
        ax.axvspan(1760, 1840, color="grey", alpha=0.15)
        ax.set_title("Year-level embedding clusters")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cluster")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "year_embedding_clusters_timeline.png", dpi=180)
        plt.close(fig)

    heat = era_top[era_top["rank"] <= 15].copy()
    if not heat.empty:
        p = heat.pivot_table(index="term", columns="era", values="relative_freq", aggfunc="max").fillna(0.0)
        p = p.loc[p.max(axis=1).sort_values(ascending=False).head(25).index]
        fig, ax = plt.subplots(figsize=(9, 10))
        sns.heatmap(p, cmap="YlGnBu", ax=ax)
        ax.set_title("Top terms by era (relative frequency)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "era_top_terms_heatmap.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(WORKBOOK)
    df.to_csv(OUT_DIR / "all_rows_with_amounts.csv", index=False)

    yearly = build_yearly_numeric(df)
    yearly.to_csv(OUT_DIR / "yearly_numeric_summary.csv", index=False)

    cp = detect_change_points(yearly)
    cp.to_csv(OUT_DIR / "yearly_change_points.csv", index=False)

    year_docs, tfidf, X_text = yearly_text_embeddings(df)
    y_terms = top_terms_by_year(year_docs, tfidf, X_text, top_n=15)
    y_terms.to_csv(OUT_DIR / "year_top_terms_tfidf.csv", index=False)

    year_clusters, cluster_profile = cluster_years(year_docs, X_text, yearly)
    year_clusters.to_csv(OUT_DIR / "year_embedding_clusters.csv", index=False)
    cluster_profile.to_csv(OUT_DIR / "year_cluster_profile_by_era.csv", index=False)

    era_top, emerging, era_term_long = era_term_analysis(df)
    era_top.to_csv(OUT_DIR / "era_top_terms.csv", index=False)
    emerging.to_csv(OUT_DIR / "era_emerging_terms.csv", index=False)
    era_term_long.to_csv(OUT_DIR / "era_term_frequencies_long.csv", index=False)

    accounts = recurring_accounts(df)
    accounts.to_csv(OUT_DIR / "recurring_descriptions_first_appearance.csv", index=False)

    header_first_app, header_yearly_counts, header_era_top = header_account_analysis(df)
    header_first_app.to_csv(OUT_DIR / "header_account_first_appearance.csv", index=False)
    header_yearly_counts.to_csv(OUT_DIR / "header_account_yearly_counts.csv", index=False)
    header_era_top.to_csv(OUT_DIR / "header_account_era_top_terms.csv", index=False)

    make_plots(yearly, cp, year_clusters, era_top)

    # concise markdown summary
    n_years = yearly["year"].nunique()
    cp_amount = cp[(cp["metric"] == "amount_sum") & (cp["is_change_point"])].copy().head(10)
    cp_text = "\n".join(
        [
            f"- {int(r.year)}: delta={r.delta:,.2f}, robust_z={r.abs_delta_robust_z:.2f}"
            for r in cp_amount.itertuples(index=False)
        ]
    )
    if not cp_text:
        cp_text = "- No strong change point on yearly amount_sum with robust_z >= 2.5"

    era_stats = (
        yearly.groupby("era", as_index=False)
        .agg(
            years=("year", "count"),
            mean_amount_sum=("amount_sum", "mean"),
            median_amount_median=("amount_median", "median"),
            mean_data_rows=("n_data_rows", "mean"),
        )
        .sort_values("era")
    )
    era_table = era_stats.to_string(index=False)

    summary = f"""# Reanalysis Summary (Year-level)

## Scope understanding applied
- Unit of analysis: **year** (not page)
- Workbook structure recognized: sheet name = year_page_image or year-year_page_image
- Row types used: header / entry / total
- Text & numeric analysis performed on entry + total rows
- Industrial Revolution window highlighted: 1760-1840

## Dataset overview
- Years covered: {n_years}
- Data rows (entry+total): {int((df['row_type'].isin(['entry','total'])).sum()):,}
- Header rows: {int((df['row_type'] == 'header').sum()):,}
- Unique normalized header/account texts: {int(header_first_app['description_norm'].nunique()):,}

## Era-level numeric summary
{era_table}

## Change points (yearly amount_sum)
{cp_text}

## Key outputs
- yearly_numeric_summary.csv
- yearly_change_points.csv
- year_top_terms_tfidf.csv
- year_embedding_clusters.csv
- year_cluster_profile_by_era.csv
- era_top_terms.csv
- era_emerging_terms.csv
- recurring_descriptions_first_appearance.csv
- header_account_first_appearance.csv
- header_account_yearly_counts.csv
- header_account_era_top_terms.csv

## Key figures
- yearly_amount_and_volume.png
- yearly_median_p90.png
- change_points_amount_sum.png
- year_embedding_clusters_timeline.png
- era_top_terms_heatmap.png
"""
    (OUT_DIR / "REANALYSIS_SUMMARY.md").write_text(summary, encoding="utf-8")

    print(f"Reanalysis complete. Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
