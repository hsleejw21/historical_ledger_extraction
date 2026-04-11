from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments" / "reports" / "ledger_clean"


def main() -> None:
    sns.set_theme(style="whitegrid")

    yearly = pd.read_csv(BASE / "yearly_numeric_summary.csv")
    header_yearly = pd.read_csv(BASE / "header_account_yearly_counts.csv")
    clusters = pd.read_csv(BASE / "year_embedding_clusters.csv")
    emerging = pd.read_csv(BASE / "era_emerging_terms.csv")

    # 1) Era-wise distribution of yearly total amount
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=yearly, x="era", y="amount_sum", ax=ax)
    sns.stripplot(data=yearly, x="era", y="amount_sum", ax=ax, color="black", alpha=0.25, size=2)
    ax.set_title("Yearly total amount by era")
    ax.set_xlabel("Era")
    ax.set_ylabel("Yearly total amount (decimal pounds)")
    fig.tight_layout()
    fig.savefig(BASE / "viz_era_amount_sum_boxplot.png", dpi=180)
    plt.close(fig)

    # 2) Header/account diversity timeline
    merged = yearly[["year", "era"]].merge(header_yearly, on="year", how="left")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(merged["year"], merged["n_unique_header_texts"], color="#1f77b4", marker="o", label="unique headers")
    ax1.set_ylabel("Unique header/account texts", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(merged["year"], merged["n_header_rows"], color="#ff7f0e", alpha=0.7, label="header rows")
    ax2.set_ylabel("Header rows", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    ax1.axvspan(1760, 1840, color="grey", alpha=0.15)
    ax1.set_title("Header/account diversity over time")
    ax1.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(BASE / "viz_header_diversity_timeline.png", dpi=180)
    plt.close(fig)

    # 3) Cluster composition by era
    pivot = (
        clusters.groupby(["era", "cluster"]).size().reset_index(name="n_years")
        .pivot(index="era", columns="cluster", values="n_years")
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("Year-embedding cluster counts by era")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Era")
    fig.tight_layout()
    fig.savefig(BASE / "viz_cluster_by_era_heatmap.png", dpi=180)
    plt.close(fig)

    # 4) Emerging terms (industrial vs pre, post vs pre)
    top_ind = (
        emerging[emerging["change"] == "industrial_vs_pre"]
        .sort_values("delta_ind_vs_pre", ascending=False)
        .head(15)
        .copy()
    )
    top_post = (
        emerging[emerging["change"] == "post_vs_pre"]
        .sort_values("delta_post_vs_pre", ascending=False)
        .head(15)
        .copy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    sns.barplot(data=top_ind, y="term", x="delta_ind_vs_pre", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Top emerging terms: industrial vs pre")
    axes[0].set_xlabel("Relative frequency increase")
    axes[0].set_ylabel("Term")

    sns.barplot(data=top_post, y="term", x="delta_post_vs_pre", ax=axes[1], color="#2ca02c")
    axes[1].set_title("Top emerging terms: post vs pre")
    axes[1].set_xlabel("Relative frequency increase")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(BASE / "viz_emerging_terms_bars.png", dpi=180)
    plt.close(fig)

    print(f"Visualization files saved in: {BASE}")


if __name__ == "__main__":
    main()
