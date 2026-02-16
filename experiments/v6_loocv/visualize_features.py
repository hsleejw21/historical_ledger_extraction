"""
experiments/v6_loocv/visualize_features.py

Visualizes extracted visual features and analyzes their correlation with
model performance (oracle scores).

Generates:
  1. Feature distribution plots
  2. Correlation heatmap (features vs oracle scores)
  3. Feature importance ranking
  4. Outlier detection plots

Outputs saved to: experiments/v6_loocv/outputs/visualizations/

Usage:
    python -m experiments.v6_loocv.visualize_features [--save-plots]
    
    --save-plots: Save plots to disk (default: show only)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import BASE_DIR, REPORT_DIR


# Paths
FEATURES_PATH = os.path.join(BASE_DIR, "data", "visual_features", "visual_features.json")
ORACLE_PATH = os.path.join(REPORT_DIR, "oracle_best_per_page.csv")
ORACLE_FALLBACK_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "v6_loocv",
    "outputs",
    "oracle_best_per_page.csv",
)
VIZ_OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "v6_loocv", "outputs", "visualizations")


def get_numeric_features(df, feature_cols):
    """Return numeric feature columns, excluding the page id."""
    return [
        col
        for col in feature_cols
        if col != 'page' and pd.api.types.is_numeric_dtype(df[col])
    ]


def load_data():
    """Load visual features and oracle scores."""
    # Load visual features
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    
    features_df = pd.DataFrame.from_dict(features, orient='index')
    features_df.index.name = 'page'
    features_df = features_df.reset_index()
    
    # Load oracle scores
    oracle_path = ORACLE_PATH
    if not os.path.exists(oracle_path) and os.path.exists(ORACLE_FALLBACK_PATH):
        oracle_path = ORACLE_FALLBACK_PATH

    if not os.path.exists(oracle_path):
        raise FileNotFoundError(
            "Oracle file not found. Looked in: "
            f"{ORACLE_PATH} and {ORACLE_FALLBACK_PATH}"
        )

    oracle_df = pd.read_csv(oracle_path)
    
    # Merge
    merged = features_df.merge(oracle_df, on='page', how='inner')
    
    print(f"[Loaded] {len(merged)} pages with features and oracle scores")
    
    return merged, features_df.columns.tolist()


def plot_feature_distributions(df, feature_cols, save_path=None):
    """Plot distributions of all features."""
    print("\n[1/5] Plotting feature distributions...")
    
    # Select only numeric features
    numeric_features = get_numeric_features(df, feature_cols)
    
    n_features = len(numeric_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_features):
        ax = axes[i]
        
        # Histogram
        ax.hist(df[feature].dropna(), bins=20, edgecolor='black', alpha=0.7)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean line
        mean_val = df[feature].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "feature_distributions.png"), dpi=150, bbox_inches='tight')
        print(f"  [Saved] feature_distributions.png")
    else:
        plt.show()
    
    plt.close()


def plot_correlation_with_oracle(df, feature_cols, save_path=None):
    """Plot correlation between features and oracle scores."""
    print("\n[2/5] Analyzing correlation with oracle scores...")
    
    # Select numeric features
    numeric_features = get_numeric_features(df, feature_cols)
    
    # Compute correlations with oracle scores
    oracle_cols = ['oracle_best_combined', 'oracle_best_axis1', 'oracle_best_axis2']
    
    correlations = {}
    for oracle_col in oracle_cols:
        if oracle_col in df.columns:
            corrs = df[numeric_features].corrwith(df[oracle_col])
            correlations[oracle_col] = corrs
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('oracle_best_combined', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(numeric_features) * 0.3)))
    
    sns.heatmap(
        corr_df,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title('Feature Correlation with Oracle Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Oracle Score Type')
    ax.set_ylabel('Visual Features')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "correlation_heatmap.png"), dpi=150, bbox_inches='tight')
        print(f"  [Saved] correlation_heatmap.png")
    else:
        plt.show()
    
    plt.close()
    
    # Print top correlations
    print("\n  Top 10 features correlated with oracle_best_combined:")
    top_features = corr_df['oracle_best_combined'].abs().sort_values(ascending=False).head(10)
    for feature, corr_val in top_features.items():
        print(f"    {feature:30s}: {corr_val:+.3f}")
    
    return corr_df


def plot_feature_vs_oracle_scatter(df, feature_cols, save_path=None):
    """Scatter plots of top features vs oracle scores."""
    print("\n[3/5] Plotting top features vs oracle scores...")
    
    # Select numeric features
    numeric_features = get_numeric_features(df, feature_cols)
    
    # Get top 6 features by correlation with oracle_best_combined
    corrs = df[numeric_features].corrwith(df['oracle_best_combined']).abs().sort_values(ascending=False)
    top_6_features = corrs.head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_6_features):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(df[feature], df['oracle_best_combined'], alpha=0.6, s=50)
        
        # Fit line
        z = np.polyfit(df[feature].dropna(), df.loc[df[feature].notna(), 'oracle_best_combined'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Labels
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Oracle Best Combined', fontsize=10)
        ax.set_title(f'{feature}\n(corr: {corrs[feature]:+.3f})', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "top_features_scatter.png"), dpi=150, bbox_inches='tight')
        print(f"  [Saved] top_features_scatter.png")
    else:
        plt.show()
    
    plt.close()


def plot_pairwise_feature_correlation(df, feature_cols, save_path=None):
    """Plot pairwise correlation heatmap of all features."""
    print("\n[4/5] Computing pairwise feature correlations...")
    
    # Select numeric features
    numeric_features = get_numeric_features(df, feature_cols)
    
    # Compute correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))
    
    sns.heatmap(
        corr_matrix,
        annot=False,  # Too many features for annotation
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title('Pairwise Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "pairwise_correlation.png"), dpi=150, bbox_inches='tight')
        print(f"  [Saved] pairwise_correlation.png")
    else:
        plt.show()
    
    plt.close()
    
    # Identify highly correlated feature pairs (potential redundancy)
    print("\n  Highly correlated feature pairs (|r| > 0.9):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        for feat1, feat2, corr_val in high_corr_pairs:
            print(f"    {feat1:30s} <-> {feat2:30s}: {corr_val:+.3f}")
    else:
        print("    None found (good - features are independent)")


def plot_outlier_detection(df, feature_cols, save_path=None):
    """Identify and visualize outlier pages based on features."""
    print("\n[5/5] Detecting outlier pages...")
    
    # Select numeric features
    numeric_features = get_numeric_features(df, feature_cols)
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[numeric_features])
    
    # Compute distance from centroid (simple outlier score)
    centroid = features_scaled.mean(axis=0)
    distances = np.sqrt(((features_scaled - centroid) ** 2).sum(axis=1))
    
    df['outlier_score'] = distances
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Outlier score distribution
    ax1.hist(df['outlier_score'], bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Outlier Score (Distance from Centroid)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Outlier Score Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Mark threshold (e.g., 95th percentile)
    threshold = np.percentile(df['outlier_score'], 95)
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'95th percentile: {threshold:.2f}')
    ax1.legend()
    
    # 2. Outlier score vs oracle performance
    ax2.scatter(df['outlier_score'], df['oracle_best_combined'], alpha=0.6, s=50)
    ax2.set_xlabel('Outlier Score', fontsize=12)
    ax2.set_ylabel('Oracle Best Combined', fontsize=12)
    ax2.set_title('Outlier Score vs Performance', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Annotate outliers
    outliers = df[df['outlier_score'] > threshold]
    for _, row in outliers.iterrows():
        ax2.annotate(
            row['page'],
            xy=(row['outlier_score'], row['oracle_best_combined']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "outlier_detection.png"), dpi=150, bbox_inches='tight')
        print(f"  [Saved] outlier_detection.png")
    else:
        plt.show()
    
    plt.close()
    
    # Print outliers
    print(f"\n  Outlier pages (top 5% unusual):")
    outliers_sorted = df.nlargest(max(1, int(len(df) * 0.05)), 'outlier_score')
    for _, row in outliers_sorted.iterrows():
        print(f"    {row['page']:15s}: score={row['outlier_score']:.2f}, oracle={row['oracle_best_combined']:.4f}")


def generate_summary_report(df, corr_df, save_path=None):
    """Generate text summary of feature analysis."""
    print("\n[Generating summary report...]")
    
    lines = []
    lines.append("="*60)
    lines.append("VISUAL FEATURES ANALYSIS SUMMARY")
    lines.append("="*60)
    
    lines.append(f"\nDataset:")
    lines.append(f"  Total pages: {len(df)}")
    lines.append(f"  Features extracted: {len(get_numeric_features(df, df.columns))}")
    
    lines.append(f"\nOracle performance range:")
    lines.append(f"  Min: {df['oracle_best_combined'].min():.4f}")
    lines.append(f"  Max: {df['oracle_best_combined'].max():.4f}")
    lines.append(f"  Mean: {df['oracle_best_combined'].mean():.4f}")
    lines.append(f"  Std: {df['oracle_best_combined'].std():.4f}")
    
    lines.append(f"\nTop 10 features correlated with oracle performance:")
    top_features = corr_df['oracle_best_combined'].abs().sort_values(ascending=False).head(10)
    for i, (feature, corr_val) in enumerate(top_features.items(), 1):
        lines.append(f"  {i:2d}. {feature:35s}: {corr_val:+.3f}")
    
    lines.append(f"\nFeature importance insights:")
    lines.append(f"  - Features with |correlation| > 0.3 may be useful predictors")
    lines.append(f"  - Features with |correlation| < 0.1 may be noise")
    lines.append(f"  - Highly correlated feature pairs (|r| > 0.9) indicate redundancy")
    
    lines.append(f"\nNext steps:")
    lines.append(f"  1. Review visualizations in experiments/v6_loocv/outputs/visualizations/")
    lines.append(f"  2. Consider feature selection based on correlation analysis")
    lines.append(f"  3. Proceed to Phase 3: LOOCV prediction")
    
    lines.append("\n" + "="*60)
    
    summary_text = "\n".join(lines)
    
    if save_path:
        summary_path = os.path.join(save_path, "feature_analysis_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text)
        print(f"  [Saved] feature_analysis_summary.txt")
    
    print("\n" + summary_text)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize extracted features")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to disk (default: show interactively)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if saving
    save_path = None
    if args.save_plots:
        os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)
        save_path = VIZ_OUTPUT_DIR
        print(f"\n[Saving plots to] {VIZ_OUTPUT_DIR}")
    
    # Load data
    df, feature_cols = load_data()
    
    # Generate visualizations
    plot_feature_distributions(df, feature_cols, save_path)
    corr_df = plot_correlation_with_oracle(df, feature_cols, save_path)
    plot_feature_vs_oracle_scatter(df, feature_cols, save_path)
    plot_pairwise_feature_correlation(df, feature_cols, save_path)
    plot_outlier_detection(df, feature_cols, save_path)
    
    # Generate summary report
    generate_summary_report(df, corr_df, save_path)
    
    if args.save_plots:
        print("\n" + "="*60)
        print(f"All visualizations saved to: {VIZ_OUTPUT_DIR}")
        print("="*60 + "\n")