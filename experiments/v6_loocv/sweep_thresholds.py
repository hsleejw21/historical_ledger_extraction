"""
experiments/v6_loocv/sweep_thresholds.py

Sweeps skip_threshold and confidence_threshold to find optimal trade-off
between cost reduction and performance preservation.

Usage:
    python -m experiments.v6_loocv.sweep_thresholds
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import BASE_DIR, REPORT_DIR
from experiments.v6_loocv.loocv_prediction import (
    load_data, build_skip_labels, run_loocv_binary, evaluate
)

OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "v6_loocv", "outputs")


def run_sweep():
    """Sweep over key hyperparameters."""
    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP")
    print("="*60)
    
    # Load data once
    merged_df, unified_df, feature_cols = load_data()
    
    # Define sweep grid
    skip_thresholds   = [0.005, 0.01, 0.02, 0.03]
    confidence_values = [0.55,  0.60, 0.65, 0.70]
    k_values          = [3, 5, 7]
    
    summary = []
    
    total = len(skip_thresholds) * len(confidence_values) * len(k_values)
    run_num = 0
    
    for skip_thr in skip_thresholds:
        labels = build_skip_labels(merged_df, unified_df, skip_threshold=skip_thr)
        
        for conf_thr in confidence_values:
            for k in k_values:
                run_num += 1
                print(f"\n[Run {run_num}/{total}] skip={skip_thr}, conf={conf_thr}, k={k}")
                
                results_df = run_loocv_binary(
                    merged_df, unified_df, labels, feature_cols,
                    k=k,
                    confidence_threshold=conf_thr,
                    min_abs_corr=0.15,
                )
                
                metrics = evaluate(results_df, skip_thr)
                
                summary.append({
                    'skip_threshold': skip_thr,
                    'confidence_threshold': conf_thr,
                    'k': k,
                    **metrics,
                })
    
    summary_df = pd.DataFrame(summary)
    
    # Compute balanced score: 0.5 * preservation + 0.3 * cost_reduction + 0.2 * binary_accuracy
    summary_df['balanced_score'] = (
        0.5  * summary_df['performance_preservation_vs_ensemble'] +
        0.3  * summary_df['cost_reduction'] +
        0.2  * summary_df['overall_binary_accuracy']
    )
    
    return summary_df


def print_sweep_results(summary_df):
    """Print formatted sweep results."""
    print("\n" + "="*80)
    print("SWEEP RESULTS (sorted by balanced_score)")
    print("="*80)
    
    summary_sorted = summary_df.sort_values('balanced_score', ascending=False)
    
    header = (f"{'skip':>6s} | {'conf':>5s} | {'k':>3s} | "
              f"{'bin_acc':>7s} | {'pres_ens':>8s} | {'cost↓':>6s} | {'balanced':>8s}")
    print(header)
    print("-" * 80)
    
    for _, row in summary_sorted.head(12).iterrows():
        print(f"{row['skip_threshold']:>6.3f} | {row['confidence_threshold']:>5.2f} | "
              f"{int(row['k']):>3d} | "
              f"{row['overall_binary_accuracy']:>7.1%} | "
              f"{row['performance_preservation_vs_ensemble']:>8.1%} | "
              f"{row['cost_reduction']:>6.1%} | "
              f"{row['balanced_score']:>8.4f}")
    
    print("\n[Best Configuration]")
    best = summary_sorted.iloc[0]
    print(f"  skip_threshold:     {best['skip_threshold']}")
    print(f"  confidence_threshold: {best['confidence_threshold']}")
    print(f"  k:                  {int(best['k'])}")
    print(f"  Binary accuracy:    {best['overall_binary_accuracy']:.1%}")
    print(f"  Preservation vs ensemble: {best['performance_preservation_vs_ensemble']:.1%}")
    print(f"  Cost reduction:     {best['cost_reduction']:.1%}")
    print(f"  Balanced score:     {best['balanced_score']:.4f}")


def plot_sweep_results(summary_df):
    """Visualize sweep results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Hyperparameter Sweep: Cost vs. Performance Trade-off", 
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(
        (summary_df['overall_binary_accuracy'] - summary_df['overall_binary_accuracy'].min()) /
        (summary_df['overall_binary_accuracy'].max() - summary_df['overall_binary_accuracy'].min() + 1e-8)
    )
    
    # 1. Cost reduction vs. preservation (colored by binary accuracy)
    ax = axes[0]
    sc = ax.scatter(
        summary_df['cost_reduction'],
        summary_df['performance_preservation_vs_ensemble'],
        c=summary_df['overall_binary_accuracy'],
        s=100,
        cmap='viridis',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8
    )
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.7, label='99% ensemble')
    ax.axhline(y=0.97, color='orange', linestyle='--', alpha=0.7, label='97% ensemble')
    ax.axvline(x=0.30, color='red', linestyle='--', alpha=0.7, label='30% cost target')
    ax.set_xlabel('Cost Reduction', fontsize=11)
    ax.set_ylabel('Performance Preservation\n(vs full ensemble)', fontsize=11)
    ax.set_title('Cost vs. Performance Trade-off', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='Binary Accuracy')
    
    # 2. Effect of skip_threshold
    ax = axes[1]
    for k in sorted(summary_df['k'].unique()):
        subset = summary_df[summary_df['k'] == k].groupby('skip_threshold').mean(numeric_only=True)
        ax.plot(subset.index, subset['cost_reduction'], 
                marker='o', label=f'k={k}', linewidth=2)
    ax.set_xlabel('Skip Threshold', fontsize=11)
    ax.set_ylabel('Avg Cost Reduction', fontsize=11)
    ax.set_title('Cost Reduction by Skip Threshold', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # 3. Effect of confidence_threshold on preservation
    ax = axes[2]
    for k in sorted(summary_df['k'].unique()):
        subset = summary_df[summary_df['k'] == k].groupby('confidence_threshold').mean(numeric_only=True)
        ax.plot(subset.index, subset['performance_preservation_vs_ensemble'], 
                marker='s', label=f'k={k}', linewidth=2)
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Confidence Threshold', fontsize=11)
    ax.set_ylabel('Avg Preservation (vs ensemble)', fontsize=11)
    ax.set_title('Preservation by Confidence Threshold', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "sweep_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] sweep_results.png")
    plt.close()


if __name__ == "__main__":
    summary_df = run_sweep()
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "sweep_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Saved] sweep_summary.csv")
    
    # Print results
    print_sweep_results(summary_df)
    
    # Plot
    plot_sweep_results(summary_df)
    
    print("\n" + "="*60)
    print("Sweep complete! Review sweep_results.png for visualization.")
    print("Then run best config with loocv_prediction.py")
    print("="*60 + "\n")