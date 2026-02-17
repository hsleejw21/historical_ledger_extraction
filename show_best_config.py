import pandas as pd
import os

base_dir = r"c:\Users\hslee\Documents\RA\HAI Lab\Historical Ledger Extraction\historical_ledger_extraction"
summary_path = os.path.join(base_dir, "experiments", "v6_loocv", "outputs", "sweep_summary.csv")

# Read sweep summary
summary_df = pd.read_csv(summary_path)

# Find best config by balanced_score
best_idx = summary_df['balanced_score'].idxmax()
best = summary_df.loc[best_idx]

print("=" * 80)
print("BEST HYPERPARAMETER CONFIGURATION FROM SWEEP")
print("=" * 80)
print()
print("Selected Configuration:")
print(f"  skip_threshold       = {best['skip_threshold']:.3f}")
print(f"  confidence_threshold = {best['confidence_threshold']:.2f}")
print(f"  k (neighbors)        = {int(best['k'])}")
print()
print("Evaluation Metrics:")
print(f"  ✓ Binary Classification Accuracy              {best['overall_binary_accuracy']:.1%}")
print(f"    (skip decision accuracy across all pages)")
print()
print(f"  ✓ Oracle Preservation                         {best['performance_preservation_vs_oracle']:.1%}")
print(f"    (maintains oracle performance level)")
print()
print(f"  ✓ Ensemble Preservation                       {best['performance_preservation_vs_ensemble']:.1%}")
print(f"    (maintains our all-extractors ensemble performance)")
print()
print(f"  ✓ Cost Reduction                              {best['cost_reduction']:.1%}")
print(f"    (fraction of pages where we skip an extractor)")
print()
print(f"  ✓ Avg API Calls per Page                      {best['avg_api_calls']:.2f} / 3.0")
print(f"    (originally 3 extractors per page)")
print()
print(f"  ✓ Balanced Score (weighted metric)            {best['balanced_score']:.6f}")
print()
print("=" * 80)
print()
print("Key Insights:")
print(f"  • This config achieves the highest balanced score across all 48 tested configs")
print(f"  • Skips extractors with confidence < {best['confidence_threshold']:.2f} using k-NN (k={int(best['k'])})")
print(f"  • Reduces API calls by {best['cost_reduction']:.1%} (saves {3 - best['avg_api_calls']:.2f} calls/page)")
print(f"  • Maintains {best['performance_preservation_vs_oracle']:.1%} of oracle performance")
print(f"  • Maintains {best['performance_preservation_vs_ensemble']:.1%} of ensemble performance")
print()
print("=" * 80)
