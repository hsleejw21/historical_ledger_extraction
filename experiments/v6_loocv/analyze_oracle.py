"""
experiments/v6_loocv/analyze_oracle.py

Analyzes the unified results to identify the "oracle best" model/ensemble per page.

This script answers:
  1. For each page, which single model performed best?
  2. How often does the ensemble (supervisor) beat the best single model?
  3. What is the performance gap between oracle and ensemble?
  4. Are there visual/structural patterns in which models excel where?

Outputs:
  - oracle_best_per_page.csv: Best config per page with detailed scores
  - oracle_analysis_summary.txt: Readable summary with insights

Usage:
    python -m experiments.v6_loocv.analyze_oracle
"""
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# V6 analysis outputs directory (same as aggregate_reports.py)
V6_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(V6_OUTPUT_DIR, exist_ok=True)


def analyze_oracle():
    """Main oracle analysis function."""
    print("\n" + "="*60)
    print("ORACLE ANALYSIS - Best Model Per Page")
    print("="*60)

    # Load unified results (from v6 outputs)
    unified_path = os.path.join(V6_OUTPUT_DIR, "unified_results.csv")
    
    if not os.path.exists(unified_path):
        print(f"\n[Error] Unified results not found: {unified_path}")
        print("Run this first:")
        print("  python -m experiments.v6_loocv.rescore_with_expanded_metrics")
        print("  python -m experiments.v6_loocv.aggregate_reports")
        return
    
    df = pd.read_csv(unified_path)
    print(f"\n[Loaded] {len(df)} records from unified_results.csv")
    print(f"  Pages: {df['page'].nunique()}")
    print(f"  Pipelines: {df['pipeline'].unique().tolist()}")
    
    # Identify oracle best per page
    print("\n[1/4] Identifying oracle best per page...")
    
    oracle_records = []
    
    for page in sorted(df['page'].unique()):
        page_df = df[df['page'] == page]
        
        if 'final_combined' not in page_df.columns:
            print(f"  [Warning] {page} - missing final_combined scores")
            continue
        
        # Find best overall
        best_idx = page_df['final_combined'].idxmax()
        best_row = page_df.loc[best_idx]
        
        # Find best single model (exclude ensembles/supervisors)
        single_model_df = page_df[~page_df['pipeline'].str.contains('v2|v3', na=False)]
        
        best_single_combined = np.nan
        best_single_model = "N/A"
        
        if not single_model_df.empty:
            best_single_idx = single_model_df['final_combined'].idxmax()
            best_single_row = single_model_df.loc[best_single_idx]
            best_single_combined = best_single_row['final_combined']
            best_single_model = best_single_row.get('model_combo', best_single_row.get('config', 'unknown'))
        
        # Find best ensemble (v2/v3 supervisors)
        ensemble_df = page_df[page_df['pipeline'].str.contains('v2|v3', na=False)]
        
        best_ensemble_combined = np.nan
        best_ensemble_config = "N/A"
        
        if not ensemble_df.empty:
            best_ensemble_idx = ensemble_df['final_combined'].idxmax()
            best_ensemble_row = ensemble_df.loc[best_ensemble_idx]
            best_ensemble_combined = best_ensemble_row['final_combined']
            best_ensemble_config = best_ensemble_row.get('model_combo', best_ensemble_row.get('config', 'unknown'))
        
        # Build oracle record
        oracle_record = {
            'page': page,
            'oracle_best_combined': best_row['final_combined'],
            'oracle_best_axis1': best_row.get('final_axis1', np.nan),
            'oracle_best_axis2': best_row.get('final_axis2', np.nan),
            'oracle_best_axis2_match': best_row.get('axis2_match', np.nan),
            'oracle_best_axis2_similarity': best_row.get('axis2_similarity', np.nan),
            'oracle_best_axis2_fraction': best_row.get('axis2_fraction', np.nan),
            'oracle_best_pipeline': best_row['pipeline'],
            'oracle_best_config': best_row.get('model_combo', best_row.get('config', 'unknown')),
            
            'best_single_combined': best_single_combined,
            'best_single_model': best_single_model,
            
            'best_ensemble_combined': best_ensemble_combined,
            'best_ensemble_config': best_ensemble_config,
            
            'ensemble_vs_single_gap': best_ensemble_combined - best_single_combined if not np.isnan(best_ensemble_combined) and not np.isnan(best_single_combined) else np.nan,
        }
        
        oracle_records.append(oracle_record)
    
    oracle_df = pd.DataFrame(oracle_records)
    
    # Save oracle best per page
    oracle_output_path = os.path.join(V6_OUTPUT_DIR, "oracle_best_per_page.csv")
    oracle_df.to_csv(oracle_output_path, index=False)
    print(f"  [OK] Oracle best per page saved: oracle_best_per_page.csv")
    
    # Analyze patterns
    print("\n[2/4] Analyzing patterns...")
    
    # How often is ensemble best?
    oracle_is_ensemble = oracle_df['oracle_best_pipeline'].str.contains('v2|v3', na=False).sum()
    oracle_is_single = len(oracle_df) - oracle_is_ensemble
    
    print(f"\n  Oracle best is:")
    print(f"    Ensemble (v2/v3): {oracle_is_ensemble}/{len(oracle_df)} pages ({oracle_is_ensemble/len(oracle_df)*100:.1f}%)")
    print(f"    Single model (v1): {oracle_is_single}/{len(oracle_df)} pages ({oracle_is_single/len(oracle_df)*100:.1f}%)")
    
    # Average performance gaps
    avg_oracle = oracle_df['oracle_best_combined'].mean()
    avg_best_single = oracle_df['best_single_combined'].mean()
    avg_best_ensemble = oracle_df['best_ensemble_combined'].mean()
    avg_gap = oracle_df['ensemble_vs_single_gap'].mean()
    
    print(f"\n  Average scores:")
    print(f"    Oracle best:      {avg_oracle:.4f}")
    print(f"    Best single:      {avg_best_single:.4f}")
    print(f"    Best ensemble:    {avg_best_ensemble:.4f}")
    print(f"    Ensemble vs single gap: {avg_gap:+.4f}")
    
    # Axis2 component breakdown
    if 'oracle_best_axis2_match' in oracle_df.columns:
        print(f"\n  Oracle axis2 components (avg):")
        print(f"    axis2_match:      {oracle_df['oracle_best_axis2_match'].mean():.4f}")
        print(f"    axis2_similarity: {oracle_df['oracle_best_axis2_similarity'].mean():.4f}")
        print(f"    axis2_fraction:   {oracle_df['oracle_best_axis2_fraction'].mean():.4f}")
    
    # Pipeline distribution
    print(f"\n[3/4] Oracle pipeline distribution:")
    pipeline_counts = oracle_df['oracle_best_pipeline'].value_counts()
    for pipeline, count in pipeline_counts.items():
        print(f"    {pipeline:20s}: {count:2d} pages ({count/len(oracle_df)*100:.1f}%)")
    
    # Generate summary report
    print("\n[4/4] Generating summary report...")
    
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("ORACLE ANALYSIS SUMMARY")
    summary_lines.append("="*60)
    summary_lines.append(f"\nTotal pages analyzed: {len(oracle_df)}")
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("BEST MODEL TYPE PER PAGE")
    summary_lines.append("="*60)
    summary_lines.append(f"Ensemble (v2/v3):    {oracle_is_ensemble:2d} pages ({oracle_is_ensemble/len(oracle_df)*100:.1f}%)")
    summary_lines.append(f"Single model (v1):   {oracle_is_single:2d} pages ({oracle_is_single/len(oracle_df)*100:.1f}%)")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("AVERAGE PERFORMANCE")
    summary_lines.append("="*60)
    summary_lines.append(f"Oracle best:         {avg_oracle:.4f}")
    summary_lines.append(f"Best single model:   {avg_best_single:.4f}")
    summary_lines.append(f"Best ensemble:       {avg_best_ensemble:.4f}")
    summary_lines.append(f"Ensemble advantage:  {avg_gap:+.4f}")
    
    if 'oracle_best_axis2_match' in oracle_df.columns:
        summary_lines.append(f"\n{'='*60}")
        summary_lines.append("AXIS2 COMPONENT BREAKDOWN (Oracle)")
        summary_lines.append("="*60)
        summary_lines.append(f"axis2_match:         {oracle_df['oracle_best_axis2_match'].mean():.4f}")
        summary_lines.append(f"axis2_similarity:    {oracle_df['oracle_best_axis2_similarity'].mean():.4f}")
        summary_lines.append(f"axis2_fraction:      {oracle_df['oracle_best_axis2_fraction'].mean():.4f}")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("ORACLE PIPELINE DISTRIBUTION")
    summary_lines.append("="*60)
    for pipeline, count in pipeline_counts.items():
        summary_lines.append(f"{pipeline:20s}: {count:2d} pages ({count/len(oracle_df)*100:.1f}%)")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("TOP 10 BEST PERFORMING PAGES")
    summary_lines.append("="*60)
    top_pages = oracle_df.nlargest(10, 'oracle_best_combined')
    for _, row in top_pages.iterrows():
        summary_lines.append(f"{row['page']:15s}: {row['oracle_best_combined']:.4f}  [{row['oracle_best_pipeline']}]")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("BOTTOM 10 WORST PERFORMING PAGES")
    summary_lines.append("="*60)
    bottom_pages = oracle_df.nsmallest(10, 'oracle_best_combined')
    for _, row in bottom_pages.iterrows():
        summary_lines.append(f"{row['page']:15s}: {row['oracle_best_combined']:.4f}  [{row['oracle_best_pipeline']}]")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("INSIGHTS FOR V6 ADAPTIVE ROUTING")
    summary_lines.append("="*60)
    summary_lines.append(f"1. Oracle avg ({avg_oracle:.4f}) sets the upper bound for v6 prediction")
    summary_lines.append(f"2. If v6 always picked best single model, avg would be {avg_best_single:.4f}")
    summary_lines.append(f"3. Performance preservation target: >={avg_oracle * 0.95:.4f} (95% of oracle)")
    summary_lines.append(f"4. Current SOTA (v2_no_claude): 0.8385")
    summary_lines.append(f"5. Pages where ensemble helps most: analyze 'ensemble_vs_single_gap' column")
    summary_lines.append(f"6. Next: Extract visual features + LOOCV prediction")
    
    summary_text = "\n".join(summary_lines)
    
    summary_output_path = os.path.join(V6_OUTPUT_DIR, "oracle_analysis_summary.txt")
    with open(summary_output_path, "w") as f:
        f.write(summary_text)
    
    print(f"  [OK] Summary report saved: oracle_analysis_summary.txt")
    
    # Print summary to console
    print("\n" + summary_text)


if __name__ == "__main__":
    analyze_oracle()