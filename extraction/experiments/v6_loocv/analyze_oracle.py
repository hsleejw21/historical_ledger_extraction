"""
experiments/v6_loocv/analyze_oracle.py

Analyzes the unified results to identify the "oracle best" v2 configuration per page.

This script answers:
  1. For each page, which v2 configuration (full ensemble or ablation) performed best?
  2. How often does the full ensemble (all 3 extractors) beat the ablations?
  3. What is the performance of the oracle best configuration?
  4. Are there visual/structural patterns in which configurations excel where?

Outputs:
  - oracle_best_per_page.csv: Best v2 config per page with detailed scores
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
    
    # All pipelines should be v2 variants (v2_full, v2_no_gemini, v2_no_gpt, v2_no_claude)
    if not all(df['pipeline'].str.startswith('v2', na=False)):
        print("\n[Warning] Found non-v2 pipelines in unified results!")
        print(f"  Expected only v2_* pipelines, but found: {df['pipeline'].unique().tolist()}")
    
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
        
        # Find best single model (exclude full ensemble, use ablations)
        ablation_df = page_df[page_df['pipeline'] != 'v2_full']
        
        best_ablation_combined = np.nan
        best_ablation_config = "N/A"
        
        if not ablation_df.empty:
            best_ablation_idx = ablation_df['final_combined'].idxmax()
            best_ablation_row = ablation_df.loc[best_ablation_idx]
            best_ablation_combined = best_ablation_row['final_combined']
            best_ablation_config = best_ablation_row.get('model_combo', best_ablation_row.get('config', 'unknown'))
        
        # Find best full ensemble (v2_full)
        ensemble_df = page_df[page_df['pipeline'] == 'v2_full']
        
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
            
            'best_ablation_combined': best_ablation_combined,
            'best_ablation_config': best_ablation_config,
            
            'best_full_ensemble_combined': best_ensemble_combined,
            'best_full_ensemble_config': best_ensemble_config,
            
            'full_ensemble_vs_ablation_gap': best_ensemble_combined - best_ablation_combined if not np.isnan(best_ensemble_combined) and not np.isnan(best_ablation_combined) else np.nan,
        }
        
        oracle_records.append(oracle_record)
    
    oracle_df = pd.DataFrame(oracle_records)
    
    # Save oracle best per page
    oracle_output_path = os.path.join(V6_OUTPUT_DIR, "oracle_best_per_page.csv")
    oracle_df.to_csv(oracle_output_path, index=False)
    print(f"  [OK] Oracle best per page saved: oracle_best_per_page.csv")
    
    # Analyze patterns
    print("\n[2/4] Analyzing patterns...")
    
    # How often is full ensemble best?
    oracle_is_full_ensemble = (oracle_df['oracle_best_pipeline'] == 'v2_full').sum()
    oracle_is_ablation = len(oracle_df) - oracle_is_full_ensemble
    
    print(f"\n  Oracle best is:")
    print(f"    Full ensemble (v2_full):   {oracle_is_full_ensemble}/{len(oracle_df)} pages ({oracle_is_full_ensemble/len(oracle_df)*100:.1f}%)")
    print(f"    Ablation (v2_no_*):        {oracle_is_ablation}/{len(oracle_df)} pages ({oracle_is_ablation/len(oracle_df)*100:.1f}%)")
    
    # Average performance gaps
    avg_oracle = oracle_df['oracle_best_combined'].mean()
    avg_best_ablation = oracle_df['best_ablation_combined'].mean()
    avg_best_full = oracle_df['best_full_ensemble_combined'].mean()
    avg_gap = oracle_df['full_ensemble_vs_ablation_gap'].mean()
    
    print(f"\n  Average scores:")
    print(f"    Oracle best:         {avg_oracle:.4f}")
    print(f"    Best ablation:       {avg_best_ablation:.4f}")
    print(f"    Full ensemble:       {avg_best_full:.4f}")
    print(f"    Full vs ablation gap: {avg_gap:+.4f}")
    
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
    summary_lines.append("BEST V2 CONFIGURATION TYPE PER PAGE")
    summary_lines.append("="*60)
    summary_lines.append(f"Full ensemble (v2_full): {oracle_is_full_ensemble:2d} pages ({oracle_is_full_ensemble/len(oracle_df)*100:.1f}%)")
    summary_lines.append(f"Ablation (v2_no_*):      {oracle_is_ablation:2d} pages ({oracle_is_ablation/len(oracle_df)*100:.1f}%)")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("AVERAGE PERFORMANCE")
    summary_lines.append("="*60)
    summary_lines.append(f"Oracle best:         {avg_oracle:.4f}")
    summary_lines.append(f"Best ablation:       {avg_best_ablation:.4f}")
    summary_lines.append(f"Full ensemble:       {avg_best_full:.4f}")
    summary_lines.append(f"Full vs ablation gap: {avg_gap:+.4f}")
    
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
    summary_lines.append(f"1. Oracle avg ({avg_oracle:.4f}) sets the upper bound for v6 skip-model")
    summary_lines.append(f"2. Best ablation avg ({avg_best_ablation:.4f}) is the baseline if extractors are skipped")
    summary_lines.append(f"3. Full ensemble advantage: {avg_gap:+.4f} (useful on {oracle_is_full_ensemble} pages)")
    summary_lines.append(f"4. Performance preservation target: >={avg_oracle * 0.95:.4f} (95% of oracle)")
    summary_lines.append(f"5. Pages where full ensemble helps most: analyze 'full_ensemble_vs_ablation_gap' column")
    summary_lines.append(f"6. Next: Extract visual features + binary skip-model LOOCV prediction")
    
    summary_text = "\n".join(summary_lines)
    
    summary_output_path = os.path.join(V6_OUTPUT_DIR, "oracle_analysis_summary.txt")
    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"  [OK] Summary report saved: oracle_analysis_summary.txt")
    
    # Print summary to console
    print("\n" + summary_text)


if __name__ == "__main__":
    analyze_oracle()