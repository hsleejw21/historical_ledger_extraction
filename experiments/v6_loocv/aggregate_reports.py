"""
experiments/v6_loocv/aggregate_reports.py

Merges all v2 experiment reports (v2 full + v2 ablations) into a single
unified CSV with consistent column naming and pipeline tagging.

This gives us a complete view of every v2 extraction attempt, which will be used for:
  1. Identifying oracle best configuration per page
  2. Training data for the adaptive routing predictor
  3. Analysis of extraction performance patterns

Usage:
    python -m experiments.v6_loocv.aggregate_reports
"""
import os
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import REPORT_DIR

# V6 analysis outputs go into their own directory to keep reports/ clean
V6_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(V6_OUTPUT_DIR, exist_ok=True)


def load_report(filename: str, pipeline_tag: str) -> pd.DataFrame:
    """Load a single experiment report and add pipeline metadata."""
    path = os.path.join(REPORT_DIR, filename)
    
    if not os.path.exists(path):
        print(f"  [Warning] Not found: {filename}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df['pipeline'] = pipeline_tag
    
    print(f"  [OK] {filename}: {len(df)} rows")
    return df


def _rename_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect which column naming convention is used and rename to final_*.

    Rescored reports (from rescore_with_expanded_metrics.py) use:
        axis1_score, axis2_score, combined_score
    Original v2 reports (from run_experiment.py) use:
        supervisor_axis1, supervisor_axis2, supervisor_combined
    """
    # Try each convention in order of likelihood
    for src_axis1, src_axis2, src_combined in [
        ('axis1_score', 'axis2_score', 'combined_score'),           # rescored
        ('supervisor_axis1', 'supervisor_axis2', 'supervisor_combined'),  # original v2
    ]:
        if src_axis1 in df.columns:
            df = df.rename(columns={
                src_axis1: 'final_axis1',
                src_axis2: 'final_axis2',
                src_combined: 'final_combined',
            })
            break

    return df


def normalize_v2_columns(df: pd.DataFrame, pipeline_tag: str) -> pd.DataFrame:
    """Rename v2 columns to unified naming convention."""
    if df.empty:
        return df

    df = _rename_final_scores(df)

    # Add model_combo column
    if 'extractor_models' in df.columns and 'supervisor_model' in df.columns:
        df['model_combo'] = df.apply(
            lambda r: f"{r['extractor_models']}->{r['supervisor_model']}",
            axis=1
        )
    elif 'config' in df.columns:
        df['model_combo'] = df['config']

    return df


def aggregate_all_reports():
    """Main aggregation function."""
    print("\n[Aggregating All Experiment Reports]")
    print("=" * 60)
    
    all_dfs = []
    
    # --- V2 Reports ---
    # The rescored v2 report contains ALL configs (full + ablations) with
    # expanded axis2 metrics.  We tag each row with a descriptive pipeline
    # label based on the extractor_models combo.
    print("\nLoading v2 reports...")

    v2_all = load_report("experiment_results_v2.csv", "v2")
    if not v2_all.empty:
        v2_all = normalize_v2_columns(v2_all, "v2")

        # Map extractor combos to pipeline tags
        ABLATION_TAGS = {
            # 3-model (full ensemble)
            "gemini-flash|gpt-5-mini|claude-haiku": "v2_full",
            # 2-model ablations
            "gemini-flash|gpt-5-mini": "v2_no_claude",
            "gpt-5-mini|claude-haiku": "v2_no_gemini",
            "gemini-flash|claude-haiku": "v2_no_gpt",
        }

        if 'extractor_models' in v2_all.columns:
            v2_all['pipeline'] = v2_all['extractor_models'].map(ABLATION_TAGS).fillna('v2_other')
        else:
            v2_all['pipeline'] = 'v2_full'

        for tag in v2_all['pipeline'].unique():
            count = (v2_all['pipeline'] == tag).sum()
            print(f"    {tag}: {count} rows")

        all_dfs.append(v2_all)
    
    # --- Merge All ---
    if not all_dfs:
        print("\n[Error] No reports found!")
        return
    
    print(f"\n[Merging {len(all_dfs)} dataframes...]")
    unified_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # Reorder columns for readability
    priority_cols = [
        'page', 'pipeline', 'model_combo',
        'final_axis1', 'final_axis2', 'final_combined',
        'axis2_match', 'axis2_similarity', 'axis2_fraction',  # New expanded metrics
    ]
    
    existing_priority = [c for c in priority_cols if c in unified_df.columns]
    other_cols = [c for c in unified_df.columns if c not in existing_priority]
    unified_df = unified_df[existing_priority + other_cols]
    
    # Save unified report
    output_path = os.path.join(V6_OUTPUT_DIR, "unified_results.csv")
    unified_df.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Unified report saved:")
    print(f"  -> {output_path}")
    print(f"  Total records: {len(unified_df)}")
    print(f"  Unique pages: {unified_df['page'].nunique()}")
    print(f"  Pipelines: {unified_df['pipeline'].unique().tolist()}")
    
    # Print summary stats
    print(f"\n[Summary Statistics]")
    if 'final_combined' in unified_df.columns:
        print(f"  Overall avg final_combined: {unified_df['final_combined'].mean():.4f}")
        print(f"\nBy pipeline:")
        for pipeline in unified_df['pipeline'].unique():
            subset = unified_df[unified_df['pipeline'] == pipeline]
            avg_score = subset['final_combined'].mean()
            print(f"    {pipeline:20s}: {avg_score:.4f} ({len(subset):3d} records)")


if __name__ == "__main__":
    aggregate_all_reports()