"""
experiments/v6_loocv/rescore_with_expanded_metrics.py

Re-scores ALL cached extraction results using the updated scorer with expanded
axis2 sub-components (axis2_match, axis2_similarity, axis2_fraction).

This script:
  1. Scans experiments/results/v1/, v2/, v3/ for all cached extraction JSONs
  2. Loads corresponding ground truth
  3. Re-scores with the new scorer
  4. Generates updated reports with expanded metrics
  5. Preserves original reports as backups

Usage:
    python -m experiments.v6_loocv.rescore_with_expanded_metrics [--pipeline v1|v2|v3|all]
    
    --pipeline: Which pipeline to re-score (default: all)
"""
import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import RESULTS_DIR, GT_DIR, REPORT_DIR
from src.evaluation.scorer import score_page


def _parse_page_and_config(filename_stem: str, stage_keyword: str):
    """
    Robustly split a filename stem into (page_name, config_key) by finding
    the stage keyword (corrector, supervisor, etc.).

    Example:
        '1759-1760_3_corrector_gemini-flash_gemini-pro_gpt-5-mini'
        -> ('1759-1760_3', 'corrector_gemini-flash_gemini-pro_gpt-5-mini')
    """
    marker = f"_{stage_keyword}_"
    idx = filename_stem.find(marker)
    if idx == -1:
        return None, None
    page_name = filename_stem[:idx]
    config_key = filename_stem[idx + 1:]  # strip leading '_'
    return page_name, config_key


def discover_cached_results(pipeline_version: str):
    """
    Scan the results directory for cached extraction outputs.

    Returns:
        dict: {page_name: {config_key: result_path}}

    Example:
        {
            "1700_7": {
                "v2_gemini-flash_gpt-5-mini_supervisor-gemini-flash": "path/to/result.json"
            }
        }
    """
    results_dir = os.path.join(RESULTS_DIR, pipeline_version)

    if not os.path.exists(results_dir):
        print(f"  [Warning] Results directory not found: {results_dir}")
        return {}

    results = {}

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue

        stem = filename.replace(".json", "")

        if pipeline_version == "v1":
            # Look for corrector outputs (final stage)
            if "_corrector_" in filename and "retry" not in filename:
                page_name, config_key = _parse_page_and_config(stem, "corrector")
                if page_name is None:
                    continue

                if page_name not in results:
                    results[page_name] = {}
                results[page_name][config_key] = os.path.join(results_dir, filename)

        elif pipeline_version in ("v2", "v3"):
            # Look for supervisor outputs (final stage)
            if "_supervisor_" in filename:
                page_name, config_key = _parse_page_and_config(stem, "supervisor")
                if page_name is None:
                    continue

                if page_name not in results:
                    results[page_name] = {}
                results[page_name][config_key] = os.path.join(results_dir, filename)

    return results


def load_ground_truth(page_name: str):
    """Load ground truth for a page."""
    gt_path = os.path.join(GT_DIR, f"{page_name}.json")
    
    if not os.path.exists(gt_path):
        return None
    
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def rescore_pipeline(pipeline_version: str):
    """Re-score all cached results for a specific pipeline."""
    print(f"\n{'='*60}")
    print(f"Re-scoring {pipeline_version.upper()} with expanded axis2 metrics")
    print(f"{'='*60}")
    
    # Discover all cached results
    print(f"\n[1/3] Discovering cached results in experiments/results/{pipeline_version}/...")
    results_map = discover_cached_results(pipeline_version)
    
    if not results_map:
        print(f"  [Warning] No cached results found for {pipeline_version}")
        return
    
    print(f"  Found {len(results_map)} pages with cached results")
    
    # Re-score each result
    print(f"\n[2/3] Re-scoring with updated metrics...")
    records = []
    
    for page_name, configs in sorted(results_map.items()):
        gt = load_ground_truth(page_name)
        
        if gt is None:
            print(f"  [Skip] {page_name} - no ground truth")
            continue
        
        print(f"  [Page] {page_name} - {len(configs)} configs")
        
        for config_key, result_path in configs.items():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    pred = json.load(f)
                
                # Score with updated scorer
                scores = score_page(pred, gt)
                
                # Build record
                record = {
                    'page': page_name,
                    'config': config_key,
                    'pipeline': pipeline_version,
                    **scores  # Includes all axis1, axis2 components, and combined
                }
                
                # Add model metadata
                if pipeline_version == "v1":
                    # Parse model names from config_key:
                    #   corrector_{structurer}_{extractor}_{corrector}
                    config_parts = config_key.split("_")
                    if len(config_parts) >= 4 and config_parts[0] == "corrector":
                        record['structurer_model'] = config_parts[1]
                        record['extractor_model'] = config_parts[2]
                        record['corrector_model'] = config_parts[3]
                    else:
                        # Fallback to _meta
                        meta = pred.get('_meta', {})
                        record['structurer_model'] = meta.get('structurer_model', '')
                        record['extractor_model'] = meta.get('extractor_model', '')
                        record['corrector_model'] = meta.get('model', '')
                elif pipeline_version in ("v2", "v3"):
                    meta = pred.get('_meta', {})
                    record['supervisor_model'] = meta.get('supervisor_model', '')
                    record['extractor_models'] = '|'.join(meta.get('candidate_models', []))
                
                records.append(record)
                
            except Exception as e:
                print(f"    [Error] {config_key}: {e}")
                continue
    
    if not records:
        print(f"  [Warning] No records generated for {pipeline_version}")
        return
    
    # Save updated report
    print(f"\n[3/3] Saving updated report...")
    df = pd.DataFrame(records)
    
    # Backup original report if it exists
    original_report_name = f"experiment_results_{pipeline_version}.csv"
    original_report_path = os.path.join(REPORT_DIR, original_report_name)
    
    if os.path.exists(original_report_path):
        backup_path = os.path.join(REPORT_DIR, f"experiment_results_{pipeline_version}_backup.csv")
        os.replace(original_report_path, backup_path)
        print(f"  [Backup] Original report saved to {original_report_name}_backup.csv")
    
    # Save new report
    df.to_csv(original_report_path, index=False)
    print(f"  [OK] Updated report: {original_report_name}")
    print(f"       Total records: {len(df)}")
    
    # Print summary
    print(f"\n[Summary]")
    if 'combined_score' in df.columns:
        print(f"  Avg combined_score: {df['combined_score'].mean():.4f}")
    if 'axis2_match' in df.columns:
        print(f"  Avg axis2_match:    {df['axis2_match'].mean():.4f}")
    if 'axis2_similarity' in df.columns:
        print(f"  Avg axis2_similarity: {df['axis2_similarity'].mean():.4f}")
    if 'axis2_fraction' in df.columns:
        print(f"  Avg axis2_fraction: {df['axis2_fraction'].mean():.4f}")


def rescore_all():
    """Re-score all pipelines."""
    for pipeline in ["v1", "v2", "v3"]:
        rescore_pipeline(pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-score cached results with expanded axis2 metrics")
    parser.add_argument(
        "--pipeline",
        choices=["v1", "v2", "v3", "all"],
        default="all",
        help="Which pipeline to re-score (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.pipeline == "all":
        rescore_all()
    else:
        rescore_pipeline(args.pipeline)
    
    print("\n" + "="*60)
    print("Re-scoring complete! Next step:")
    print("  python -m experiments.v6_loocv.aggregate_reports")
    print("="*60 + "\n")