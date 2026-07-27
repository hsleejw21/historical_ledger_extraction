"""
experiments/v6_loocv/loocv_prediction.py

Binary skip model prediction via LOOCV using visual features.

APPROACH:
  Reframe as a binary decision:
    "Can we SKIP extractor X for this page and still get near-oracle results?"
  
  For each extractor in {gemini, gpt, claude}:
    - Label each page: skip=1 (dropping X costs ≤ threshold) or skip=0 (keep X)
    - Train binary k-NN classifier on visual features
    - If all three say skip → use only remaining extractor (cheapest)
    - If uncertain → use all three (safest)
  
  WHY THIS WORKS BETTER:
    - Binary classification: much easier than 6-class
    - Each model has ~16 pages per class (balanced)
    - Directly maps to cost savings (skip 1 model = save 33% API calls)
    - More interpretable: "This page doesn't need Claude"

COST SAVINGS MAP:
  Skip 0 models: 3 API calls (full ensemble)
  Skip 1 model:  2 API calls (33% savings)
  Skip 2 models: 1 API call  (67% savings)  ← Aggressive but possible
  Skip 3 models: impossible (need ≥1 extractor)

Usage:
    python -m experiments.v6_loocv.loocv_prediction [--skip-threshold 0.01] [--k 5]
    
    --skip-threshold: Max score loss to allow skipping (default: 0.01)
    --k: Number of nearest neighbors (default: 5)
    --confidence-threshold: Min confidence to trust prediction (default: 0.65)
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import BASE_DIR, REPORT_DIR


# Paths
FEATURES_PATH = os.path.join(BASE_DIR, "data", "visual_features", "visual_features.json")
ORACLE_PATH = os.path.join(REPORT_DIR, "oracle_best_per_page.csv")
UNIFIED_PATH = os.path.join(REPORT_DIR, "unified_results.csv")
ORACLE_FALLBACK_PATH = os.path.join(BASE_DIR, "experiments", "v6_loocv", "outputs", "oracle_best_per_page.csv")
UNIFIED_FALLBACK_PATH = os.path.join(BASE_DIR, "experiments", "v6_loocv", "outputs", "unified_results.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "v6_loocv", "outputs")


# The three extractors we can potentially skip
EXTRACTORS = ['gemini', 'gpt', 'claude']

# Map extractor names to pipeline names containing their results
EXTRACTOR_TO_PIPELINE = {
    'gemini': 'v2_no_gemini',   # Result when gemini is skipped
    'gpt':    'v2_no_gpt',      # Result when gpt is skipped
    'claude': 'v2_no_claude',   # Result when claude is skipped
}

# Full ensemble pipeline (baseline: use all three)
FULL_ENSEMBLE_PIPELINE = 'v2_full'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def get_numeric_features(df, feature_cols):
    """Return numeric feature columns, excluding the page id."""
    return [
        col
        for col in feature_cols
        if col != 'page' and pd.api.types.is_numeric_dtype(df[col])
    ]

def load_data():
    """Load all necessary data."""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    features_df = pd.DataFrame.from_dict(features, orient='index')
    features_df.index.name = 'page'
    features_df = features_df.reset_index()

    oracle_path = ORACLE_PATH
    if not os.path.exists(oracle_path) and os.path.exists(ORACLE_FALLBACK_PATH):
        oracle_path = ORACLE_FALLBACK_PATH
    if not os.path.exists(oracle_path):
        raise FileNotFoundError(
            "Oracle file not found. Looked in: "
            f"{ORACLE_PATH} and {ORACLE_FALLBACK_PATH}"
        )

    unified_path = UNIFIED_PATH
    if not os.path.exists(unified_path) and os.path.exists(UNIFIED_FALLBACK_PATH):
        unified_path = UNIFIED_FALLBACK_PATH
    if not os.path.exists(unified_path):
        raise FileNotFoundError(
            "Unified results file not found. Looked in: "
            f"{UNIFIED_PATH} and {UNIFIED_FALLBACK_PATH}"
        )

    oracle_df = pd.read_csv(oracle_path)
    unified_df = pd.read_csv(unified_path)

    merged = features_df.merge(oracle_df, on='page', how='inner')
    
    print(f"[Loaded] {len(merged)} pages | "
          f"Features: {len(features_df.columns)-1} | "
          f"Pipelines in unified: {unified_df['pipeline'].nunique()}")
    
    return merged, unified_df, features_df.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Label building
# ─────────────────────────────────────────────────────────────────────────────

def build_skip_labels(merged_df, unified_df, skip_threshold=0.01):
    """
    For each page and each extractor, determine if it's safe to skip.
    
    skip_label[page][extractor] = 1  if skipping extractor costs ≤ skip_threshold
                                = 0  if skipping hurts more than skip_threshold

    Example:
        Full ensemble score for page 1700_7: 0.8500
        Score without gemini (v2_no_gemini): 0.8450
        Difference: 0.0050 ≤ threshold (0.01) → safe to skip gemini → label=1
    """
    labels = {}
    
    for _, row in merged_df.iterrows():
        page = row['page']
        labels[page] = {}
        
        # Get full ensemble score for this page
        full_results = unified_df[
            (unified_df['page'] == page) & 
            (unified_df['pipeline'] == FULL_ENSEMBLE_PIPELINE)
        ]
        
        if full_results.empty:
            # Try any v2_full variant
            full_results = unified_df[
                (unified_df['page'] == page) & 
                (unified_df['pipeline'].str.contains('v2_full', na=False))
            ]
        
        if full_results.empty:
            # Fall back to oracle score as reference
            full_score = row['oracle_best_combined']
        else:
            full_score = full_results['final_combined'].max()
        
        # For each extractor, check if skipping it is safe
        for extractor in EXTRACTORS:
            skip_pipeline = EXTRACTOR_TO_PIPELINE[extractor]
            
            skip_results = unified_df[
                (unified_df['page'] == page) & 
                (unified_df['pipeline'] == skip_pipeline)
            ]
            
            if skip_results.empty:
                # No data for this pipeline on this page → can't skip (unknown risk)
                labels[page][extractor] = 0
            else:
                skip_score = skip_results['final_combined'].max()
                score_drop = full_score - skip_score
                
                # Label 1 = safe to skip (score drop is small)
                labels[page][extractor] = 1 if score_drop <= skip_threshold else 0
    
    return labels


def print_label_distribution(labels, merged_df):
    """Show how many pages are safe to skip each extractor."""
    print("\n[Skip Label Distribution]")
    print(f"  (threshold: skip allowed if score drop <= threshold)")
    print()
    
    for extractor in EXTRACTORS:
        skip_count = sum(1 for page_labels in labels.values() 
                        if page_labels.get(extractor, 0) == 1)
        keep_count = len(labels) - skip_count
        print(f"  {extractor:8s}: safe_to_skip={skip_count:2d}/{len(labels)} "
              f"({skip_count/len(labels):.0%})  |  must_keep={keep_count:2d}/{len(labels)}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_correlations(merged_df, feature_cols):
    """Compute correlations between numeric features and oracle_best_combined."""
    numeric_features = get_numeric_features(merged_df, feature_cols)
    corrs = merged_df[numeric_features].corrwith(merged_df['oracle_best_combined'])
    return corrs.fillna(0)


def get_selected_features(feature_corrs, min_abs_correlation=0.15):
    """Select features with |r| >= threshold based on current oracle scores."""
    selected = feature_corrs[feature_corrs.abs() >= min_abs_correlation].index.tolist()

    print(f"\n[Feature Selection] Using {len(selected)} features (|r| >= {min_abs_correlation})")
    for f in feature_corrs.abs().sort_values(ascending=False).index:
        if f in selected:
            print(f"  {f:30s}: r = {feature_corrs[f]:+.3f}")

    return selected


def get_feature_weights(selected_features, feature_corrs):
    """Convert correlations to positive weights (normalize to sum to 1)."""
    abs_corrs = feature_corrs[selected_features].abs()
    total = abs_corrs.sum()
    if total == 0:
        uniform_weight = 1.0 / max(1, len(selected_features))
        return {f: uniform_weight for f in selected_features}
    return {f: v / total for f, v in abs_corrs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# LOOCV
# ─────────────────────────────────────────────────────────────────────────────

def run_loocv_binary(merged_df, unified_df, labels, feature_cols,
                     k=5, confidence_threshold=0.65, min_abs_corr=0.15):
    """
    Binary LOOCV: for each extractor, predict skip/keep.
    
    For each page i (leave-one-out):
        - Train k-NN on other 32 pages
        - Predict: skip_gemini? skip_gpt? skip_claude?
        - Combine predictions → decide which extractors to use
        - Measure: cost saved + performance preserved
    """
    print("\n" + "="*60)
    print(f"BINARY LOOCV - k={k}, confidence_threshold={confidence_threshold}")
    print("="*60)
    
    # Select and scale features
    feature_corrs = compute_feature_correlations(merged_df, feature_cols)
    selected_features = get_selected_features(feature_corrs, min_abs_corr)
    
    if not selected_features:
        print("[Warning] No features selected! Falling back to all numeric features.")
        selected_features = get_numeric_features(merged_df, feature_cols)
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(merged_df[selected_features].fillna(0))
    X_df = pd.DataFrame(X_all, columns=selected_features, index=merged_df.index)
    
    # Apply feature weights (scale by |correlation|)
    weights = get_feature_weights(selected_features, feature_corrs)
    for feat, w in weights.items():
        X_df[feat] *= w
    
    results = []
    
    # Get full ensemble scores for reference
    full_ensemble_scores = {}
    for page in merged_df['page']:
        full_res = unified_df[
            (unified_df['page'] == page) & 
            (unified_df['pipeline'] == FULL_ENSEMBLE_PIPELINE)
        ]
        if not full_res.empty:
            full_ensemble_scores[page] = full_res['final_combined'].max()
        else:
            # Use oracle score as proxy
            oracle_row = merged_df[merged_df['page'] == page]
            if not oracle_row.empty:
                full_ensemble_scores[page] = oracle_row['oracle_best_combined'].values[0]
    
    # Pre-build axis2 lookup: {page: {pipeline: {col: val}}}
    _ax2_cols = ["final_axis1", "final_axis2", "axis2_match", "axis2_similarity", "axis2_fraction"]
    axis2_lookup = {}
    for page in merged_df['page']:
        page_rows = unified_df[unified_df['page'] == page]
        axis2_lookup[page] = {}
        for _, row in page_rows.iterrows():
            axis2_lookup[page][row['pipeline']] = {c: row.get(c, float("nan")) for c in _ax2_cols}

    print(f"\n[Running LOOCV on {len(merged_df)} pages...]")

    for test_idx, test_row in merged_df.iterrows():
        test_page = test_row['page']
        
        # Train/test split
        train_mask = merged_df['page'] != test_page
        train_df = merged_df[train_mask]
        
        X_train = X_df[train_mask]
        X_test = X_df.loc[[test_idx]]
        
        # For each extractor: binary classification (skip or keep)
        skip_predictions = {}
        skip_confidences = {}
        
        for extractor in EXTRACTORS:
            # Build binary labels for training set
            y_train = [labels[p][extractor] for p in train_df['page']]
            
            # Check if we have both classes in training set
            if len(set(y_train)) < 2:
                # Only one class in training → can't learn, default to keep
                skip_predictions[extractor] = 0
                skip_confidences[extractor] = 0.5
                continue
            
            # Train k-NN
            knn = KNeighborsClassifier(n_neighbors=min(k, sum(y_train), sum(1-v for v in y_train)))
            knn.fit(X_train, y_train)
            
            # Predict with probability
            proba = knn.predict_proba(X_test)[0]
            classes = knn.classes_
            
            # Get probability of skip=1
            if 1 in classes:
                skip_idx = list(classes).index(1)
                skip_prob = proba[skip_idx]
            else:
                skip_prob = 0.0
            
            predicted_skip = 1 if skip_prob >= 0.5 else 0
            confidence = max(proba)  # Confidence = max probability
            
            skip_predictions[extractor] = predicted_skip
            skip_confidences[extractor] = confidence
        
        # Decide final strategy based on predictions + confidence
        extractors_to_use = []
        extractors_skipped = []
        
        for extractor in EXTRACTORS:
            if (skip_predictions[extractor] == 1 and 
                skip_confidences[extractor] >= confidence_threshold):
                extractors_skipped.append(extractor)
            else:
                extractors_to_use.append(extractor)
        
        # Safety: always use at least one extractor
        if not extractors_to_use:
            extractors_to_use = EXTRACTORS[:]
            extractors_skipped = []
        
        # Map to pipeline name
        n_skipped = len(extractors_skipped)
        api_calls = len(extractors_to_use)
        
        if n_skipped == 0:
            pipeline_used = FULL_ENSEMBLE_PIPELINE
        elif n_skipped == 1:
            pipeline_used = EXTRACTOR_TO_PIPELINE[extractors_skipped[0]]
        else:
            # 2+ models skipped → no single-extractor pipeline exists in data.
            # Fall back to full ensemble to avoid using a pipeline that still
            # runs one of the "skipped" extractors.
            pipeline_used = FULL_ENSEMBLE_PIPELINE
            extractors_skipped = []
            api_calls = 3
        
        # Get actual score for chosen strategy
        chosen_result = unified_df[
            (unified_df['page'] == test_page) & 
            (unified_df['pipeline'] == pipeline_used)
        ]
        
        if not chosen_result.empty:
            final_score = chosen_result['final_combined'].max()
        else:
            # Pipeline not available → fall back to ensemble
            pipeline_used = FULL_ENSEMBLE_PIPELINE
            api_calls = 3
            final_score = full_ensemble_scores.get(test_page, test_row['oracle_best_combined'])

        # Axis2 components for chosen pipeline
        ax2_data = axis2_lookup.get(test_page, {}).get(pipeline_used, {c: float("nan") for c in _ax2_cols})

        # Oracle score (upper bound)
        oracle_score = test_row['oracle_best_combined']
        full_score = full_ensemble_scores.get(test_page, oracle_score)
        
        # Per-extractor accuracy
        per_extractor_correct = {}
        for extractor in EXTRACTORS:
            true_label = labels[test_page][extractor]
            pred_label = skip_predictions[extractor]
            per_extractor_correct[f'{extractor}_correct'] = (true_label == pred_label)
            per_extractor_correct[f'{extractor}_true'] = true_label
            per_extractor_correct[f'{extractor}_pred'] = pred_label
            per_extractor_correct[f'{extractor}_conf'] = skip_confidences[extractor]
        
        results.append({
            'page': test_page,
            'oracle_score': oracle_score,
            'full_ensemble_score': full_score,
            'final_score': final_score,
            'score_vs_oracle': final_score / oracle_score if oracle_score > 0 else 1.0,
            'score_vs_ensemble': final_score / full_score if full_score > 0 else 1.0,
            'pipeline_used': pipeline_used,
            'api_calls': api_calls,
            'n_skipped': n_skipped,
            'extractors_skipped': ','.join(extractors_skipped) if extractors_skipped else 'none',
            # Axis2 components
            'final_axis1':     ax2_data.get('final_axis1',     float('nan')),
            'final_axis2':     ax2_data.get('final_axis2',     float('nan')),
            'axis2_match':     ax2_data.get('axis2_match',     float('nan')),
            'axis2_similarity':ax2_data.get('axis2_similarity',float('nan')),
            'axis2_fraction':  ax2_data.get('axis2_fraction',  float('nan')),
            **per_extractor_correct
        })
        
        # Console output
        skip_str = f"skip={','.join(extractors_skipped)}" if extractors_skipped else "full_ensemble"
        print(f"  {test_page:15s}: {skip_str:30s} | "
              f"score={final_score:.4f} | "
              f"calls={api_calls}")
    
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(results_df, skip_threshold):
    """Print comprehensive evaluation of binary LOOCV results."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    n = len(results_df)
    oracle_avg = results_df['oracle_score'].mean()
    full_avg   = results_df['full_ensemble_score'].mean()
    final_avg  = results_df['final_score'].mean()
    
    print(f"\n[1] Per-Extractor Skip Accuracy:")
    for extractor in EXTRACTORS:
        correct_col = f'{extractor}_correct'
        if correct_col in results_df.columns:
            acc = results_df[correct_col].mean()
            skip_true = results_df[f'{extractor}_true'].sum()
            skip_pred = results_df[f'{extractor}_pred'].sum()
            print(f"  {extractor:8s}: accuracy={acc:.1%}  "
                  f"(true_skips={skip_true:2d}, predicted_skips={skip_pred:2d})")
    
    overall_correct = sum(
        results_df[f'{e}_correct'].sum() for e in EXTRACTORS
        if f'{e}_correct' in results_df.columns
    )
    overall_total = n * len(EXTRACTORS)
    print(f"\n  Overall binary accuracy: {overall_correct}/{overall_total} "
          f"({overall_correct/overall_total:.1%})")
    
    print(f"\n[2] Performance Preservation:")
    print(f"  Oracle avg:        {oracle_avg:.4f}")
    print(f"  Full ensemble avg: {full_avg:.4f}  ({full_avg/oracle_avg:.1%} of oracle)")
    print(f"  Final avg:         {final_avg:.4f}  ({final_avg/oracle_avg:.1%} of oracle)")
    print(f"  vs full ensemble:  {final_avg/full_avg:.1%} of ensemble")
    
    print(f"\n[3] Cost Reduction:")
    baseline_calls = n * 3  # Always use all 3 extractors
    actual_calls = results_df['api_calls'].sum()
    cost_reduction = (baseline_calls - actual_calls) / baseline_calls
    avg_calls = results_df['api_calls'].mean()
    
    print(f"  Baseline (3 extractors × {n} pages): {baseline_calls} API calls")
    print(f"  Actual:                               {actual_calls} API calls")
    print(f"  Reduction: {cost_reduction:.1%}  ({baseline_calls - actual_calls} calls saved)")
    print(f"  Avg calls per page: {avg_calls:.2f}")
    
    print(f"\n[4] Skip Decision Distribution:")
    skip_dist = results_df['n_skipped'].value_counts().sort_index()
    for n_skip, count in skip_dist.items():
        label = {0: 'Full ensemble', 1: 'Skip 1 model', 2: 'Skip 2 models'}.get(n_skip, f'Skip {n_skip}')
        calls = 3 - n_skip
        print(f"  {label}: {count:2d} pages ({count/n:.0%}) -> {calls} API calls/page")
    
    print(f"\n[5] Score Distribution by Strategy:")
    for n_skip in sorted(results_df['n_skipped'].unique()):
        subset = results_df[results_df['n_skipped'] == n_skip]
        if not subset.empty:
            label = {0: 'Full ensemble (0 skipped)', 
                     1: 'Skip 1 model', 
                     2: 'Skip 2 models'}.get(n_skip, f'Skip {n_skip}')
            print(f"  {label}: avg_score={subset['final_score'].mean():.4f}, "
                  f"n={len(subset)}")
    
    print(f"\n[6] Pages Where Skipping Hurt Performance:")
    threshold_drop = skip_threshold * 2  # Flag pages where we lost more than 2× threshold
    hurt = results_df[results_df['score_vs_ensemble'] < (1 - threshold_drop)]
    if hurt.empty:
        print(f"  None! All skip decisions were within acceptable range.")
    else:
        for _, row in hurt.iterrows():
            drop = (1 - row['score_vs_ensemble']) * 100
            print(f"  {row['page']:15s}: dropped {drop:.1f}% below ensemble "
                  f"(skipped: {row['extractors_skipped']})")
    
    # Axis2 component averages for chosen pipeline
    axis1_avg     = results_df['final_axis1'].mean()      if 'final_axis1'      in results_df.columns else float('nan')
    axis2_avg     = results_df['final_axis2'].mean()      if 'final_axis2'      in results_df.columns else float('nan')
    ax2_match_avg = results_df['axis2_match'].mean()      if 'axis2_match'      in results_df.columns else float('nan')
    ax2_sim_avg   = results_df['axis2_similarity'].mean() if 'axis2_similarity' in results_df.columns else float('nan')
    ax2_frac_avg  = results_df['axis2_fraction'].mean()   if 'axis2_fraction'   in results_df.columns else float('nan')

    # Score distribution stats
    min_score = results_df['final_score'].min()
    score_std = results_df['final_score'].std()

    return {
        'skip_threshold': skip_threshold,
        'overall_binary_accuracy': overall_correct / overall_total,
        'performance_preservation_vs_oracle': final_avg / oracle_avg,
        'performance_preservation_vs_ensemble': final_avg / full_avg,
        'cost_reduction': cost_reduction,
        'avg_api_calls': avg_calls,
        'oracle_avg': oracle_avg,
        'ensemble_avg': full_avg,
        'final_avg': final_avg,
        'avg_axis1':      round(axis1_avg, 4),
        'avg_axis2':      round(axis2_avg, 4),
        'avg_axis2_match':round(ax2_match_avg, 4),
        'avg_axis2_sim':  round(ax2_sim_avg, 4),
        'avg_axis2_frac': round(ax2_frac_avg, 4),
        # Score distribution
        'min_score':  round(min_score, 4),
        'score_std':  round(score_std, 4),
        # Balanced score matching clip scripts: 0.3*acc + 0.4*pres_ens + 0.3*cost
        'balanced_score': (0.3 * (overall_correct / overall_total) +
                           0.4 * (final_avg / full_avg) +
                           0.3 * cost_reduction),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(merged_df, unified_df, feature_cols):
    """Sweep hyperparameters and save sweep_summary.csv matching clip script format."""
    from itertools import product as iproduct

    k_values    = [3, 5, 7]
    thresholds  = [0.005, 0.010, 0.020, 0.030]
    conf_values = [0.55, 0.60, 0.65, 0.70]

    total = len(k_values) * len(thresholds) * len(conf_values)
    print(f"\n{'='*60}")
    print(f"VISUAL BINARY SWEEP ({total} configs)")
    print(f"{'='*60}")

    rows = []
    for skip_thr, conf, k in iproduct(thresholds, conf_values, k_values):
        labels = build_skip_labels(merged_df, unified_df, skip_threshold=skip_thr)
        results_df = run_loocv_binary(
            merged_df, unified_df, labels, feature_cols,
            k=k, confidence_threshold=conf, min_abs_corr=0.15,
        )
        metrics = evaluate(results_df, skip_thr)
        metrics['k'] = k
        metrics['confidence_threshold'] = conf
        rows.append(metrics)

    sweep_df = pd.DataFrame(rows).sort_values("balanced_score", ascending=False)

    print(f"\n{'='*80}")
    print("SWEEP RESULTS (sorted by balanced_score)")
    print(f"{'='*80}")
    print(f"  {'skip':>6} | {'conf':>5} | {'k':>3} | {'bin_acc':>8} | "
          f"{'pres_ens':>9} | {'cost_r':>6} | {'balanced':>9}")
    print("-"*80)
    for _, row in sweep_df.head(15).iterrows():
        print(f"  {row['skip_threshold']:>6.3f} | {row['confidence_threshold']:>5.2f} | "
              f"{row['k']:>3} | {row['overall_binary_accuracy']:>8.1%} | "
              f"{row['performance_preservation_vs_ensemble']:>9.1%} | "
              f"{row['cost_reduction']:>6.1%} | {row['balanced_score']:>9.4f}")

    best = sweep_df.iloc[0]
    print(f"\n[Best Configuration — Visual Binary]")
    print(f"  skip_threshold:     {best['skip_threshold']}")
    print(f"  confidence:         {best['confidence_threshold']}")
    print(f"  k:                  {best['k']}")
    print(f"  Binary accuracy:    {best['overall_binary_accuracy']:.1%}")
    print(f"  Preservation vs ensemble: {best['performance_preservation_vs_ensemble']:.1%}")
    print(f"  Cost reduction:     {best['cost_reduction']:.1%}")
    print(f"  Final avg score:    {best['final_avg']:.4f}")
    print(f"  Balanced score:     {best['balanced_score']:.4f}")

    out_path = os.path.join(OUTPUT_DIR, "sweep_summary.csv")
    sweep_df.to_csv(out_path, index=False)
    print(f"\n[Saved] sweep_summary.csv")
    return sweep_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary skip-model LOOCV prediction")
    parser.add_argument("--skip-threshold", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--min-abs-corr", type=float, default=0.15)
    parser.add_argument("--sweep", action="store_true",
                        help="Run full hyperparameter sweep")
    args = parser.parse_args()

    merged_df, unified_df, feature_cols = load_data()

    if args.sweep:
        run_sweep(merged_df, unified_df, feature_cols)
    else:
        print(f"\nConfig: skip_threshold={args.skip_threshold}, k={args.k}, "
              f"confidence={args.confidence_threshold}")

        labels = build_skip_labels(merged_df, unified_df, skip_threshold=args.skip_threshold)
        print_label_distribution(labels, merged_df)

        results_df = run_loocv_binary(
            merged_df, unified_df, labels, feature_cols,
            k=args.k,
            confidence_threshold=args.confidence_threshold,
            min_abs_corr=args.min_abs_corr,
        )
        metrics = evaluate(results_df, args.skip_threshold)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR,
                                f"loocv_v2_k{args.k}_thr{args.skip_threshold}_"
                                f"conf{args.confidence_threshold}.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\n[Saved] {os.path.basename(out_path)}")

        print("\n" + "="*60)
        print("Summary:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
        print("="*60 + "\n")