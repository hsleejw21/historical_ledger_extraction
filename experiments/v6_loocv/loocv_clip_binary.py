"""
experiments/v6_loocv/loocv_clip_binary.py

Phase 4A: LOOCV with CLIP embeddings — binary skip task.

IDENTICAL evaluation logic to loocv_prediction.py (visual features k-NN),
but replaces the 26 visual features with 512-dim CLIP embeddings as the
similarity measure.

This gives a direct apples-to-apples comparison:
  Phase 3:  visual features → k-NN → binary skip decisions
  Phase 4A: CLIP embeddings → k-NN → binary skip decisions (same task)

Key difference: cosine similarity instead of Euclidean distance on
standardized features, because CLIP embeddings are already L2-normalized.

Output:
  experiments/v6_loocv/outputs/loocv_clip_binary_k{k}_thr{t}_conf{c}.csv

Usage:
    python -m experiments.v6_loocv.loocv_clip_binary [--k 5] [--skip-threshold 0.01] [--conf 0.65]
    python -m experiments.v6_loocv.loocv_clip_binary --sweep   # test multiple configs
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import BASE_DIR, REPORT_DIR

# ── Paths ──────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "visual_features", "clip_embeddings.json")
V6_OUTPUT_DIR   = os.path.join(BASE_DIR, "experiments", "v6_loocv", "outputs")
ORACLE_PATH     = os.path.join(V6_OUTPUT_DIR, "oracle_best_per_page.csv")
UNIFIED_PATH    = os.path.join(V6_OUTPUT_DIR, "unified_results.csv")
OUTPUT_DIR      = V6_OUTPUT_DIR

EXTRACTORS      = ["gemini", "gpt", "claude"]
PIPELINE_MAP    = {
    # maps which extractors were skipped → pipeline name in unified_results
    frozenset():                    "v2_full",
    frozenset(["claude"]):          "v2_no_claude",
    frozenset(["gpt"]):             "v2_no_gpt",
    frozenset(["gemini"]):          "v2_no_gemini",
    # Note: v6_loocv only has 2-model ablations (no single-extractor pipelines)
}


# ── Data loading ───────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"CLIP embeddings not found: {EMBEDDINGS_PATH}\n"
            f"Run this first: python -m experiments.v6_loocv.extract_clip_embeddings"
        )
    
    with open(EMBEDDINGS_PATH) as f:
        raw = json.load(f)

    # Build arrays: pages × 512
    pages  = sorted(raw.keys())
    matrix = np.array([raw[p] for p in pages], dtype=np.float32)   # already L2-normalized

    if not os.path.exists(ORACLE_PATH):
        raise FileNotFoundError(f"Oracle file not found: {ORACLE_PATH}")
    if not os.path.exists(UNIFIED_PATH):
        raise FileNotFoundError(f"Unified results not found: {UNIFIED_PATH}")
    
    oracle_df  = pd.read_csv(ORACLE_PATH)
    unified_df = pd.read_csv(UNIFIED_PATH)

    # Align oracle to page order
    oracle_df = oracle_df.set_index("page").reindex(pages).reset_index()

    print(f"[Loaded] {len(pages)} pages | embedding dim={matrix.shape[1]}")
    print(f"         Oracle pages matched: {oracle_df['page'].notna().sum()}/{len(pages)}")

    return pages, matrix, oracle_df, unified_df


# ── Skip label builder ─────────────────────────────────────────────────────
def build_skip_labels(pages, oracle_df, unified_df, skip_threshold):
    """
    For each (page, extractor) pair, compute binary label:
      1 = safe to skip (removing this extractor costs ≤ skip_threshold)
      0 = must keep
    Also stores the full-ensemble score and per-ablation scores for evaluation.
    """
    labels = {p: {} for p in pages}
    scores = {}   # page → {pipeline → score}

    for page in pages:
        page_rows = unified_df[unified_df["page"] == page]
        page_scores = {}
        for _, row in page_rows.iterrows():
            page_scores[row["pipeline"]] = row["final_combined"]
        scores[page] = page_scores

        full_score = page_scores.get("v2_full", np.nan)

        for ext in EXTRACTORS:
            # pipeline that drops this extractor
            ablation_key = f"v2_no_{ext}"
            ablation_score = page_scores.get(ablation_key, np.nan)

            if np.isnan(full_score) or np.isnan(ablation_score):
                labels[page][ext] = 0   # no data → assume must keep
            else:
                drop = full_score - ablation_score
                labels[page][ext] = 1 if drop <= skip_threshold else 0

    return labels, scores


# ── k-NN with cosine similarity ────────────────────────────────────────────
def cosine_knn(matrix, query_idx, k):
    """
    Find k nearest neighbors of matrix[query_idx] using cosine similarity.
    Since embeddings are L2-normalized, cosine similarity = dot product.
    Returns (neighbor_indices, similarities) — query index excluded.
    """
    query = matrix[query_idx]                  # (512,)
    sims  = matrix @ query                     # (N,) dot products
    sims[query_idx] = -np.inf                  # exclude self
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]


# ── Single LOOCV run ───────────────────────────────────────────────────────
def run_loocv(pages, matrix, oracle_df, unified_df,
              skip_threshold, confidence_threshold, k):
    """Run one complete LOOCV pass. Returns per-page results DataFrame."""

    labels, scores = build_skip_labels(pages, oracle_df, unified_df, skip_threshold)
    n = len(pages)
    records = []

    for i, test_page in enumerate(pages):
        # ── Find k nearest neighbors (excluding self) ──────────────────────
        neighbor_idxs, neighbor_sims = cosine_knn(matrix, i, k)

        # ── Per-extractor binary prediction ───────────────────────────────
        ext_results = {}
        for ext in EXTRACTORS:
            # Collect neighbor labels for this extractor
            neighbor_labels = np.array([labels[pages[j]][ext] for j in neighbor_idxs])
            # Similarity-weighted vote
            weights   = neighbor_sims.copy()
            weights   = np.clip(weights, 0, None)   # negative sim → 0 weight
            total_w   = weights.sum()
            if total_w > 0:
                conf_skip = float((neighbor_labels * weights).sum() / total_w)
            else:
                conf_skip = float(neighbor_labels.mean())

            pred_skip = 1 if conf_skip >= confidence_threshold else 0
            true_skip = labels[test_page][ext]

            ext_results[ext] = {
                "pred": pred_skip,
                "true": true_skip,
                "conf": round(conf_skip, 3),
                "correct": (pred_skip == true_skip),
            }

        # ── Determine which pipeline to run ───────────────────────────────
        skipped = frozenset(ext for ext in EXTRACTORS if ext_results[ext]["pred"] == 1)

        # Safety: never skip all 3
        if len(skipped) == 3:
            skipped = frozenset()

        pipeline_key = PIPELINE_MAP.get(skipped, "v2_full")

        # Look up actual score for chosen pipeline
        page_scores   = scores[test_page]
        final_score   = page_scores.get(pipeline_key, np.nan)
        full_score    = page_scores.get("v2_full",     np.nan)
        oracle_score  = oracle_df.loc[oracle_df["page"] == test_page, "oracle_best_combined"]
        oracle_score  = float(oracle_score.iloc[0]) if len(oracle_score) > 0 else np.nan

        # If the pipeline has no cached result, fall back to full ensemble
        if np.isnan(final_score):
            pipeline_key = "v2_full"
            final_score  = full_score
            skipped      = frozenset()

        records.append({
            "page":               test_page,
            "oracle_score":       oracle_score,
            "full_ensemble_score":full_score,
            "final_score":        final_score,
            "score_vs_oracle":    final_score / oracle_score if oracle_score else np.nan,
            "score_vs_ensemble":  final_score / full_score   if full_score   else np.nan,
            "pipeline_used":      pipeline_key,
            "api_calls":          3 - len(skipped),
            "n_skipped":          len(skipped),
            "extractors_skipped": ",".join(sorted(skipped)) if skipped else "none",
            # Per-extractor details
            **{f"{e}_correct": ext_results[e]["correct"] for e in EXTRACTORS},
            **{f"{e}_true":    ext_results[e]["true"]    for e in EXTRACTORS},
            **{f"{e}_pred":    ext_results[e]["pred"]    for e in EXTRACTORS},
            **{f"{e}_conf":    ext_results[e]["conf"]    for e in EXTRACTORS},
        })

    return pd.DataFrame(records)


# ── Evaluation + printing ──────────────────────────────────────────────────
def evaluate(results_df, skip_threshold, confidence_threshold, k, verbose=True):
    """Compute summary metrics. Returns dict for sweep comparison."""
    n = len(results_df)
    baseline_calls = n * 3

    # Binary accuracy
    total_correct = sum(results_df[f"{e}_correct"].sum() for e in EXTRACTORS)
    total_decisions = n * len(EXTRACTORS)
    bin_acc = total_correct / total_decisions

    # Per-extractor accuracy
    ext_acc = {}
    for ext in EXTRACTORS:
        true_skips = results_df[f"{ext}_true"].sum()
        pred_skips = results_df[f"{ext}_pred"].sum()
        correct    = results_df[f"{ext}_correct"].sum()
        ext_acc[ext] = {"acc": correct / n, "true": int(true_skips), "pred": int(pred_skips)}

    # Performance
    oracle_avg   = results_df["oracle_score"].mean()
    ensemble_avg = results_df["full_ensemble_score"].mean()
    final_avg    = results_df["final_score"].mean()
    pres_oracle  = final_avg / oracle_avg
    pres_ensemble= final_avg / ensemble_avg

    # Cost
    actual_calls  = results_df["api_calls"].sum()
    cost_reduction= (baseline_calls - actual_calls) / baseline_calls

    # Skip distribution
    skip_dist = results_df["n_skipped"].value_counts().sort_index()

    # Failures
    failures = results_df[results_df["score_vs_ensemble"] < 0.9].copy()
    failures["drop_pct"] = (1 - failures["score_vs_ensemble"]) * 100

    # Balanced score: 0.3×acc + 0.4×pres_ens + 0.3×cost
    balanced = 0.3 * bin_acc + 0.4 * pres_ensemble + 0.3 * cost_reduction

    if verbose:
        print(f"\n{'='*60}")
        print(f"CLIP BINARY LOOCV — k={k}, skip_thr={skip_threshold}, conf={confidence_threshold}")
        print(f"{'='*60}")

        print(f"\n[1] Per-Extractor Skip Accuracy:")
        for ext in EXTRACTORS:
            d = ext_acc[ext]
            print(f"  {ext:8s}: accuracy={d['acc']:.1%}  "
                  f"(true_skips={d['true']}, predicted_skips={d['pred']})")
        print(f"  Overall binary accuracy: {total_correct}/{total_decisions} ({bin_acc:.1%})")

        print(f"\n[2] Performance Preservation:")
        print(f"  Oracle avg:        {oracle_avg:.4f}")
        print(f"  Full ensemble avg: {ensemble_avg:.4f}  ({ensemble_avg/oracle_avg:.1%} of oracle)")
        print(f"  Final avg:         {final_avg:.4f}  ({pres_oracle:.1%} of oracle)")
        print(f"  vs full ensemble:  {pres_ensemble:.1%} of ensemble")

        print(f"\n[3] Cost Reduction:")
        print(f"  Baseline (3 × {n} pages): {baseline_calls} API calls")
        print(f"  Actual:                    {actual_calls} API calls")
        print(f"  Reduction: {cost_reduction:.1%}  ({baseline_calls - actual_calls} calls saved)")
        print(f"  Avg calls per page: {actual_calls/n:.2f}")

        print(f"\n[4] Skip Decision Distribution:")
        for n_sk, count in skip_dist.items():
            label = "full ensemble" if n_sk == 0 else f"skip {n_sk} model{'s' if n_sk>1 else ''}"
            print(f"  {label}: {count} pages ({count/n:.0%}) → {3-n_sk} API calls/page")

        if not failures.empty:
            print(f"\n[5] Pages Where Skipping Hurt (>10% below ensemble):")
            for _, row in failures.sort_values("drop_pct", ascending=False).iterrows():
                print(f"  {row['page']:15s}: dropped {row['drop_pct']:.1f}% "
                      f"(skipped: {row['extractors_skipped']})")
        else:
            print(f"\n[5] No pages dropped >10% below ensemble ✓")

        print(f"\n  Balanced score: {balanced:.4f}")

    return {
        "method":        "CLIP_binary",
        "k":             k,
        "skip_threshold":skip_threshold,
        "conf_threshold":confidence_threshold,
        "bin_acc":       round(bin_acc, 4),
        "pres_oracle":   round(pres_oracle, 4),
        "pres_ensemble": round(pres_ensemble, 4),
        "cost_reduction":round(cost_reduction, 4),
        "avg_calls":     round(actual_calls / n, 2),
        "balanced":      round(balanced, 4),
        "oracle_avg":    round(oracle_avg, 4),
        "ensemble_avg":  round(ensemble_avg, 4),
        "final_avg":     round(final_avg, 4),
    }


# ── Sweep ──────────────────────────────────────────────────────────────────
def run_sweep(pages, matrix, oracle_df, unified_df):
    """Sweep over hyperparameters and collect summary metrics."""
    k_values    = [3, 5, 7]
    thresholds  = [0.005, 0.010, 0.020, 0.030]
    conf_values = [0.60, 0.65, 0.70]

    print(f"\n{'='*60}")
    print(f"CLIP BINARY SWEEP ({len(k_values)*len(thresholds)*len(conf_values)} configs)")
    print(f"{'='*60}")

    rows = []
    for skip_thr, conf, k in product(thresholds, conf_values, k_values):
        results_df = run_loocv(pages, matrix, oracle_df, unified_df, skip_thr, conf, k)
        metrics    = evaluate(results_df, skip_thr, conf, k, verbose=False)
        rows.append(metrics)

    sweep_df = pd.DataFrame(rows).sort_values("balanced", ascending=False)

    print(f"\n{'='*80}")
    print("SWEEP RESULTS (sorted by balanced_score)")
    print(f"{'='*80}")
    print(f"  {'skip':>6} | {'conf':>5} | {'k':>3} | {'bin_acc':>8} | "
          f"{'pres_ens':>9} | {'cost↓':>6} | {'balanced':>9}")
    print("-"*80)
    for _, row in sweep_df.head(15).iterrows():
        print(f"  {row['skip_threshold']:>6.3f} | {row['conf_threshold']:>5.2f} | "
              f"{row['k']:>3} | {row['bin_acc']:>8.1%} | "
              f"{row['pres_ensemble']:>9.1%} | {row['cost_reduction']:>6.1%} | "
              f"{row['balanced']:>9.4f}")

    best = sweep_df.iloc[0]
    print(f"\n[Best Configuration — CLIP Binary]")
    print(f"  skip_threshold:     {best['skip_threshold']}")
    print(f"  confidence_threshold: {best['conf_threshold']}")
    print(f"  k:                  {best['k']}")
    print(f"  Binary accuracy:    {best['bin_acc']:.1%}")
    print(f"  Preservation vs ensemble: {best['pres_ensemble']:.1%}")
    print(f"  Cost reduction:     {best['cost_reduction']:.1%}")
    print(f"  Balanced score:     {best['balanced']:.4f}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "sweep_clip_binary_summary.csv")
    sweep_df.to_csv(out_path, index=False)
    print(f"\n[Saved] sweep_clip_binary_summary.csv")

    return sweep_df


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",              type=int,   default=5)
    parser.add_argument("--skip-threshold", type=float, default=0.01)
    parser.add_argument("--conf",           type=float, default=0.65)
    parser.add_argument("--sweep",          action="store_true",
                        help="Run full hyperparameter sweep")
    args = parser.parse_args()

    pages, matrix, oracle_df, unified_df = load_data()

    if args.sweep:
        run_sweep(pages, matrix, oracle_df, unified_df)
    else:
        results_df = run_loocv(pages, matrix, oracle_df, unified_df,
                               args.skip_threshold, args.conf, args.k)
        metrics = evaluate(results_df, args.skip_threshold, args.conf, args.k)

        tag = f"k{args.k}_thr{args.skip_threshold}_conf{args.conf}"
        out = os.path.join(OUTPUT_DIR, f"loocv_clip_binary_{tag}.csv")
        results_df.to_csv(out, index=False)
        print(f"[Saved] {os.path.basename(out)}")