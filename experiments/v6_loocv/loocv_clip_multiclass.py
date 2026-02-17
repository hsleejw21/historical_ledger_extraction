"""
experiments/v6_loocv/loocv_clip_multiclass.py

Phase 4B: LOOCV with CLIP embeddings — multi-class pipeline selection.

Instead of asking "which models to skip?" (binary, Phase 3 + 4A), this asks:
  "Which complete v2 pipeline should I run for this page?"

Candidates (from your oracle distribution):
  v2_no_gemini  → 12/33 pages (36%) ← wins most often!
  v2_no_claude  →  8/33 pages (24%) ← current SOTA
  v2_no_gpt     →  8/33 pages (24%)
  v2_full       →  5/33 pages (15%)

For each test page, k-NN finds similar pages by CLIP cosine similarity,
checks which pipeline won on those neighbors, and runs that pipeline.
Fallback: if neighbor agreement < confidence_threshold → run v2_no_claude (SOTA default).

This directly fixes the core flaw identified in Phase 3:
  The binary skip system NEVER predicted v2_no_gemini (because gemini was
  treated as the backbone), but oracle shows it's best on 36% of pages.

Output:
  experiments/v6_loocv/outputs/loocv_clip_multiclass_k{k}_conf{c}.csv

Usage:
    python -m experiments.v6_loocv.loocv_clip_multiclass [--k 5] [--conf 0.5]
    python -m experiments.v6_loocv.loocv_clip_multiclass --sweep
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import BASE_DIR, REPORT_DIR

# ── Paths ──────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "visual_features", "clip_embeddings.json")
V6_OUTPUT_DIR   = os.path.join(BASE_DIR, "experiments", "v6_loocv", "outputs")
ORACLE_PATH     = os.path.join(V6_OUTPUT_DIR, "oracle_best_per_page.csv")
UNIFIED_PATH    = os.path.join(V6_OUTPUT_DIR, "unified_results.csv")
OUTPUT_DIR      = V6_OUTPUT_DIR

# The 4 candidate pipelines — exactly what the oracle considers
CANDIDATES      = ["v2_no_gemini", "v2_no_claude", "v2_no_gpt", "v2_full"]
FALLBACK        = "v2_no_claude"   # default if confidence is too low

# API call cost per pipeline (supervisor counts as 1 call, each extractor is 1)
API_CALLS       = {
    "v2_no_gemini": 2,   # gpt + claude + supervisor
    "v2_no_claude": 2,   # gemini + gpt + supervisor
    "v2_no_gpt":    2,   # gemini + claude + supervisor
    "v2_full":      3,   # all 3 + supervisor
}
BASELINE_CALLS  = 3      # always running v2_full baseline


# ── Data loading ───────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"CLIP embeddings not found: {EMBEDDINGS_PATH}\n"
            f"Run this first: python -m experiments.v6_loocv.extract_clip_embeddings"
        )
    
    with open(EMBEDDINGS_PATH) as f:
        raw = json.load(f)

    pages  = sorted(raw.keys())
    matrix = np.array([raw[p] for p in pages], dtype=np.float32)

    if not os.path.exists(ORACLE_PATH):
        raise FileNotFoundError(f"Oracle file not found: {ORACLE_PATH}")
    if not os.path.exists(UNIFIED_PATH):
        raise FileNotFoundError(f"Unified results not found: {UNIFIED_PATH}")
    
    oracle_df  = pd.read_csv(ORACLE_PATH)
    unified_df = pd.read_csv(UNIFIED_PATH)

    oracle_df = oracle_df.set_index("page").reindex(pages).reset_index()

    print(f"[Loaded] {len(pages)} pages | dim={matrix.shape[1]}")

    # Show oracle distribution
    print(f"\n[Oracle pipeline distribution]")
    dist = oracle_df["oracle_best_pipeline"].value_counts()
    for pipeline, count in dist.items():
        bar = "█" * count
        print(f"  {pipeline:15s}: {count:2d} pages  {bar}")

    return pages, matrix, oracle_df, unified_df


# ── Score lookup ───────────────────────────────────────────────────────────
def build_score_lookup(pages, unified_df):
    """Build {page → {pipeline → score}} for fast lookup."""
    lookup = {p: {} for p in pages}
    for _, row in unified_df.iterrows():
        page = row["page"]
        if page in lookup:
            lookup[page][row["pipeline"]] = row["final_combined"]
    return lookup


# ── k-NN cosine similarity ─────────────────────────────────────────────────
def cosine_knn(matrix, query_idx, k):
    """k nearest neighbors by cosine similarity (embeddings are L2-normalized)."""
    sims = matrix @ matrix[query_idx]
    sims[query_idx] = -np.inf
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]


# ── Single LOOCV run ───────────────────────────────────────────────────────
def run_loocv(pages, matrix, oracle_df, unified_df,
              k, confidence_threshold):
    """Run one complete LOOCV pass with multi-class prediction."""

    score_lookup = build_score_lookup(pages, unified_df)

    # Oracle labels: best pipeline per page
    oracle_labels = dict(zip(oracle_df["page"], oracle_df["oracle_best_pipeline"]))

    records = []

    for i, test_page in enumerate(pages):
        # ── Find k nearest neighbors ───────────────────────────────────────
        neighbor_idxs, neighbor_sims = cosine_knn(matrix, i, k)

        # ── Weighted vote over candidates ──────────────────────────────────
        weights = np.clip(neighbor_sims, 0, None)   # negative cosine → 0 weight
        total_w = weights.sum()

        vote_scores = {c: 0.0 for c in CANDIDATES}

        for j, (n_idx, w) in enumerate(zip(neighbor_idxs, weights)):
            neighbor_page = pages[n_idx]
            winner = oracle_labels.get(neighbor_page)
            if winner in vote_scores and total_w > 0:
                vote_scores[winner] += w / total_w

        # ── Predict pipeline ───────────────────────────────────────────────
        predicted = max(vote_scores, key=vote_scores.get)
        confidence = vote_scores[predicted]

        # Fallback if not confident enough
        used_fallback = confidence < confidence_threshold
        chosen_pipeline = FALLBACK if used_fallback else predicted

        # ── Collect actual scores ──────────────────────────────────────────
        page_scores  = score_lookup[test_page]
        final_score  = page_scores.get(chosen_pipeline, np.nan)
        full_score   = page_scores.get("v2_full", np.nan)
        oracle_score = float(oracle_df.loc[oracle_df["page"] == test_page,
                                           "oracle_best_combined"].iloc[0])

        # If chosen pipeline has no cached result, fall back
        if np.isnan(final_score):
            chosen_pipeline = "v2_full"
            final_score = full_score

        oracle_pipeline = oracle_labels.get(test_page, "")
        prediction_correct = (chosen_pipeline == oracle_pipeline)

        # Scores for all candidates (for reference)
        candidate_scores = {c: page_scores.get(c, np.nan) for c in CANDIDATES}

        records.append({
            "page":               test_page,
            "oracle_pipeline":    oracle_pipeline,
            "oracle_score":       oracle_score,
            "full_ensemble_score":full_score,
            "predicted_pipeline": predicted,
            "prediction_conf":    round(confidence, 3),
            "used_fallback":      used_fallback,
            "chosen_pipeline":    chosen_pipeline,
            "final_score":        final_score,
            "prediction_correct": prediction_correct,
            "score_vs_oracle":    final_score / oracle_score if oracle_score else np.nan,
            "score_vs_ensemble":  final_score / full_score   if full_score   else np.nan,
            "api_calls":          API_CALLS.get(chosen_pipeline, 3),
            # Vote scores for each candidate
            **{f"vote_{c}": round(vote_scores[c], 3) for c in CANDIDATES},
            # Actual score for each candidate
            **{f"score_{c}": candidate_scores[c] for c in CANDIDATES},
        })

    return pd.DataFrame(records)


# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(results_df, k, confidence_threshold, verbose=True):
    """Compute and optionally print evaluation metrics."""
    n = len(results_df)
    baseline_calls = n * BASELINE_CALLS

    # Pipeline selection accuracy
    accuracy    = results_df["prediction_correct"].mean()
    n_correct   = results_df["prediction_correct"].sum()
    fallback_n  = results_df["used_fallback"].sum()
    fallback_r  = fallback_n / n

    # Performance
    oracle_avg   = results_df["oracle_score"].mean()
    ensemble_avg = results_df["full_ensemble_score"].mean()
    final_avg    = results_df["final_score"].mean()
    pres_oracle  = final_avg / oracle_avg
    pres_ensemble= final_avg / ensemble_avg

    # Cost
    actual_calls  = results_df["api_calls"].sum()
    cost_reduction= (baseline_calls - actual_calls) / baseline_calls

    # Per-candidate distribution of chosen pipeline
    chosen_dist = results_df["chosen_pipeline"].value_counts()

    # Per-candidate accuracy (when that pipeline was predicted, was it correct?)
    per_candidate_acc = {}
    for c in CANDIDATES:
        rows = results_df[results_df["chosen_pipeline"] == c]
        if len(rows) > 0:
            per_candidate_acc[c] = rows["prediction_correct"].mean()

    # Failures
    failures = results_df[results_df["score_vs_ensemble"] < 0.9].copy()
    failures["drop_pct"] = (1 - failures["score_vs_ensemble"]) * 100

    # Balanced score
    balanced = 0.3 * accuracy + 0.4 * pres_ensemble + 0.3 * cost_reduction

    if verbose:
        print(f"\n{'='*60}")
        print(f"CLIP MULTI-CLASS — k={k}, conf_threshold={confidence_threshold}")
        print(f"{'='*60}")

        print(f"\n[1] Pipeline Selection Accuracy:")
        print(f"  Correct: {n_correct}/{n} ({accuracy:.1%})")
        print(f"  Fallback to {FALLBACK}: {fallback_n}/{n} ({fallback_r:.1%})")

        print(f"\n  Per-candidate breakdown:")
        for c in CANDIDATES:
            oracle_count = (results_df["oracle_pipeline"] == c).sum()
            chosen_count = (results_df["chosen_pipeline"] == c).sum()
            acc = per_candidate_acc.get(c, float("nan"))
            print(f"    {c:15s}: oracle={oracle_count:2d} pages, "
                  f"chosen={chosen_count:2d}, accuracy={acc:.1%}")

        print(f"\n[2] Performance Preservation:")
        print(f"  Oracle avg:        {oracle_avg:.4f}")
        print(f"  Full ensemble avg: {ensemble_avg:.4f}  ({ensemble_avg/oracle_avg:.1%} of oracle)")
        print(f"  Final avg:         {final_avg:.4f}  ({pres_oracle:.1%} of oracle)")
        print(f"  vs full ensemble:  {pres_ensemble:.1%} of ensemble")

        print(f"\n[3] Cost Reduction:")
        print(f"  Baseline (v2_full × {n} pages): {baseline_calls} API calls")
        print(f"  Actual:                          {actual_calls} API calls")
        print(f"  Reduction: {cost_reduction:.1%}  ({baseline_calls - actual_calls} calls saved)")
        print(f"  Note: all v2 ablations cost 2 calls (vs 3 for v2_full)")

        print(f"\n[4] Chosen Pipeline Distribution:")
        for pipeline, count in chosen_dist.items():
            print(f"  {pipeline:15s}: {count:2d} pages ({count/n:.0%})")

        if not failures.empty:
            print(f"\n[5] Pages Where Choice Hurt (>10% below ensemble):")
            for _, row in failures.sort_values("drop_pct", ascending=False).iterrows():
                print(f"  {row['page']:15s}: dropped {row['drop_pct']:.1f}%  "
                      f"chosen={row['chosen_pipeline']}  oracle={row['oracle_pipeline']}")
        else:
            print(f"\n[5] No pages dropped >10% below ensemble ✓")

        print(f"\n  Balanced score: {balanced:.4f}")

    return {
        "method":          "CLIP_multiclass",
        "k":               k,
        "conf_threshold":  confidence_threshold,
        "accuracy":        round(accuracy, 4),
        "fallback_rate":   round(fallback_r, 4),
        "pres_oracle":     round(pres_oracle, 4),
        "pres_ensemble":   round(pres_ensemble, 4),
        "cost_reduction":  round(cost_reduction, 4),
        "avg_calls":       round(actual_calls / n, 2),
        "balanced":        round(balanced, 4),
        "oracle_avg":      round(oracle_avg, 4),
        "ensemble_avg":    round(ensemble_avg, 4),
        "final_avg":       round(final_avg, 4),
    }


# ── Sweep ──────────────────────────────────────────────────────────────────
def run_sweep(pages, matrix, oracle_df, unified_df):
    k_values    = [3, 5, 7, 9]
    conf_values = [0.30, 0.40, 0.50, 0.60, 0.70]

    print(f"\n{'='*60}")
    print(f"CLIP MULTI-CLASS SWEEP ({len(k_values)*len(conf_values)} configs)")
    print(f"{'='*60}")

    rows = []
    for k, conf in product(k_values, conf_values):
        results_df = run_loocv(pages, matrix, oracle_df, unified_df, k, conf)
        metrics    = evaluate(results_df, k, conf, verbose=False)
        rows.append(metrics)

    sweep_df = pd.DataFrame(rows).sort_values("balanced", ascending=False)

    print(f"\n{'='*80}")
    print("SWEEP RESULTS (sorted by balanced_score)")
    print(f"{'='*80}")
    print(f"  {'k':>3} | {'conf':>5} | {'accuracy':>9} | {'pres_ens':>9} | "
          f"{'cost↓':>6} | {'fallback':>9} | {'balanced':>9}")
    print("-"*80)
    for _, row in sweep_df.head(15).iterrows():
        print(f"  {row['k']:>3} | {row['conf_threshold']:>5.2f} | "
              f"{row['accuracy']:>9.1%} | {row['pres_ensemble']:>9.1%} | "
              f"{row['cost_reduction']:>6.1%} | {row['fallback_rate']:>9.1%} | "
              f"{row['balanced']:>9.4f}")

    best = sweep_df.iloc[0]
    print(f"\n[Best Configuration — CLIP Multi-Class]")
    print(f"  k:                  {best['k']}")
    print(f"  conf_threshold:     {best['conf_threshold']}")
    print(f"  Accuracy:           {best['accuracy']:.1%}")
    print(f"  Preservation vs ensemble: {best['pres_ensemble']:.1%}")
    print(f"  Cost reduction:     {best['cost_reduction']:.1%}")
    print(f"  Balanced score:     {best['balanced']:.4f}")

    out_path = os.path.join(OUTPUT_DIR, "sweep_clip_multiclass_summary.csv")
    sweep_df.to_csv(out_path, index=False)
    print(f"\n[Saved] sweep_clip_multiclass_summary.csv")

    return sweep_df


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",    type=int,   default=5)
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Min confidence to use prediction (else fallback). "
                             "Lower = more predictions, less fallback.")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    pages, matrix, oracle_df, unified_df = load_data()

    if args.sweep:
        run_sweep(pages, matrix, oracle_df, unified_df)
    else:
        results_df = run_loocv(pages, matrix, oracle_df, unified_df,
                               args.k, args.conf)
        metrics = evaluate(results_df, args.k, args.conf)

        tag = f"k{args.k}_conf{args.conf}"
        out = os.path.join(OUTPUT_DIR, f"loocv_clip_multiclass_{tag}.csv")
        results_df.to_csv(out, index=False)
        print(f"[Saved] {os.path.basename(out)}")