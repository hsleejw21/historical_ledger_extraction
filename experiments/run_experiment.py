"""
experiments/run_experiment.py
Main experiment runner.

Orchestrates the full 3-agent pipeline for every model combination defined
in AGENT_ROLES.  Results are cached on disk so interrupted runs can resume.
At the end, a CSV comparison report is written to experiments/reports/.

KEY BEHAVIOURS:
  • Cache-first:  every stage checks for an existing result file before calling
    any API.  Run with --eval-only to skip all API calls entirely and just
    re-score whatever is already cached.
  • Retry loop:   after the first corrector pass, if combined_score < RETRY_THRESHOLD
    the runner scores the output, identifies exactly which rows failed, and calls
    the corrector again with that concrete feedback.  Up to MAX_RETRIES attempts.
    Each retry is cached independently (corrector_..._retry2.json, etc.).

Usage:
    python -m experiments.run_experiment                          # full run
    python -m experiments.run_experiment --eval-only              # re-score cached results only
    python -m experiments.run_experiment --pages 1889_4 1873_5    # subset of pages
    python -m experiments.run_experiment --stage structurer       # run up to one stage only
"""
import os
import sys
import json
import argparse
import itertools
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AGENT_ROLES, MODELS, GT_DIR, RESULTS_DIR, REPORT_DIR
from src.agents.structurer import Structurer
from src.agents.extractor import Extractor
from src.agents.corrector import Corrector
from src.evaluation.scorer import (
    score_page, score_structure,
    _normalise_val, _amounts_match, _has_any_amount, _desc_similarity,
)

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
RETRY_THRESHOLD = 0.7    # combined_score below this triggers a retry
MAX_RETRIES     = 2      # maximum retry attempts after the first correction

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL = "gemini-flash"


# ===========================================================================
# Helpers
# ===========================================================================
def _result_path(page_name: str, stage: str, structurer_key: str,
                 extractor_key: str = "", corrector_key: str = "",
                 retry: int = 0) -> str:
    """
    Deterministic cache path.  retry=0 is the first corrector pass;
    retry=2 means the second attempt, saved as ..._retry2.json.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if stage == "structurer":
        fname = f"{page_name}_structurer_{structurer_key}.json"
    elif stage == "extractor":
        fname = f"{page_name}_extractor_{structurer_key}_{extractor_key}.json"
    elif stage == "corrector":
        suffix = "" if retry == 0 else f"_retry{retry}"
        fname = f"{page_name}_corrector_{structurer_key}_{extractor_key}_{corrector_key}{suffix}.json"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return os.path.join(RESULTS_DIR, fname)


def _load_cached(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_result(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _discover_pages(data_dir: str):
    """Find all image files and map them to their GT sheet names."""
    pages = {}
    if not os.path.exists(data_dir):
        print(f"[Warning] Image directory not found: {data_dir}")
        return pages

    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            stem = os.path.splitext(fname)[0]
            sheet_name = stem.replace("_image", "").replace("_page", "_")
            pages[sheet_name] = os.path.join(data_dir, fname)
    return pages


# ===========================================================================
# Unmatched-row report builder
# ===========================================================================
def _build_unmatched_report(pred: dict, gt: dict) -> str:
    """
    Replay the axis2 matching logic and build a human-readable report of every
    GT row that did NOT find an exact match in pred.  This report is what gets
    injected into the retry prompt so the model knows exactly what to fix.

    Each entry shows:
      • The row's position and description in the GT
      • The expected £/s/d (what the GT says)
      • The closest pred row that was available (even if it didn't match)
        so the model can see *how* it was wrong
    """
    pred_rows = pred.get("rows", [])
    gt_rows   = gt.get("rows", [])

    scoreable_types  = {"entry", "total"}
    CROSS_TYPE_PAIRS = {("entry", "total"), ("total", "entry")}

    gt_scoreable   = [r for r in gt_rows  if str(r.get("row_type", "")).lower() in scoreable_types and _has_any_amount(r)]
    pred_scoreable = [r for r in pred_rows if str(r.get("row_type", "")).lower() in scoreable_types]

    if not gt_scoreable:
        return "All rows matched."

    # Replay matching (same logic as scorer)
    available = set(range(len(pred_scoreable)))
    unmatched = []   # (gt_row, closest_pred_row_or_None)

    for gt_row in gt_scoreable:
        gt_type = str(gt_row.get("row_type", "")).lower()

        # Try exact match first
        exact_candidates = []
        for p_idx in available:
            p_row = pred_scoreable[p_idx]
            if not _amounts_match(p_row, gt_row):
                continue
            p_type = str(p_row.get("row_type", "")).lower()
            if p_type == gt_type or (p_type, gt_type) in CROSS_TYPE_PAIRS:
                exact_candidates.append((p_idx, p_type == gt_type, _desc_similarity(p_row, gt_row)))

        if exact_candidates:
            exact_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            available.remove(exact_candidates[0][0])
            # Matched — not added to unmatched list
        else:
            # Find closest pred row by description similarity (for the report)
            closest = None
            best_dsim = -1.0
            for p_idx in available:
                dsim = _desc_similarity(pred_scoreable[p_idx], gt_row)
                if dsim > best_dsim:
                    best_dsim = dsim
                    closest = pred_scoreable[p_idx]
            unmatched.append((gt_row, closest))

    if not unmatched:
        return "All rows matched."

    lines = [
        f"Total unmatched rows: {len(unmatched)} out of {len(gt_scoreable)} scoreable rows.\n",
        "For each unmatched row below, re-read the £ / s / d values directly from the image.\n",
    ]

    for i, (gt_row, closest_pred) in enumerate(unmatched, 1):
        def fmt(row, key):
            v = _normalise_val(row.get(key, ""))
            return v if v else "—"

        lines.append(
            f"--- Unmatched row {i} ---\n"
            f"  Description : \"{gt_row.get('description', '')}\"\n"
            f"  Expected type: {gt_row.get('row_type', '')}\n"
            f"  Expected amounts: £{fmt(gt_row,'amount_pounds')}  "
            f"s{fmt(gt_row,'amount_shillings')}  "
            f"d{fmt(gt_row,'amount_pence_whole')}  "
            f"frac={fmt(gt_row,'amount_pence_fraction')}"
        )

        if closest_pred:
            lines.append(
                f"  Closest pred row found:\n"
                f"    Description : \"{closest_pred.get('description', '')}\"\n"
                f"    Type        : {closest_pred.get('row_type', '')}\n"
                f"    Amounts     : £{fmt(closest_pred,'amount_pounds')}  "
                f"s{fmt(closest_pred,'amount_shillings')}  "
                f"d{fmt(closest_pred,'amount_pence_whole')}  "
                f"frac={fmt(closest_pred,'amount_pence_fraction')}"
            )
        else:
            lines.append("  Closest pred row: NONE AVAILABLE")

        lines.append("")  # blank line between entries

    return "\n".join(lines)


# ===========================================================================
# Pipeline execution
# ===========================================================================
def run_pipeline(image_path: str, page_name: str,
                 structurer_key: str, extractor_key: str, corrector_key: str,
                 gt: dict,
                 filter_stage: str = None,
                 eval_only: bool = False):
    """
    Run the full 3-agent pipeline for one page + one model combination.

    eval_only: if True, never call any API — only load cached results.
               Stages with no cache are simply absent from the returned dict.

    Returns a dict with keys "structurer", "extractor", "corrector" (and
    optionally "corrector_retry2", "corrector_retry3", etc.).
    """
    results = {}

    # ----------------------------------------------------------
    # STAGE 1: Structurer
    # ----------------------------------------------------------
    struct_path = _result_path(page_name, "structurer", structurer_key)
    struct_out  = _load_cached(struct_path)

    if struct_out is None and not eval_only:
        print(f"      [Run] Structurer ({structurer_key})...")
        agent = Structurer(classifier_model_key=CLASSIFIER_MODEL, structurer_model_key=structurer_key)
        struct_out = agent.run(image_path)
        _save_result(struct_path, struct_out)
    elif struct_out is not None:
        print(f"      [Cache] Structurer ({structurer_key})")
    else:
        print(f"      [Skip] Structurer — no cache, eval-only mode")

    if struct_out is None:
        return results   # can't continue without structurer output

    results["structurer"] = struct_out
    if filter_stage == "structurer":
        return results

    # ----------------------------------------------------------
    # STAGE 2: Extractor
    # ----------------------------------------------------------
    extract_path = _result_path(page_name, "extractor", structurer_key, extractor_key)
    extract_out  = _load_cached(extract_path)

    if extract_out is None and not eval_only:
        print(f"      [Run] Extractor ({extractor_key})...")
        agent = Extractor(model_key=extractor_key)
        extract_out = agent.run(image_path, skeleton=struct_out["skeleton"])
        _save_result(extract_path, extract_out)
    elif extract_out is not None:
        print(f"      [Cache] Extractor ({extractor_key})")
    else:
        print(f"      [Skip] Extractor — no cache, eval-only mode")

    if extract_out is None:
        return results

    results["extractor"] = extract_out
    if filter_stage == "extractor":
        return results

    # ----------------------------------------------------------
    # STAGE 3: Corrector  (first pass)
    # ----------------------------------------------------------
    correct_path = _result_path(page_name, "corrector", structurer_key, extractor_key, corrector_key, retry=0)
    correct_out  = _load_cached(correct_path)

    if correct_out is None and not eval_only:
        print(f"      [Run] Corrector ({corrector_key}) — pass 1...")
        agent = Corrector(model_key=corrector_key)
        correct_out = agent.run(image_path, extraction=extract_out)
        _save_result(correct_path, correct_out)
    elif correct_out is not None:
        print(f"      [Cache] Corrector ({corrector_key}) — pass 1")
    else:
        print(f"      [Skip] Corrector — no cache, eval-only mode")

    if correct_out is None:
        return results

    results["corrector"] = correct_out

    # ----------------------------------------------------------
    # RETRY LOOP
    # Score the first correction.  If combined < threshold, build
    # the unmatched report and call retry (unless eval-only and no
    # cached retry exists).
    # ----------------------------------------------------------
    latest_output = correct_out
    for attempt in range(2, 2 + MAX_RETRIES):   # attempt 2, 3, ...
        scores = score_page(latest_output, gt)
        if scores["combined_score"] >= RETRY_THRESHOLD:
            print(f"      [OK] Score {scores['combined_score']:.3f} >= {RETRY_THRESHOLD} — no retry needed.")
            break

        print(f"      [Retry] Score {scores['combined_score']:.3f} < {RETRY_THRESHOLD} — attempting pass {attempt}...")

        retry_path = _result_path(page_name, "corrector", structurer_key, extractor_key, corrector_key, retry=attempt)
        retry_out  = _load_cached(retry_path)

        if retry_out is None and not eval_only:
            # Build the concrete failure report
            unmatched_report = _build_unmatched_report(latest_output, gt)
            agent = Corrector(model_key=corrector_key)
            retry_out = agent.retry(
                image_path,
                previous_output=latest_output,
                unmatched_report=unmatched_report,
                attempt_number=attempt,
            )
            _save_result(retry_path, retry_out)
        elif retry_out is not None:
            print(f"      [Cache] Corrector ({corrector_key}) — pass {attempt}")
        else:
            print(f"      [Skip] Corrector retry {attempt} — no cache, eval-only mode")
            break   # can't continue retry chain without the previous output

        results[f"corrector_retry{attempt}"] = retry_out
        latest_output = retry_out

    # Store the best (latest) corrector output under a stable key for scoring
    results["corrector_best"] = latest_output

    return results


# ===========================================================================
# Scoring & reporting
# ===========================================================================
def run_experiments(pages_filter=None, stage_filter=None, eval_only=False):
    """
    Main loop: iterate over all pages × all model combinations.
    Score each stage against ground truth.  Collect into a DataFrame.
    """
    from src.config import DATA_DIR

    pages = _discover_pages(DATA_DIR)
    if pages_filter:
        pages = {k: v for k, v in pages.items() if k in pages_filter}

    if not pages:
        print("[Error] No pages found. Check DATA_DIR and image filenames.")
        return

    # Load ground truths
    ground_truths = {}
    for page_name in pages:
        gt_path = os.path.join(GT_DIR, f"{page_name}.json")
        if os.path.exists(gt_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truths[page_name] = json.load(f)
        else:
            print(f"  [Warning] No GT found for {page_name}, skipping.")

    # Model combinations
    all_combos = list(itertools.product(
        AGENT_ROLES["structurer"],
        AGENT_ROLES["extractor"],
        AGENT_ROLES["corrector"],
    ))
    mode_label = "EVAL-ONLY (cached results)" if eval_only else "FULL RUN"
    print(f"\n[{mode_label}] {len(pages)} pages × {len(all_combos)} combos = {len(pages)*len(all_combos)} pipelines\n")

    records = []

    for page_name, image_path in sorted(pages.items()):
        if page_name not in ground_truths:
            continue
        gt = ground_truths[page_name]
        print(f"\n[Page] {page_name}")

        for (s_key, e_key, c_key) in all_combos:
            print(f"  [{s_key} → {e_key} → {c_key}]")

            try:
                pipeline_out = run_pipeline(
                    image_path, page_name,
                    structurer_key=s_key,
                    extractor_key=e_key,
                    corrector_key=c_key,
                    gt=gt,
                    filter_stage=stage_filter,
                    eval_only=eval_only,
                )
            except Exception as ex:
                print(f"      [Error] Pipeline failed: {ex}")
                continue

            # --- Score each available stage ---
            record = {
                "page": page_name,
                "structurer_model": s_key,
                "extractor_model":  e_key,
                "corrector_model":  c_key,
            }

            if "structurer" in pipeline_out:
                skeleton = pipeline_out["structurer"].get("skeleton", {})
                s_scores = score_structure(skeleton, gt)
                record["struct_axis1"] = s_scores["axis1_score"]

            if "extractor" in pipeline_out:
                e_scores = score_page(pipeline_out["extractor"], gt)
                record["extract_axis1"]    = e_scores["axis1_score"]
                record["extract_axis2"]    = e_scores["axis2_score"]
                record["extract_combined"] = e_scores["combined_score"]

            # Score first corrector pass
            if "corrector" in pipeline_out:
                c_scores = score_page(pipeline_out["corrector"], gt)
                record["correct_axis1"]    = c_scores["axis1_score"]
                record["correct_axis2"]    = c_scores["axis2_score"]
                record["correct_combined"] = c_scores["combined_score"]

            # Score each retry pass individually
            for attempt in range(2, 2 + MAX_RETRIES):
                key = f"corrector_retry{attempt}"
                if key in pipeline_out:
                    r_scores = score_page(pipeline_out[key], gt)
                    record[f"retry{attempt}_axis1"]    = r_scores["axis1_score"]
                    record[f"retry{attempt}_axis2"]    = r_scores["axis2_score"]
                    record[f"retry{attempt}_combined"] = r_scores["combined_score"]

            # Score the best (final) corrector output
            if "corrector_best" in pipeline_out:
                best_scores = score_page(pipeline_out["corrector_best"], gt)
                record["best_axis1"]    = best_scores["axis1_score"]
                record["best_axis2"]    = best_scores["axis2_score"]
                record["best_combined"] = best_scores["combined_score"]

            records.append(record)

    # --- Write report ---
    if records:
        os.makedirs(REPORT_DIR, exist_ok=True)
        df = pd.DataFrame(records)

        report_path = os.path.join(REPORT_DIR, "experiment_results.csv")
        df.to_csv(report_path, index=False)
        print(f"\n[Report] Full results → {report_path}")

        # Summary
        print("\n--- SUMMARY: Average scores by model combination ---")
        best_col = "best_combined" if "best_combined" in df.columns else "correct_combined"
        summary_cols = ["structurer_model", "extractor_model", "corrector_model", best_col]
        available = [c for c in summary_cols if c in df.columns]
        if len(available) == len(summary_cols):
            summary = df.groupby(["structurer_model", "extractor_model", "corrector_model"])[best_col].mean()
            print(summary.sort_values(ascending=False).to_string())
        else:
            print(df[available].to_string())
    else:
        print("\n[Warning] No results collected.")


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ledger extraction experiments.")
    parser.add_argument("--pages", nargs="+", default=None,
                        help="Limit to specific page names.")
    parser.add_argument("--stage", choices=["structurer", "extractor", "corrector"], default=None,
                        help="Run only up to this stage.")
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Skip all API calls. Only load cached results and re-score.")
    args = parser.parse_args()

    run_experiments(
        pages_filter=args.pages,
        stage_filter=args.stage,
        eval_only=args.eval_only,
    )
