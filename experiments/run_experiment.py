"""
experiments/run_experiment.py
Unified experiment runner for v1 and v2 pipelines.

Dispatches on --pipeline (default: v2).  Both pipelines share:
  • The same scorer (pipeline-agnostic)
  • The same cache discipline (check disk before calling any API)
  • The same report format (one CSV per pipeline version, plus a combined comparison)

Usage:
    python -m experiments.run_experiment --pipeline v2                    # run v2
    python -m experiments.run_experiment --pipeline v1                    # run v1
    python -m experiments.run_experiment --pipeline v2 --eval-only        # re-score cached v2
    python -m experiments.run_experiment --pipeline v2 --pages 1889_4     # single page
    python -m experiments.run_experiment --compare                        # compare v1 vs v2 reports
"""
import os
import sys
import json
import argparse
import itertools
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PIPELINES, MODELS, GT_DIR, RESULTS_DIR, REPORT_DIR, DATA_DIR
from src.evaluation.scorer import score_page, score_structure

# ---------------------------------------------------------------------------
# v1 imports (only used when pipeline == v1)
# ---------------------------------------------------------------------------
def _import_v1():
    from src.agents.structurer import Structurer
    from src.agents.extractor import Extractor
    from src.agents.corrector import Corrector
    from src.evaluation.scorer import _normalise_val, _amounts_match, _has_any_amount, _desc_similarity
    return Structurer, Extractor, Corrector, _normalise_val, _amounts_match, _has_any_amount, _desc_similarity

# ---------------------------------------------------------------------------
# v2 imports
# ---------------------------------------------------------------------------
def _import_v2():
    from src.agents.standalone_extractor import StandaloneExtractor
    from src.agents.supervisor import Supervisor
    return StandaloneExtractor, Supervisor


# ===========================================================================
# Shared helpers
# ===========================================================================
def _versioned_results_dir(version: str) -> str:
    path = os.path.join(RESULTS_DIR, version)
    os.makedirs(path, exist_ok=True)
    return path


def _load_cached(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_result(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _discover_pages(data_dir: str):
    """Find all image files and map them to GT sheet names."""
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


def _load_ground_truths(pages: dict):
    gts = {}
    for page_name in pages:
        gt_path = os.path.join(GT_DIR, f"{page_name}.json")
        if os.path.exists(gt_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                gts[page_name] = json.load(f)
        else:
            print(f"  [Warning] No GT for {page_name}")
    return gts


# ===========================================================================
# V1 Pipeline  (preserved exactly as before)
# ===========================================================================
CLASSIFIER_MODEL_V1 = "gemini-flash"
RETRY_THRESHOLD_V1  = 0.7
MAX_RETRIES_V1      = 2


def _v1_result_path(page_name, stage, s_key, e_key="", c_key="", retry=0):
    rdir = _versioned_results_dir("v1")
    if stage == "structurer":
        fname = f"{page_name}_structurer_{s_key}.json"
    elif stage == "extractor":
        fname = f"{page_name}_extractor_{s_key}_{e_key}.json"
    elif stage == "corrector":
        suffix = "" if retry == 0 else f"_retry{retry}"
        fname = f"{page_name}_corrector_{s_key}_{e_key}_{c_key}{suffix}.json"
    else:
        raise ValueError(f"Unknown v1 stage: {stage}")
    return os.path.join(rdir, fname)


def _v1_build_unmatched_report(pred, gt):
    """Build the retry feedback report (v1 logic preserved)."""
    _, _, _, _normalise_val, _amounts_match, _has_any_amount, _desc_similarity = _import_v1()

    pred_rows = pred.get("rows", [])
    gt_rows   = gt.get("rows", [])
    scoreable_types  = {"entry", "total"}
    CROSS_TYPE_PAIRS = {("entry", "total"), ("total", "entry")}

    gt_scoreable   = [r for r in gt_rows  if str(r.get("row_type","")).lower() in scoreable_types and _has_any_amount(r)]
    pred_scoreable = [r for r in pred_rows if _has_any_amount(r)]

    if not gt_scoreable:
        return "All rows matched."

    available = set(range(len(pred_scoreable)))
    unmatched = []

    for gt_row in gt_scoreable:
        gt_type = str(gt_row.get("row_type","")).lower()
        exact_candidates = []
        for p_idx in available:
            p_row = pred_scoreable[p_idx]
            if not _amounts_match(p_row, gt_row):
                continue
            p_type = str(p_row.get("row_type","")).lower()
            if p_type == gt_type or (p_type, gt_type) in CROSS_TYPE_PAIRS:
                exact_candidates.append((p_idx, p_type == gt_type, _desc_similarity(p_row, gt_row)))
        if exact_candidates:
            exact_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            available.remove(exact_candidates[0][0])
        else:
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

    def fmt(row, key):
        v = _normalise_val(row.get(key, ""))
        return v if v else "—"

    lines = [f"Total unmatched rows: {len(unmatched)} out of {len(gt_scoreable)} scoreable rows.\n"]
    for i, (gt_row, closest_pred) in enumerate(unmatched, 1):
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
                f"  Closest pred row:\n"
                f"    Description : \"{closest_pred.get('description', '')}\"\n"
                f"    Amounts     : £{fmt(closest_pred,'amount_pounds')}  "
                f"s{fmt(closest_pred,'amount_shillings')}  "
                f"d{fmt(closest_pred,'amount_pence_whole')}  "
                f"frac={fmt(closest_pred,'amount_pence_fraction')}"
            )
        lines.append("")
    return "\n".join(lines)


def run_v1_pipeline(image_path, page_name, s_key, e_key, c_key, gt, eval_only=False):
    Structurer, Extractor, Corrector, *_ = _import_v1()
    results = {}

    # Stage 1: Structurer
    path = _v1_result_path(page_name, "structurer", s_key)
    out  = _load_cached(path)
    if out is None and not eval_only:
        print(f"      [Run] Structurer ({s_key})...")
        out = Structurer(classifier_model_key=CLASSIFIER_MODEL_V1, structurer_model_key=s_key).run(image_path)
        _save_result(path, out)
    elif out:
        print(f"      [Cache] Structurer ({s_key})")
    if out is None:
        return results
    results["structurer"] = out

    # Stage 2: Extractor
    path = _v1_result_path(page_name, "extractor", s_key, e_key)
    out  = _load_cached(path)
    if out is None and not eval_only:
        print(f"      [Run] Extractor ({e_key})...")
        out = Extractor(model_key=e_key).run(image_path, skeleton=results["structurer"]["skeleton"])
        _save_result(path, out)
    elif out:
        print(f"      [Cache] Extractor ({e_key})")
    if out is None:
        return results
    results["extractor"] = out

    # Stage 3: Corrector + retry loop
    path = _v1_result_path(page_name, "corrector", s_key, e_key, c_key, retry=0)
    out  = _load_cached(path)
    if out is None and not eval_only:
        print(f"      [Run] Corrector ({c_key}) pass 1...")
        out = Corrector(model_key=c_key).run(image_path, extraction=results["extractor"])
        _save_result(path, out)
    elif out:
        print(f"      [Cache] Corrector ({c_key}) pass 1")
    if out is None:
        return results
    results["corrector"] = out

    latest = out
    for attempt in range(2, 2 + MAX_RETRIES_V1):
        scores = score_page(latest, gt)
        if scores["combined_score"] >= RETRY_THRESHOLD_V1:
            break
        print(f"      [Retry] Score {scores['combined_score']:.3f} < {RETRY_THRESHOLD_V1} — pass {attempt}...")
        retry_path = _v1_result_path(page_name, "corrector", s_key, e_key, c_key, retry=attempt)
        retry_out  = _load_cached(retry_path)
        if retry_out is None and not eval_only:
            report = _v1_build_unmatched_report(latest, gt)
            retry_out = Corrector(model_key=c_key).retry(image_path, latest, report, attempt)
            _save_result(retry_path, retry_out)
        elif retry_out:
            print(f"      [Cache] Corrector ({c_key}) pass {attempt}")
        else:
            break
        results[f"corrector_retry{attempt}"] = retry_out
        latest = retry_out

    results["corrector_best"] = latest
    return results


def score_v1(pipeline_out, gt):
    """Score all stages of a v1 run."""
    record = {}
    if "structurer" in pipeline_out:
        s = score_structure(pipeline_out["structurer"].get("skeleton", {}), gt)
        record["struct_axis1"] = s["axis1_score"]
    if "extractor" in pipeline_out:
        s = score_page(pipeline_out["extractor"], gt)
        record["extract_axis1"]    = s["axis1_score"]
        record["extract_axis2"]    = s["axis2_score"]
        record["extract_combined"] = s["combined_score"]
    if "corrector" in pipeline_out:
        s = score_page(pipeline_out["corrector"], gt)
        record["correct_axis1"]    = s["axis1_score"]
        record["correct_axis2"]    = s["axis2_score"]
        record["correct_combined"] = s["combined_score"]
    if "corrector_best" in pipeline_out:
        s = score_page(pipeline_out["corrector_best"], gt)
        record["best_axis1"]    = s["axis1_score"]
        record["best_axis2"]    = s["axis2_score"]
        record["best_combined"] = s["combined_score"]
    return record


# ===========================================================================
# V2 Pipeline  (competitive extraction + supervisor)
# ===========================================================================
def _v2_extractor_path(page_name, model_key):
    return os.path.join(_versioned_results_dir("v2"),
                        f"{page_name}_extractor_{model_key}.json")


def _v2_supervisor_path(page_name, supervisor_key, extractor_keys):
    e_part = "_".join(sorted(extractor_keys))
    return os.path.join(_versioned_results_dir("v2"),
                        f"{page_name}_supervisor_{supervisor_key}_[{e_part}].json")


def run_v2_pipeline(image_path, page_name, extractor_keys, supervisor_key, eval_only=False):
    StandaloneExtractor, Supervisor = _import_v2()
    results = {}

    # Stage 1: Run all extractors independently
    candidates = {}
    for e_key in extractor_keys:
        path = _v2_extractor_path(page_name, e_key)
        out  = _load_cached(path)
        if out is None and not eval_only:
            print(f"      [Run] Extractor ({e_key})...")
            out = StandaloneExtractor(model_key=e_key).run(image_path)
            _save_result(path, out)
        elif out:
            print(f"      [Cache] Extractor ({e_key})")
        else:
            print(f"      [Skip] Extractor ({e_key}) — no cache, eval-only mode")

        if out is not None:
            candidates[e_key] = out
            results[f"extractor_{e_key}"] = out

    if not candidates:
        print("      [Error] No extractor output available.")
        return results

    # Stage 2: Supervisor
    sup_path = _v2_supervisor_path(page_name, supervisor_key, list(candidates.keys()))
    sup_out  = _load_cached(sup_path)
    if sup_out is None and not eval_only:
        print(f"      [Run] Supervisor ({supervisor_key}) on {len(candidates)} candidates...")
        sup_out = Supervisor(model_key=supervisor_key).run(image_path, candidates)
        _save_result(sup_path, sup_out)
    elif sup_out:
        print(f"      [Cache] Supervisor ({supervisor_key})")
    else:
        print(f"      [Skip] Supervisor — no cache, eval-only mode")

    if sup_out is not None:
        results["supervisor"] = sup_out

    return results


def score_v2(pipeline_out, gt, extractor_keys):
    """Score all stages of a v2 run."""
    record = {}
    for e_key in extractor_keys:
        key = f"extractor_{e_key}"
        if key in pipeline_out:
            s = score_page(pipeline_out[key], gt)
            record[f"ext_{e_key}_axis1"]    = s["axis1_score"]
            record[f"ext_{e_key}_axis2"]    = s["axis2_score"]
            record[f"ext_{e_key}_combined"] = s["combined_score"]
    if "supervisor" in pipeline_out:
        s = score_page(pipeline_out["supervisor"], gt)
        record["supervisor_axis1"]    = s["axis1_score"]
        record["supervisor_axis2"]    = s["axis2_score"]
        record["supervisor_combined"] = s["combined_score"]
    return record


# ===========================================================================
# Main experiment loop
# ===========================================================================
def run_experiments(pipeline_version="v2", pages_filter=None, eval_only=False):
    pages = _discover_pages(DATA_DIR)
    if pages_filter:
        pages = {k: v for k, v in pages.items() if k in pages_filter}
    if not pages:
        print("[Error] No pages found.")
        return

    gts = _load_ground_truths(pages)
    cfg = PIPELINES[pipeline_version]

    mode = "EVAL-ONLY" if eval_only else "FULL RUN"
    print(f"\n[{mode}] Pipeline: {pipeline_version} — {cfg['description']}")
    print(f"  Pages: {len(pages)}  |  GT loaded: {len(gts)}\n")

    records = []

    for page_name, image_path in sorted(pages.items()):
        if page_name not in gts:
            continue
        gt = gts[page_name]
        print(f"\n[Page] {page_name}")

        if pipeline_version == "v1":
            combos = list(itertools.product(
                cfg["structurer"], cfg["extractor"], cfg["corrector"]
            ))
            for (s_key, e_key, c_key) in combos:
                print(f"  [{s_key} → {e_key} → {c_key}]")
                try:
                    pipeline_out = run_v1_pipeline(image_path, page_name, s_key, e_key, c_key, gt, eval_only)
                except Exception as ex:
                    print(f"      [Error] {ex}")
                    continue
                record = {"page": page_name, "structurer_model": s_key,
                          "extractor_model": e_key, "corrector_model": c_key}
                record.update(score_v1(pipeline_out, gt))
                records.append(record)

        elif pipeline_version == "v2":
            extractor_keys = cfg["extractors"]
            for sup_key in cfg["supervisor"]:
                print(f"  [extractors: {extractor_keys} → supervisor: {sup_key}]")
                try:
                    pipeline_out = run_v2_pipeline(image_path, page_name, extractor_keys, sup_key, eval_only)
                except Exception as ex:
                    print(f"      [Error] {ex}")
                    continue
                record = {"page": page_name, "supervisor_model": sup_key,
                          "extractor_models": "|".join(extractor_keys)}
                record.update(score_v2(pipeline_out, gt, extractor_keys))
                records.append(record)

    # --- Write report ---
    if records:
        os.makedirs(REPORT_DIR, exist_ok=True)
        df = pd.DataFrame(records)
        report_path = os.path.join(REPORT_DIR, f"experiment_results_{pipeline_version}.csv")
        df.to_csv(report_path, index=False)
        print(f"\n[Report] → {report_path}")

        print(f"\n--- {pipeline_version.upper()} SUMMARY ---")
        if pipeline_version == "v2" and "supervisor_combined" in df.columns:
            print(f"  Avg supervisor_combined : {df['supervisor_combined'].mean():.4f}")
            for e_key in cfg["extractors"]:
                col = f"ext_{e_key}_combined"
                if col in df.columns:
                    print(f"  Avg {col:42s}: {df[col].mean():.4f}")
        elif pipeline_version == "v1" and "best_combined" in df.columns:
            print(f"  Avg best_combined: {df['best_combined'].mean():.4f}")
    else:
        print("\n[Warning] No results collected.")


# ===========================================================================
# Comparison mode  (v1 vs v2 side by side)
# ===========================================================================
def run_comparison():
    v1_path = os.path.join(REPORT_DIR, "experiment_results_v1.csv")
    v2_path = os.path.join(REPORT_DIR, "experiment_results_v2.csv")

    missing = []
    if not os.path.exists(v1_path):
        missing.append("v1")
    if not os.path.exists(v2_path):
        missing.append("v2")
    if missing:
        print(f"[Error] Missing report(s): {missing}")
        print("  Run --pipeline v1 and --pipeline v2 first.")
        return

    v1 = pd.read_csv(v1_path)
    v2 = pd.read_csv(v2_path)

    v1_best = v1.groupby("page")["best_combined"].max().reset_index()
    v1_best.columns = ["page", "v1_best"]

    v2_best = v2.groupby("page")["supervisor_combined"].max().reset_index()
    v2_best.columns = ["page", "v2_supervisor"]

    merged = v1_best.merge(v2_best, on="page", how="outer").sort_values("page")
    merged["winner"] = merged.apply(
        lambda r: "v2" if (r.get("v2_supervisor", 0) or 0) > (r.get("v1_best", 0) or 0) else "v1",
        axis=1
    )
    merged["delta"] = (merged["v2_supervisor"].fillna(0) - merged["v1_best"].fillna(0)).round(4)

    print("\n--- V1 vs V2 COMPARISON (best combo per page) ---")
    print(merged.to_string(index=False))
    print(f"\n  V1 avg: {merged['v1_best'].mean():.4f}")
    print(f"  V2 avg: {merged['v2_supervisor'].mean():.4f}")
    print(f"  V2 wins on {(merged['winner']=='v2').sum()}/{len(merged)} pages")

    comp_path = os.path.join(REPORT_DIR, "v1_vs_v2_comparison.csv")
    merged.to_csv(comp_path, index=False)
    print(f"\n[Comparison] → {comp_path}")


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ledger extraction experiments.")
    parser.add_argument("--pipeline", choices=["v1", "v2"], default="v2",
                        help="Which pipeline to run (default: v2).")
    parser.add_argument("--pages", nargs="+", default=None,
                        help="Limit to specific page names.")
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Skip all API calls. Only load cached results and re-score.")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Compare v1 and v2 results side by side.")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    else:
        run_experiments(
            pipeline_version=args.pipeline,
            pages_filter=args.pages,
            eval_only=args.eval_only,
        )
