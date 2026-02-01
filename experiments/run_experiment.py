"""
experiments/run_experiment.py
Main experiment runner.

Orchestrates the full 3-agent pipeline for every model combination defined
in AGENT_ROLES.  Results are cached on disk so interrupted runs can resume.
At the end, a CSV comparison report is written to experiments/reports/.

Usage:
    python -m experiments.run_experiment

    # Run only a subset of pages (by sheet name):
    python -m experiments.run_experiment --pages 1889_4 1873_5

    # Run a single agent role to isolate it:
    python -m experiments.run_experiment --stage structurer
"""
import os
import sys
import json
import argparse
import itertools
import pandas as pd

# Make sure src is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AGENT_ROLES, MODELS, GT_DIR, RESULTS_DIR, REPORT_DIR
from src.agents.structurer import Structurer
from src.agents.extractor import Extractor
from src.agents.corrector import Corrector
from src.evaluation.scorer import score_page

# ---------------------------------------------------------------------------
# Defaults — which model is used for the layout classifier (always cheap)
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL = "gemini-flash"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result_path(page_name: str, stage: str, structurer_key: str,
                 extractor_key: str = "", corrector_key: str = "") -> str:
    """
    Build a deterministic cache path for a given pipeline configuration.
    Stage is one of: structurer, extractor, corrector
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if stage == "structurer":
        fname = f"{page_name}_structurer_{structurer_key}.json"
    elif stage == "extractor":
        fname = f"{page_name}_extractor_{structurer_key}_{extractor_key}.json"
    elif stage == "corrector":
        fname = f"{page_name}_corrector_{structurer_key}_{extractor_key}_{corrector_key}.json"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return os.path.join(RESULTS_DIR, fname)


def _load_cached(path: str):
    """Load a cached result if it exists, else return None."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_result(path: str, data: dict):
    """Save a result dict to disk."""
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
            # Strip extension and common suffixes like _image to get the sheet name
            stem = os.path.splitext(fname)[0]
            # Try common naming patterns: "1889_4_image" → "1889_4", "1889_page4_image" → "1889_4"
            sheet_name = stem.replace("_image", "").replace("_page", "_")
            pages[sheet_name] = os.path.join(data_dir, fname)
    return pages


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
def run_pipeline(image_path: str, page_name: str,
                 structurer_key: str, extractor_key: str, corrector_key: str,
                 filter_stage: str = None):
    """
    Run the full 3-agent pipeline for one page + one model combination.
    Caches each stage independently.  Returns a dict of all three outputs.

    filter_stage: if set, only run up to that stage (for isolated benchmarking).
    """
    results = {}

    # ----------------------------------------------------------
    # STAGE 1: Structurer
    # ----------------------------------------------------------
    struct_path = _result_path(page_name, "structurer", structurer_key)
    struct_out  = _load_cached(struct_path)

    if struct_out is None:
        print(f"      [Run] Structurer ({structurer_key})...")
        agent = Structurer(classifier_model_key=CLASSIFIER_MODEL, structurer_model_key=structurer_key)
        struct_out = agent.run(image_path)
        _save_result(struct_path, struct_out)
    else:
        print(f"      [Cache] Structurer ({structurer_key})")

    results["structurer"] = struct_out

    if filter_stage == "structurer":
        return results

    # ----------------------------------------------------------
    # STAGE 2: Extractor  (needs skeleton from Structurer)
    # ----------------------------------------------------------
    extract_path = _result_path(page_name, "extractor", structurer_key, extractor_key)
    extract_out  = _load_cached(extract_path)

    if extract_out is None:
        print(f"      [Run] Extractor ({extractor_key})...")
        agent = Extractor(model_key=extractor_key)
        extract_out = agent.run(image_path, skeleton=struct_out["skeleton"])
        _save_result(extract_path, extract_out)
    else:
        print(f"      [Cache] Extractor ({extractor_key})")

    results["extractor"] = extract_out

    if filter_stage == "extractor":
        return results

    # ----------------------------------------------------------
    # STAGE 3: Corrector  (needs extraction from Extractor)
    # ----------------------------------------------------------
    correct_path = _result_path(page_name, "corrector", structurer_key, extractor_key, corrector_key)
    correct_out  = _load_cached(correct_path)

    if correct_out is None:
        print(f"      [Run] Corrector ({corrector_key})...")
        agent = Corrector(model_key=corrector_key)
        correct_out = agent.run(image_path, extraction=extract_out)
        _save_result(correct_path, correct_out)
    else:
        print(f"      [Cache] Corrector ({corrector_key})")

    results["corrector"] = correct_out

    return results


# ---------------------------------------------------------------------------
# Scoring & reporting
# ---------------------------------------------------------------------------
def run_experiments(pages_filter=None, stage_filter=None):
    """
    Main loop: iterate over all pages × all model combinations.
    Score each stage against ground truth.  Collect into a DataFrame.
    """
    from src.config import DATA_DIR

    # Discover pages
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

    # Build all model combinations
    structurer_models = AGENT_ROLES["structurer"]
    extractor_models  = AGENT_ROLES["extractor"]
    corrector_models  = AGENT_ROLES["corrector"]

    all_combos = list(itertools.product(structurer_models, extractor_models, corrector_models))
    print(f"\nRunning {len(pages)} pages × {len(all_combos)} model combos = {len(pages)*len(all_combos)} pipeline runs\n")

    # Results collector
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
                    filter_stage=stage_filter,
                )
            except Exception as ex:
                print(f"      [Error] Pipeline failed: {ex}")
                continue

            # Score each available stage
            record = {
                "page": page_name,
                "structurer_model": s_key,
                "extractor_model":  e_key,
                "corrector_model":  c_key,
            }

            if "structurer" in pipeline_out:
                skeleton = pipeline_out["structurer"].get("skeleton", {})
                s_scores = score_page(skeleton, gt)
                record["struct_axis1"]    = s_scores["axis1_score"]
                record["struct_combined"] = s_scores["combined_score"]

            if "extractor" in pipeline_out:
                e_scores = score_page(pipeline_out["extractor"], gt)
                record["extract_axis1"]    = e_scores["axis1_score"]
                record["extract_axis2"]    = e_scores["axis2_score"]
                record["extract_combined"] = e_scores["combined_score"]

            if "corrector" in pipeline_out:
                c_scores = score_page(pipeline_out["corrector"], gt)
                record["correct_axis1"]    = c_scores["axis1_score"]
                record["correct_axis2"]    = c_scores["axis2_score"]
                record["correct_combined"] = c_scores["combined_score"]

            records.append(record)

    # ---------------------
    # Write report
    # ---------------------
    if records:
        os.makedirs(REPORT_DIR, exist_ok=True)
        df = pd.DataFrame(records)

        # Full detailed report
        report_path = os.path.join(REPORT_DIR, "experiment_results.csv")
        df.to_csv(report_path, index=False)
        print(f"\n[Report] Full results → {report_path}")

        # Summary: best model per role (by average combined score across pages)
        print("\n--- SUMMARY: Average scores by model combination ---")
        summary_cols = ["structurer_model", "extractor_model", "corrector_model", "correct_combined"]
        available = [c for c in summary_cols if c in df.columns]
        if len(available) == len(summary_cols):
            summary = df.groupby(["structurer_model", "extractor_model", "corrector_model"])["correct_combined"].mean()
            print(summary.sort_values(ascending=False).to_string())
        else:
            print(df[available].to_string())
    else:
        print("\n[Warning] No results collected.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ledger extraction experiments.")
    parser.add_argument("--pages", nargs="+", default=None,
                        help="Limit to specific page names (sheet names from GT).")
    parser.add_argument("--stage", choices=["structurer", "extractor", "corrector"], default=None,
                        help="Run only up to this stage (useful for isolated benchmarking).")
    args = parser.parse_args()

    run_experiments(pages_filter=args.pages, stage_filter=args.stage)
