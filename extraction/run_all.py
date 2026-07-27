"""
run_all.py

One-shot script to extract all annual accounts (1700–1900).

Steps:
  1. Convert any not-yet-converted PDFs in data/annual_accounts_1700-1900/ to
     PNG images in data/images/  (skips pages that already exist).
  2. Seed the pipeline cache from experiments/results/sample_pdf/ so pages
     already extracted in previous experiments are not re-run.
  3. Run the extraction pipeline on all images, reusing any cached
     extractor/supervisor results so already-processed pages are skipped.
  4. Write a single Excel file to pipeline/output/ledger_<timestamp>.xlsx

Run from the project root:
    python run_all.py

To force re-extraction of everything (ignore cache):
    python run_all.py --no-cache
"""

import os
import sys
import shutil
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Directories (all relative to this file's location = project root)
# ---------------------------------------------------------------------------
ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
PDF_DIR          = os.path.join(ROOT_DIR, "data", "annual_accounts_1700-1900")
IMG_DIR          = os.path.join(ROOT_DIR, "data", "images")
CACHE_DIR        = os.path.join(ROOT_DIR, "pipeline", "cache")
OUTPUT_DIR       = os.path.join(ROOT_DIR, "pipeline", "output")
SAMPLE_RESULTS   = os.path.join(ROOT_DIR, "experiments", "results", "sample_pdf")

# ---------------------------------------------------------------------------
# Step 1 — PDF → images
# ---------------------------------------------------------------------------
def convert_pdfs(pdf_dir: str, img_dir: str):
    try:
        import fitz  # pymupdf
    except ImportError:
        print("[Warning] PyMuPDF (fitz) not installed — skipping PDF conversion.")
        print("          Install with: pip install pymupdf")
        return

    os.makedirs(img_dir, exist_ok=True)
    DPI = 150
    MATRIX = fitz.Matrix(DPI / 72, DPI / 72)

    pdf_files = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    print(f"[Step 1] Converting PDFs: {len(pdf_files)} files in {pdf_dir}")

    total_new = 0
    total_skipped = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        stem = os.path.splitext(pdf_file)[0]

        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        new_count = 0
        for page_idx in range(num_pages):
            page_num = page_idx + 1
            out_name = f"{stem}_{page_num}_image.png"
            out_path = os.path.join(img_dir, out_name)

            if os.path.exists(out_path):
                total_skipped += 1
                continue

            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=MATRIX)
            pix.save(out_path)
            new_count += 1
            total_new += 1

        doc.close()

        status = f"{new_count} new" if new_count else "all cached"
        print(f"    {pdf_file:25s}  {num_pages:2d} pages  ({status})")

    print(f"\n  Done. {total_new} new images, {total_skipped} already existed.\n")


# ---------------------------------------------------------------------------
# Step 2 — seed pipeline cache from prior experiment results
# ---------------------------------------------------------------------------
def seed_cache_from_experiments(results_dir: str, cache_dir: str):
    """Copy experiment results into pipeline/cache/ so those pages are skipped.

    Experiment filenames:  {page}_extractor_{model}.json
                           {page}_supervisor_{model}_[...].json
    Pipeline cache names:  {page}_image_extractor_{model}.json
                           {page}_image_supervisor_{model}.json

    The '_image' infix matches the image stem (e.g. '1700_1_image.png').
    The trailing '[...]' in supervisor names is stripped.
    """
    if not os.path.isdir(results_dir):
        print(f"[Step 2] Experiment results dir not found, skipping: {results_dir}")
        return

    os.makedirs(cache_dir, exist_ok=True)
    copied = skipped = 0

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue

        base = fname[:-5]  # strip .json

        if "_extractor_" in base:
            page, model = base.split("_extractor_", 1)
            cache_name = f"{page}_image_extractor_{model}.json"

        elif "_supervisor_" in base:
            page, rest = base.split("_supervisor_", 1)
            # strip trailing _[gemini-flash_gpt-5-mini] style suffix
            bracket = rest.find("_[")
            model = rest[:bracket] if bracket != -1 else rest
            cache_name = f"{page}_image_supervisor_{model}.json"

        else:
            continue

        dst = os.path.join(cache_dir, cache_name)
        if os.path.exists(dst):
            skipped += 1
            continue

        shutil.copy2(os.path.join(results_dir, fname), dst)
        copied += 1

    print(f"[Step 2] Cache seeded: {copied} files copied, {skipped} already existed.\n")


# ---------------------------------------------------------------------------
# Step 3 — extraction pipeline
# ---------------------------------------------------------------------------
def run_extraction(img_dir: str, output_path: str, cache_dir: str, use_cache: bool):
    sys.path.insert(0, ROOT_DIR)
    from pipeline.run_pipeline import run_pipeline

    print(f"[Step 3] Running extraction pipeline")
    run_pipeline(
        image_paths=[img_dir],
        output_path=output_path,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: PDF conversion + extraction for annual_accounts_1700-1900.",
    )
    parser.add_argument(
        "--no-cache", action="store_true", default=False,
        help="Ignore cached results and re-run all API calls.",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Output Excel path (default: pipeline/output/ledger_<timestamp>.xlsx).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or os.path.join(OUTPUT_DIR, f"ledger_{timestamp}.xlsx")
    use_cache = not args.no_cache

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: PDF → PNG (skips already-converted pages)
    convert_pdfs(PDF_DIR, IMG_DIR)

    # Step 2: Seed cache from prior experiment results (skips re-extraction of sample pages)
    if use_cache:
        seed_cache_from_experiments(SAMPLE_RESULTS, CACHE_DIR)

    # Step 3: Extraction (skips cached pages when use_cache=True)
    run_extraction(IMG_DIR, output_path, CACHE_DIR, use_cache)


if __name__ == "__main__":
    main()
