#!/usr/bin/env bash
# =============================================================================
# run_robustness.sh
# Robustness check pipeline for the historical ledger analysis.
#
# HOW TO RUN IN TMUX
# ------------------
#   # Start a new named tmux session and run the full pipeline:
#   tmux new-session -d -s robustness \
#     "cd /Users/jungwoo.hong/Documents/Miscellaneous/historical_ledger_extraction/historical_ledger_extraction && bash run_robustness.sh 2>&1 | tee robustness.log"
#
#   # Attach to watch progress:
#   tmux attach -t robustness
#
#   # Detach at any time with: Ctrl-b d
#
# WHAT IT DOES
# ------------
# Step 1  Sample ~100 pages (stratified by decade) from the 1,581 enriched pages.
# Step 2a Re-run enrichment on sampled pages using Claude Haiku 4.5.
# Step 2b Re-run enrichment on sampled pages using Gemini 2.5 Flash.
# Step 3  Compute inter-model reliability metrics:
#           - Cohen's κ (pairwise) for direction, category, language, payment_period
#           - Fleiss' κ (all 3 models simultaneously)
#           - % exact agreement per field
#           - Bootstrap 95% CI on aggregate statistics
# Step 4  Measurement validation (parameter sensitivity):
#           - Era boundary sensitivity (4 schemes)
#           - Price deflation sensitivity (Clark/baseline, O'Donoghue-Feinstein, nominal)
#           - Year-weight sensitivity (equal, first-year, last-year)
#           - Arrears treatment (include vs exclude)
#           - Section-header direction validation
#           - Change-point threshold sensitivity (z=2.0, 2.5, 3.0)
# Step 5  Generate combined HTML report.
#
# OUTPUTS
# -------
# Raw enrichment reruns:    experiments/results/robustness/{haiku45,gemini25flash}/
# Reliability CSVs:         experiments/reports/robustness/reliability/
# Measurement CSVs:         experiments/reports/robustness/measurement/
# HTML report:              experiments/reports/robustness/robustness_report.html
#
# NOTES
# -----
# - Steps 2a/2b make real API calls (~105 pages × 2 models).
#   Estimated cost: a few USD total.
# - Steps 2a/2b are resumable: already-processed pages are skipped.
#   Rerun with --force to overwrite.
# - Step 3 requires both model reruns to have completed.
# - Step 4 runs entirely on existing enriched data (no API calls).
# =============================================================================

set -euo pipefail

PYTHON=".venv/bin/python"
LOG_DIR="experiments/reports/robustness"

# Verify we're in the right directory
if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: .venv/bin/python not found."
    echo "Please run this script from the project root:"
    echo "  cd /Users/jungwoo.hong/Documents/Miscellaneous/historical_ledger_extraction/historical_ledger_extraction"
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "========================================================"
echo " ROBUSTNESS PIPELINE"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 1: Sample pages ---"
echo "Stratified sample: 5 pages/decade × 20 decades ≈ 100 pages"
$PYTHON -m experiments.robustness.sample_pages
echo "Step 1 complete."

# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 2a: Re-run enrichment — Claude Haiku 4.5 ---"
$PYTHON -m experiments.robustness.rerun_enrichment --model haiku45
echo "Step 2a complete."

echo ""
echo "--- STEP 2b: Re-run enrichment — Gemini 2.5 Flash ---"
$PYTHON -m experiments.robustness.rerun_enrichment --model gemini25flash
echo "Step 2b complete."

# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 3: Reliability metrics ---"
echo "Cohen's κ, Fleiss' κ, agreement rates, bootstrap CIs (B=1000)"
$PYTHON -m experiments.robustness.reliability_metrics --n-bootstrap 1000
echo "Step 3 complete."

# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 4: Measurement validation ---"
echo "Era / deflation / year-weight / arrears / header / change-point sensitivity"
$PYTHON -m experiments.robustness.measurement_validation
echo "Step 4 complete."

# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 5: Generate HTML report ---"
$PYTHON -m experiments.robustness.robustness_report
echo "Step 5 complete."

# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " DONE  $(date '+%Y-%m-%d %H:%M:%S')"
echo " Report: experiments/reports/robustness/robustness_report.html"
echo "========================================================"
