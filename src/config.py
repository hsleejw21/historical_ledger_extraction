"""
src/config.py
Single source of truth: API keys, model registry, agent role assignments, and all paths.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Model Registry
# Each entry: { "provider": str, "model_name": str }
# ---------------------------------------------------------------------------
MODELS = {
    # Google
    "gemini-flash":      {"provider": "google",    "model_name": "gemini-2.5-flash"},

    # OpenAI
    "gpt-4o-mini":       {"provider": "openai",    "model_name": "gpt-4o-mini"},
    "gpt-4.1-mini":      {"provider": "openai",    "model_name": "gpt-4.1-mini"},

    # Anthropic
    "claude-haiku":      {"provider": "anthropic", "model_name": "claude-haiku-4-5-20251001"},
    "claude-sonnet":     {"provider": "anthropic", "model_name": "claude-sonnet-4-5-20250929"},
}

# ---------------------------------------------------------------------------
# Agent Role Assignments
#
# Each agent role maps to a LIST of model keys to benchmark.
# The experiment runner iterates over every combination.
#
# STRUCTURER  — scouts the page, outputs the skeleton.  Needs good spatial
#               reasoning but the output is lightweight, so cheaper models work.
# EXTRACTOR   — fills £/s/d into the skeleton.  This is the heaviest OCR task;
#               accuracy matters most here.
# CORRECTOR   — audits & fixes with chain-of-thought.  Needs strong reasoning
#               but only runs once per page, so a stronger model is justified.
# ---------------------------------------------------------------------------
AGENT_ROLES = {
    "structurer": ["gemini-flash", "claude-haiku", "gpt-4o-mini"],
    "extractor":  ["gemini-flash", "claude-haiku", "gpt-4.1-mini", "claude-sonnet"],
    "corrector":  ["claude-sonnet", "gpt-4.1-mini", "gemini-flash"],
}

# ---------------------------------------------------------------------------
# Pipeline Registry
#
# Each pipeline is a named configuration.  The experiment runner dispatches
# on the "version" key.  Adding a new pipeline here is all it takes to make
# it available via --pipeline on the CLI.
#
# v1 — original 3-agent sequential pipeline (structurer → extractor → corrector)
# v2 — competitive extraction + supervisor  (N extractors → supervisor)
# ---------------------------------------------------------------------------
PIPELINES = {
    "v1": {
        "version": "v1",
        "description": "Structurer → Extractor → Corrector (skeleton-based)",
        "stages": ["structurer", "extractor", "corrector"],
        "structurer": ["gemini-flash", "claude-haiku", "gpt-4o-mini"],
        "extractor":  ["gemini-flash", "claude-haiku", "gpt-4.1-mini", "claude-sonnet"],
        "corrector":  ["claude-sonnet", "gpt-4.1-mini", "gemini-flash"],
    },
    "v2": {
        "version": "v2",
        "description": "Competitive extraction (no skeleton) + Supervisor",
        "stages": ["extractors", "supervisor"],
        # Every model here independently extracts the full page (structure + amounts).
        # The supervisor sees all candidates and picks the best row-by-row.
        "extractors": ["gemini-flash", "gpt-4.1-mini", "claude-sonnet"],
        # Supervisor model — needs strong reasoning; one model is enough.
        "supervisor": ["claude-sonnet"],
    },
}

# Legacy alias so old code importing AGENT_ROLES still works.
AGENT_ROLES = PIPELINES["v1"]

# ---------------------------------------------------------------------------
# Paths  (all relative to the repo root)
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
DATA_DIR     = os.path.join(BASE_DIR, "data", "images")
GT_DIR       = os.path.join(BASE_DIR, "data", "ground_truth")
RESULTS_DIR  = os.path.join(BASE_DIR, "experiments", "results")   # contains v1/ and v2/
REPORT_DIR   = os.path.join(BASE_DIR, "experiments", "reports")
