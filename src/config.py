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
    "gpt-4o":       {"provider": "openai",    "model_name": "gpt-4o"},
    "gpt-5-mini":      {"provider": "openai",    "model_name": "gpt-5-mini"},

    # Anthropic
    "claude-haiku":      {"provider": "anthropic", "model_name": "claude-haiku-4-5"},
    "claude-sonnet":     {"provider": "anthropic", "model_name": "claude-sonnet-4-5"},
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
    "structurer": ["gemini-flash"],
    "extractor":  ["gemini-flash", "gpt-4o", "gpt-5-mini", "claude-sonnet"],
    "corrector":  ["claude-sonnet", "gpt-5-mini", "gemini-flash"],
}

# ---------------------------------------------------------------------------
# Paths  (all relative to the repo root)
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
DATA_DIR     = os.path.join(BASE_DIR, "data", "images")
GT_DIR       = os.path.join(BASE_DIR, "data", "ground_truth")
RESULTS_DIR  = os.path.join(BASE_DIR, "experiments", "results")
REPORT_DIR   = os.path.join(BASE_DIR, "experiments", "reports")
