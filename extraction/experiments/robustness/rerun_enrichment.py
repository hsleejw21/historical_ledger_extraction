"""
experiments/robustness/rerun_enrichment.py

Re-runs the enrichment pipeline on the sampled pages using two alternative
models to test LLM reliability (inter-model agreement).

The original pipeline used gpt-5-mini.  We additionally test:

haiku45
    Anthropic Claude Haiku 4.5 — a different provider and architecture from
    the original OpenAI model.  Called via the native Anthropic SDK.
    Tests cross-provider consistency between OpenAI and Anthropic model families.

gemini25flash
    Google Gemini 2.5 Flash via the OpenAI-compatible REST endpoint.
    Tests cross-provider, cross-architecture consistency with a third
    independent model family (Google DeepMind).

Combined with the original gpt-5-mini run, this gives 3 raters spanning
three entirely independent providers (OpenAI, Anthropic, Google), which is
the strongest possible inter-model reliability design.

Why this approach?
    Mirrors the "LLM-as-annotator" reliability paradigm (Gilardi et al. 2023;
    Moller et al. 2023; Ding et al. 2023).  High cross-model agreement
    (kappa > 0.80) across three independent provider architectures constitutes
    strong evidence that labels reflect genuine signals in the historical data
    rather than model-specific biases.

Outputs
-------
experiments/results/robustness/<model>/<page_id>_enriched.json

Usage
-----
    .venv/bin/python -m experiments.robustness.rerun_enrichment --model haiku45
    .venv/bin/python -m experiments.robustness.rerun_enrichment --model gemini25flash
    .venv/bin/python -m experiments.robustness.rerun_enrichment --model haiku45 --force
    .venv/bin/python -m experiments.robustness.rerun_enrichment --model haiku45 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic as anthropic_sdk
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR   = ROOT / "experiments" / "results" / "cache"
SAMPLE_FILE = ROOT / "experiments" / "results" / "robustness" / "sample_pages.json"
ROBUSTNESS_DIR = ROOT / "experiments" / "results" / "robustness"

# ---------------------------------------------------------------------------
# Import shared logic from the original enrichment script
# ---------------------------------------------------------------------------
# These imports pull in prompt strings, validation logic, and helper functions
# without re-running any API calls (the original script is guarded by __main__).
from experiments.enrich_supervisor_rows import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    ENRICHMENT_FIELDS,
    VALID_VALUES,
    FALLBACK,
    propagate_section_headers,
    strip_for_llm,
    validate_row,
    merge_enrichment,
)

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS: dict[str, dict] = {
    "haiku45": {
        "model_id":    "claude-haiku-4-5-20251001",
        "api_key_env": "ANTHROPIC_API_KEY",
        "backend":     "anthropic",   # use Anthropic SDK, not OpenAI client
        "description": "Anthropic Claude Haiku 4.5",
    },
    "gemini25flash": {
        "model_id":    "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "backend":     "openai",
        "base_url":    "https://generativelanguage.googleapis.com/v1beta/openai/",
        "description": "Google Gemini 2.5 Flash via OpenAI-compatible endpoint",
        "extra_kwargs": {},
    },
}


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _get_api_key(env_var: str) -> str:
    key = os.environ.get(env_var) or os.environ.get(env_var + " ")
    if not key:
        sys.exit(f"API key not found in environment: {env_var}")
    return key.strip()


def build_client(config: dict) -> object:
    """Return the appropriate client object for the configured backend."""
    api_key = _get_api_key(config["api_key_env"])
    if config["backend"] == "anthropic":
        return anthropic_sdk.Anthropic(api_key=api_key)
    # OpenAI-compatible (OpenAI or Gemini)
    kwargs: dict = {"api_key": api_key}
    if config.get("base_url"):
        kwargs["base_url"] = config["base_url"]
    return OpenAI(**kwargs)


def _parse_llm_text(text: str) -> list[dict]:
    """Parse a JSON string returned by any LLM into a list of row dicts."""
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    return parsed


def call_llm(
    client: object,
    config: dict,
    rows_json: str,
    dry_run: bool = False,
) -> list[dict]:
    """Call the LLM using the appropriate backend and return enriched row list."""
    prompt = USER_PROMPT_TEMPLATE.format(rows_json=rows_json)
    model_id = config["model_id"]

    if dry_run:
        print(f"  [dry-run] would call model={model_id} via {config['backend']}")
        print(f"  prompt length: {len(prompt)} chars")
        return []

    if config["backend"] == "anthropic":
        # Anthropic Messages API
        response = client.messages.create(
            model=model_id,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return _parse_llm_text(text)

    # OpenAI-compatible backend (OpenAI or Gemini)
    create_kwargs: dict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "timeout": 180,
        **(config.get("extra_kwargs") or {}),
    }
    try:
        create_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**create_kwargs)
    except Exception as e:
        if "response_format" in str(e).lower() or "json_object" in str(e).lower():
            del create_kwargs["response_format"]
            response = client.chat.completions.create(**create_kwargs)
        else:
            raise
    text = response.choices[0].message.content.strip()
    return _parse_llm_text(text)


def get_supervisor_file(page_id: str) -> Path | None:
    """Return the supervisor JSON for a given page_id (e.g. '1700_1_image')."""
    candidate = CACHE_DIR / f"{page_id}_supervisor_gemini-flash.json"
    return candidate if candidate.exists() else None


def process_page(
    page_id: str,
    client: object,
    config: dict,
    output_dir: Path,
    batch_size: int = 30,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Enrich a single page.  Returns True if processed, False if skipped."""
    out_path = output_dir / f"{page_id}_enriched.json"

    if out_path.exists() and not force:
        print(f"  [skip] {page_id} — already done (use --force to overwrite)")
        return False

    sup_file = get_supervisor_file(page_id)
    if sup_file is None:
        print(f"  [warn] {page_id} — supervisor file not found in cache, skipping")
        return False

    with open(sup_file, encoding="utf-8") as fh:
        data = json.load(fh)

    rows: list[dict] = data.get("rows", [])
    meta: dict       = data.get("_meta", {})

    rows = propagate_section_headers(rows)
    llm_input_rows = [strip_for_llm(r) for r in rows]

    all_enriched: list[dict] = []
    for i in range(0, len(llm_input_rows), batch_size):
        batch = llm_input_rows[i : i + batch_size]
        rows_json = json.dumps(batch, indent=2, ensure_ascii=False)

        try:
            enriched_batch = call_llm(client, config, rows_json, dry_run=dry_run)
        except Exception as e:
            print(f"  [error] {page_id} batch {i // batch_size}: {e}")
            enriched_batch = [{"row_index": r["row_index"]} for r in batch]

        all_enriched.extend(enriched_batch)

        if dry_run:
            break

        if i + batch_size < len(llm_input_rows):
            time.sleep(0.3)

    if dry_run:
        return True

    merged = merge_enrichment(rows, all_enriched, page_id=page_id)

    output = {
        "page_id": page_id,
        "rows": merged,
        "_meta": {
            **meta,
            "robustness_model":   config["model_id"],
            "robustness_backend": config["backend"],
            "enrichment_fields":  ENRICHMENT_FIELDS,
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"  [done] {page_id} -> {out_path.name}  ({len(merged)} rows)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run enrichment on sampled pages with an alternative model/temperature"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Which model configuration to run",
    )
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--page-sleep", type=float, default=1.0,
                        help="Sleep between pages (seconds)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt for first page/batch without calling API")
    args = parser.parse_args()

    load_env()

    if not SAMPLE_FILE.exists():
        sys.exit(f"Sample file not found: {SAMPLE_FILE}\nRun sample_pages.py first.")

    with open(SAMPLE_FILE, encoding="utf-8") as fh:
        sample = json.load(fh)

    page_ids: list[str] = sample["page_ids"]
    config = MODEL_CONFIGS[args.model]

    print(f"Model config: {config['description']}")
    print(f"Pages to process: {len(page_ids)}")

    output_dir = ROBUSTNESS_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(config)

    n_done = 0
    for i, page_id in enumerate(page_ids):
        print(f"[{i+1}/{len(page_ids)}] {page_id}")
        processed = process_page(
            page_id=page_id,
            client=client,
            config=config,
            output_dir=output_dir,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            force=args.force,
        )
        if processed:
            n_done += 1

        if not args.dry_run and processed and i < len(page_ids) - 1:
            time.sleep(args.page_sleep)

        if args.dry_run:
            break  # one page is enough for dry-run

    print(f"\nDone. Processed {n_done} pages -> {output_dir}")


if __name__ == "__main__":
    main()
