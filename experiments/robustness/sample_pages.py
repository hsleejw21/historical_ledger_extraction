"""
experiments/robustness/sample_pages.py

Draws a stratified random sample of enriched pages for LLM reliability testing.
Strategy: 5 pages per decade across 1700–1909 (20 decades) → ~100 pages.

Why page-level stratification?  Pages vary in era, language mix, and category
composition, so simple uniform sampling risks concentrating in the most-common
mid-century era. Decade stratification ensures coverage across the full 200-year
span and all major structural regimes in the ledger.

Outputs
-------
experiments/results/robustness/sample_pages.json
  {
    "seed": 42,
    "pages_per_decade": 5,
    "n_total": <int>,
    "decade_counts": {<decade>: <n>, ...},
    "page_ids": ["1700_1_image", ...]   # page_id without _enriched.json suffix
  }

Usage
-----
    .venv/bin/python -m experiments.robustness.sample_pages
    .venv/bin/python -m experiments.robustness.sample_pages --pages-per-decade 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_DIR = ROOT / "experiments" / "results" / "enriched"
OUT_DIR = ROOT / "experiments" / "results" / "robustness"

DEFAULT_SEED = 42
DEFAULT_PER_DECADE = 5


def _year_from_filename(name: str) -> int | None:
    """Extract the first 4-digit year from an enriched filename."""
    m = re.match(r"^(\d{4})", name)
    return int(m.group(1)) if m else None


def _page_id_from_filename(name: str) -> str:
    """Strip '_enriched.json' suffix to get the canonical page_id."""
    return name.replace("_enriched.json", "")


def sample_pages(per_decade: int = DEFAULT_PER_DECADE, seed: int = DEFAULT_SEED) -> dict:
    enriched_files = sorted(ENRICHED_DIR.glob("*_image_enriched.json"))
    if not enriched_files:
        raise FileNotFoundError(f"No enriched JSON files in {ENRICHED_DIR}")

    # Group page_ids by decade
    decade_map: dict[int, list[str]] = defaultdict(list)
    for fp in enriched_files:
        yr = _year_from_filename(fp.name)
        if yr is None:
            continue
        decade = (yr // 10) * 10
        decade_map[decade].append(_page_id_from_filename(fp.name))

    # Stratified sample
    rng = random.Random(seed)
    sampled: list[str] = []
    decade_counts: dict[str, int] = {}
    for decade in sorted(decade_map):
        pages = decade_map[decade]
        n = min(per_decade, len(pages))
        chosen = rng.sample(pages, n)
        sampled.extend(chosen)
        decade_counts[str(decade)] = n

    return {
        "seed": seed,
        "pages_per_decade": per_decade,
        "n_total": len(sampled),
        "decade_counts": decade_counts,
        "page_ids": sampled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified page sampler for robustness checks")
    parser.add_argument("--pages-per-decade", type=int, default=DEFAULT_PER_DECADE,
                        help=f"Pages to sample per decade (default: {DEFAULT_PER_DECADE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = sample_pages(per_decade=args.pages_per_decade, seed=args.seed)

    out_path = OUT_DIR / "sample_pages.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"Sampled {result['n_total']} pages across {len(result['decade_counts'])} decades "
          f"({args.pages_per_decade}/decade, seed={args.seed})")
    print(f"Decade breakdown: { {k: v for k, v in result['decade_counts'].items()} }")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
