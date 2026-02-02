# Ledger Extraction Pipeline

A multi-agent system for extracting structured data from scans of 18th–19th century English parish ledgers. Two pipeline architectures are implemented and benchmarked head-to-head against the same ground truth and scoring framework.

**v1** is a sequential skeleton-based pipeline: a Structurer maps the page, an Extractor fills in the numbers, and a Corrector audits the result. **v2** is a competitive pipeline: multiple extractors independently transcribe the full page (structure and amounts together), then a Supervisor agent judges all candidates row-by-row and produces the final merged output.

Both pipelines run across Google Gemini, OpenAI GPT, and Anthropic Claude models. The experiment harness benchmarks every configured combination, caches every intermediate result to disk, and generates side-by-side comparison reports.

---

## Architecture

### v1 — Sequential (skeleton-based)

```
Image
  │
  ▼
┌──────────────┐   skeleton JSON   ┌──────────────┐   filled JSON    ┌──────────────┐
│  Structurer  │ ────────────────► │  Extractor   │ ────────────────►│  Corrector   │
└──────────────┘                   └──────────────┘                  └──────────────┘
  Layout classify                    Column-position OCR               Chain-of-thought audit
  Row count + type skeleton          £ / s / d + fraction fill         Rule-violation fixes
  Description transcription          Confidence scoring                Visual re-verification
```

The Structurer's skeleton is a binding contract — the Extractor can only fill rows that already exist in it. This isolates structural errors from numerical errors, but it also means any row the Structurer misses is permanently lost. The Corrector runs a retry loop: if the combined score after the first correction pass is below 0.7, it receives a concrete report of which rows failed to match and tries again (up to 2 retries).

### v2 — Competitive (no skeleton)

```
Image
  │
  ├──────────────────────────────────────────┐
  ▼                  ▼                       ▼
┌────────────┐  ┌────────────┐   ┌────────────┐
│ Extractor  │  │ Extractor  │   │ Extractor  │   ← each works independently
│ (gemini)   │  │ (gpt)      │   │ (claude)   │     from the raw image
└────────────┘  └────────────┘   └────────────┘
      │                │                │
      ▼                ▼                ▼
┌─────────────────────────────────────────────┐
│              Supervisor                     │   ← sees all candidates + image
│  row-by-row selection from best candidate   │     decides per-row, not wholesale
└─────────────────────────────────────────────┘
```

Each extractor does structure and amounts in one pass. Every row it outputs includes a `confidence_score` and a `notes` field explaining its reasoning — what it saw in the image and why it made each choice. The Supervisor reads all candidates alongside the original image and applies a decision protocol: consensus first, then hard-rule filtering (any candidate row with shillings ≥ 20 or pence ≥ 12 is disqualified), then note-quality assessment, then its own visual reading of the image as a final override. If the Supervisor LLM returns nothing usable, the pipeline falls back to whichever candidate extracted the most rows.

---

## Project Layout

```
ledger-extraction/
├── src/
│   ├── config.py                          # Model registry, pipeline configs, paths
│   ├── clients.py                         # Unified LLM client (OpenAI / Google / Anthropic)
│   ├── agents/
│   │   ├── structurer.py                  # v1 — Agent 1: layout skeleton
│   │   ├── extractor.py                   # v1 — Agent 2: fill skeleton with amounts
│   │   ├── corrector.py                   # v1 — Agent 3: audit + retry loop
│   │   ├── standalone_extractor.py        # v2 — full extraction in one pass
│   │   └── supervisor.py                  # v2 — row-by-row candidate selection
│   ├── prompts/
│   │   ├── structurer.py                  # v1 prompts
│   │   ├── extractor.py                   # v1 prompts
│   │   ├── corrector.py                   # v1 prompts
│   │   ├── standalone_extractor.py        # v2 extractor prompt (mandates notes)
│   │   └── supervisor.py                  # v2 supervisor decision protocol
│   └── evaluation/
│       ├── scorer.py                      # Two-axis scoring (structure + numbers)
│       └── gt_converter.py                # Converts ground_truth.xlsx → per-page JSON
├── data/
│   ├── images/                            # Ledger page scans (PNG/JPG)
│   └── ground_truth/                      # ground_truth.xlsx + exported per-page JSONs
├── experiments/
│   ├── run_experiment.py                  # Unified benchmark runner (v1 + v2)
│   ├── results/
│   │   ├── v1/                            # Cached v1 stage outputs
│   │   └── v2/                            # Cached v2 stage outputs
│   └── reports/                           # Generated CSV reports
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with your API keys (never committed)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=anthropic-...

# 4. Place ledger images in data/images/
# Naming: <sheet_name>_image.png
# Example: 1889_4_image.png  →  matches ground truth sheet "1889_4"

# 5. Export ground truth JSONs (run once; re-run after editing the xlsx)
python -m src.evaluation.gt_converter
```

---

## Running Experiments

```bash
# --- v2 (default) ---
python -m experiments.run_experiment --pipeline v2           # full run, all pages
python -m experiments.run_experiment --pipeline v2 --pages 1889_4 1873_5   # subset
python -m experiments.run_experiment --pipeline v2 --eval-only             # re-score cached only

# --- v1 ---
python -m experiments.run_experiment --pipeline v1           # full run
python -m experiments.run_experiment --pipeline v1 --eval-only

# --- Head-to-head comparison (both pipelines must have reports already) ---
python -m experiments.run_experiment --compare
```

Results are cached under `experiments/results/v1/` and `experiments/results/v2/`. Completed stages are loaded from disk on re-run; delete the folder to force fresh API calls. Each pipeline writes its own report CSV to `experiments/reports/`. The `--compare` flag merges them into a per-page side-by-side table.

### API call cost per page

| Pipeline | Calls per page | Notes |
|---|---|---|
| v1 | 3 (+ up to 2 retries) | Structurer + Extractor + Corrector |
| v2 | 4 | 3 extractors + 1 Supervisor |

---

## Evaluation: Two-Axis Scoring

Both pipelines are scored identically. The scorer is pipeline-agnostic — it takes any `{"rows": [...]}` JSON and a ground truth JSON and produces the same metrics.

### Axis 1 — Structural Accuracy

| Sub-score | What it measures |
|---|---|
| Row count score | `1 - abs(pred_count - gt_count) / gt_count` |
| Type count score | Per-type (header / entry / total) count accuracy, averaged |
| Header text score | Greedy fuzzy match on header descriptions (Levenshtein ratio ≥ 0.8) |

Axis 1 = average of the three sub-scores.

### Axis 2 — Numerical Accuracy *(entry + total rows only)*

Matching is description-aware and greedy: when multiple predicted rows share the same £/s/d triplet and type, the one whose description best fuzzy-matches the ground truth row is consumed first. This prevents cascade failures on pages with duplicate amounts.

| Sub-score | Weight | What it measures |
|---|---|---|
| Amount match | 50% | Exact £/s/d triplet match. Cross-type entry↔total matches score 0.5 instead of 1.0. |
| Amount similarity | 30% | Partial credit: `1 - abs_diff_pence / max_pence`. A 1-penny error on a £50 row scores 0.999. |
| Fraction match | 20% | Exact match on ob/q/3q fractions, scored only on GT rows that have one. |

Axis 2 = `0.5 × match + 0.3 × similarity + 0.2 × fraction`.

### Combined score

`combined = (axis1 + axis2) / 2`

Entry rows with no amounts (sub-items in braced groups) are excluded from Axis 2 but still counted in Axis 1. Predicted rows the LLM mislabelled (e.g. calling an entry a "section_header") are still eligible to match on amounts — the type mismatch is penalised via the cross-type 0.5 credit rule, but correct numbers are never silently dropped.

### Normalisation rules

`0` and `""` (empty) are treated as equivalent in amount fields. LLMs consistently omit zero-valued columns rather than writing 0; ground truth sometimes stores explicit zeros. Collapsing both to the same value prevents false negatives on every row where only one or two of the three columns carry real data.

---

## Image Size Handling

Anthropic enforces a 5 MB limit on base64-encoded images. The client layer handles this transparently: if a scan exceeds the limit, it is re-saved as JPEG at decreasing quality (85 → 75 → 60 → 45) until it fits, then re-encoded. Images that already fit are sent untouched. OpenAI and Google are unaffected.

---

## Adding a New Model

1. Add it to `MODELS` in `src/config.py` with its provider and model name.
2. Add the key to the relevant pipeline config in `PIPELINES` — either the `extractors` list (v2), or the `structurer` / `extractor` / `corrector` lists (v1).
3. Run the experiment. No other changes needed.

## Adding a New Pipeline

Add an entry to `PIPELINES` in `src/config.py` with a unique version key, a description, and whatever stage configuration it needs. Then add a corresponding `run_<version>_pipeline` and `score_<version>` function in `experiments/run_experiment.py` and wire it into the main loop's dispatch. The scorer, clients, and caching infrastructure are all reusable as-is.
