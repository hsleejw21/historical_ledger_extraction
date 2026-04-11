# CLAUDE.md — Historical Ledger Extraction

This file provides context for Claude Code sessions in this repository.

---

## What This Project Does

Extracts structured financial data (£/s/d amounts, row types, descriptions) from scanned 18th–19th century English parish accounting ledgers using a multi-agent LLM pipeline. After extraction, an enrichment pipeline adds semantic metadata (economic category, direction, people/place names, etc.) to each row. Analysis scripts then explore temporal, linguistic, and economic patterns across the full 1700–1900 corpus.

---

## Current State (as of 2026-04-11)

- **Extraction:** Production-ready. SOTA pipeline is `v2_no_claude` (gemini-flash + gpt-5-mini → gemini-flash supervisor). Combined score **0.8385**, axis1 0.8255, axis2 0.8515.
- **Enrichment:** Complete. All 1,581 pages (1700–1900) enriched with semantic metadata. Results in `experiments/results/enriched/`.
- **Analysis:** In progress. Temporal, embedding, and category analyses done. Next phase: find additional features from enriched data.

---

## Repository Layout

```
pipeline/               Production pipeline (run this for new images)
src/                    Core library: agents, prompts, evaluation, clients, config
data/                   Ground truth (33 pages) + visual features (1,581 pages)
tools/                  Utility scripts (Excel export)
experiments/
  run_experiment.py     Extraction benchmark runner (v1–v6 ablations)
  enrichment/           Enrichment pipeline scripts
  analysis/             Downstream analysis and visualisation scripts
  robustness/           Robustness testing framework
  v6_loocv/             Phase 6: CLIP-based adaptive routing (active research)
  results/              All experiment outputs (v1/, v2/, enriched/, etc.)
  reports/              Generated HTML/CSV/PNG reports
```

---

## Key Commands

```bash
# Production extraction (new ledger images → Excel)
python pipeline/run_pipeline.py --images data/images/
python pipeline/run_pipeline.py --images data/images/ --use-cache

# Run extraction experiment (against 33 ground-truth pages)
python -m experiments.run_experiment --pipeline v2_no_claude
python -m experiments.run_experiment --pipeline v2_no_claude --pages 1700_7  # single page
python -m experiments.run_experiment --compare-ablations

# Run enrichment (adds semantic fields to extracted rows)
python experiments/enrichment/enrich_supervisor_rows.py

# Run analysis
python experiments/analysis/analysis_enriched.py
python experiments/analysis/embedding_analysis_enriched.py
python experiments/analysis/reanalysis_ledger_yearly_v2.py
```

---

## Architecture: SOTA Pipeline (v2_no_claude)

```
Image
  ├──────────────────────┐
  ▼                      ▼
gemini-flash          gpt-5-mini
Extractor             Extractor
  └──────────┬──────────┘
             ▼
     gemini-flash Supervisor
     (row-by-row arbitration)
             ▼
     Structured JSON → Excel
```

**Do NOT add Claude as an extractor.** It was tested and hurts both accuracy and cost (v2 vs v2_no_claude).

---

## Key Files

| File | Purpose |
|------|---------|
| `src/config.py` | Model registry, pipeline configs, all path constants |
| `src/clients.py` | Unified LLM client — add new models here |
| `src/agents/standalone_extractor.py` | Extractor agent used in v2 pipeline |
| `src/agents/supervisor.py` | Supervisor agent — core of SOTA pipeline |
| `src/evaluation/scorer.py` | Two-axis scoring (axis1=structure, axis2=numbers) |
| `pipeline/run_pipeline.py` | Production entry point |
| `experiments/run_experiment.py` | Research/benchmark entry point |
| `experiments/enrichment/enrich_supervisor_rows.py` | Enrichment pipeline |

---

## Code Conventions

### Path Resolution

All scripts resolve the project root relative to `__file__`. The correct depth depends on where the script lives:

| Location | ROOT expression |
|----------|----------------|
| `experiments/*.py` | `Path(__file__).resolve().parents[1]` |
| `experiments/analysis/*.py` | `Path(__file__).resolve().parents[2]` |
| `experiments/enrichment/*.py` | `Path(__file__).parent.parent.parent` |
| `experiments/robustness/*.py` | `Path(__file__).resolve().parents[2]` |

Always write outputs to `experiments/reports/<name>/` and use `OUT_DIR.mkdir(parents=True, exist_ok=True)`.

### Adding a New Analysis Script

1. Place the script in `experiments/analysis/`
2. Use `ROOT = Path(__file__).resolve().parents[2]` for path resolution
3. Read from `experiments/results/enriched/` (enriched data) or `experiments/results/cache/` (raw supervisor output)
4. Write reports to `experiments/reports/<your-analysis-name>/`
5. No need to modify any other files

### Adding a New Extraction Experiment

1. Create new branch: `git checkout -b experiments/<name>`
2. Add agent in `src/agents/`, prompts in `src/prompts/`
3. Register pipeline in `src/config.py` under `PIPELINES`
4. Add runner in `experiments/run_experiment.py`
5. Test: `python -m experiments.run_experiment --pipeline <name> --pages 1700_7`
6. Full run + compare: `python -m experiments.run_experiment --compare-ablations`
7. Merge to `main` only if improvement ≥ 0.01 on combined score

---

## Data

### Ground Truth
- `data/ground_truth/ground_truth.xlsx` — 33 pages, manually annotated
- Parsed by `src/evaluation/gt_converter.py`
- Used by `src/evaluation/scorer.py` for axis1/axis2 scoring

### Enriched Data
- `experiments/results/enriched/` — 1,581 JSON files, one per ledger page
- Each file: list of rows with extraction fields + semantic enrichment fields
- Enrichment fields: `direction`, `category`, `language`, `english_description`, `place_name`, `person_name`, `payment_period`, `is_arrears`, `is_signature`

### Raw Supervisor Cache
- `experiments/results/cache/` — intermediate extractor + supervisor JSONs
- Named: `<page>_image_extractor_<model>.json`, `<page>_image_supervisor_<model>.json`

### Visual Features
- `data/visual_features/clip_embeddings.json` — CLIP vectors (1,581 pages)
- `data/visual_features/visual_features.json` — additional CV features
- Used by `experiments/v6_loocv/` for adaptive routing

---

## Currency System

- **£** (pounds), **s** (shillings: 0–19), **d** (pence: 0–11)
- **ob** = obolus = ½ penny (looks like 'd' in handwriting)
- Fractions stored as float: ¼ = 0.25, ½ = 0.5, ¾ = 0.75
- Validation rules enforced by `src/validation.py`

---

## What Not to Do

- **Don't add Claude as an extractor** — tested in v2 vs v2_no_claude, it hurts accuracy.
- **Don't run ablations on main** — use a feature branch, merge only on ≥0.01 improvement.
- **Don't put analysis scripts in `experiments/` root** — put them in `experiments/analysis/`.
- **Don't hardcode absolute paths** — always resolve relative to `__file__` using the table above.
- **Don't commit `.env`** — it's in `.gitignore` for a reason.
