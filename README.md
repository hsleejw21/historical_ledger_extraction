# Historical Ledger Extraction Pipeline

Automated extraction and analysis of 18th–19th century English parish accounting ledgers using a multi-agent LLM architecture.

---

## Project Overview

**Goal:** Extract structured data (£/s/d amounts, descriptions, row types) from scanned historical accounting ledgers, then enrich and analyse the records to surface economic and social patterns across 1700–1900.

**Dataset:**
- 33 research pages (ground-truth annotated, used for evaluation)
- 1,581 enriched pages (1700–1900, full corpus with semantic metadata)

**Current Best Performance (SOTA):**

| Metric | Value |
|--------|-------|
| Pipeline | v2_no_claude (gemini-flash + gpt-5-mini → gemini-flash Supervisor) |
| Combined Score | 0.8385 |
| Axis 1 (Structure) | 0.8255 |
| Axis 2 (Numerical) | 0.8515 |

---

## Setup

### Prerequisites
- Python 3.8+
- API keys for OpenAI and Google AI (Anthropic optional — not used in SOTA pipeline)

### Installation

```bash
git clone <repository-url>
cd historical_ledger_extraction

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...  # optional
```

---

## Project Structure

```
historical_ledger_extraction/
│
├── pipeline/                       # Production pipeline (start here)
│   ├── run_pipeline.py             # v2_no_claude runner → Excel output
│   ├── cache/                      # Intermediate extractor/supervisor JSONs
│   └── output/                     # Final Excel files
│
├── run_all.py                      # One-shot: PDF conversion + full extraction
│
├── tools/
│   └── export_to_excel.py          # Convert any results dir → Excel
│
├── data/
│   ├── ground_truth/
│   │   └── ground_truth.xlsx       # Manual annotations (33 pages)
│   └── visual_features/
│       ├── clip_embeddings.json    # CLIP visual feature vectors (1,581 pages)
│       └── visual_features.json    # Additional CV features
│
├── src/                            # Core library
│   ├── agents/                     # Extractor & supervisor agents
│   ├── prompts/                    # System prompts
│   ├── evaluation/                 # Axis1/Axis2 scoring
│   ├── clients.py                  # Unified LLM client (OpenAI, Google, Anthropic)
│   ├── config.py                   # Model registry & pipeline configurations
│   └── validation.py               # Currency rule checks (£/s/d)
│
└── experiments/                    # Research & analysis
    ├── run_experiment.py           # Extraction benchmark runner (v1–v6)
    │
    ├── enrichment/
    │   └── enrich_supervisor_rows.py  # LLM enrichment of extracted rows
    │
    ├── analysis/                   # Downstream analysis & visualisation
    │   ├── analysis_v4.py          # Financial restructuring, vocabulary, networks
    │   ├── analysis_v5.py          # Outcome variables, OLS, income composition
    │   ├── analysis_v6.py          # Four-level proxy construction (L1–L4)
    │   ├── analysis_v7.py          # ITS / RDD / Granger / placebo multi-method
    │   ├── analysis_v8.py          # L1–L4 operationalization + sequential testing
    │   ├── analysis_embeddings.py
    │   ├── analysis_hypotheses.py
    │   ├── analysis_supplier_networks.py
    │   ├── analysis_text_trends.py
    │   ├── analysis_vocabulary.py
    │   ├── reanalysis_ledger_yearly_v2.py  # Yearly aggregation (feeds v4–v8)
    │   └── misc/                   # Experimental / one-off scripts
    │
    ├── robustness/                 # Robustness testing framework
    ├── v6_loocv/                   # CLIP-based adaptive routing (active research)
    │
    ├── results/                    # Experiment outputs (large files gitignored)
    │   ├── enriched/               # LLM-enriched rows (1,581 pages)
    │   └── sample_pdf/             # JSON cache for sample pages (tracked)
    │
    └── reports/                    # Generated HTML/CSV/PNG reports (gitignored)
```

---

## Key Commands

### Production Extraction

```bash
# Process images → Excel
python pipeline/run_pipeline.py --images data/images/
python pipeline/run_pipeline.py --images data/images/ --use-cache

# Full pipeline: PDF conversion + extraction
python run_all.py
python run_all.py --no-cache  # force re-extraction
```

### Extraction Experiments

```bash
python -m experiments.run_experiment --pipeline v2_no_claude
python -m experiments.run_experiment --pipeline v2_no_claude --pages 1700_7
python -m experiments.run_experiment --compare-ablations
```

### Enrichment

```bash
python experiments/enrichment/enrich_supervisor_rows.py
```

### Analysis

```bash
# Versioned analysis pipeline (run latest)
python experiments/analysis/analysis_v8.py

# Topic-specific analyses
python experiments/analysis/analysis_embeddings.py
python experiments/analysis/analysis_text_trends.py
python experiments/analysis/analysis_supplier_networks.py

# Yearly aggregation (prerequisite for v4–v8)
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

**Do not add Claude as an extractor** — tested in v2 vs v2_no_claude; it reduces accuracy and saves no cost.

---

## Evaluation Metrics

**Axis 1 (Structural Accuracy, 40%):** Row count, row type counts, header text fuzzy matching (>80% similarity).

**Axis 2 (Numerical Accuracy, 60%):** Exact £/s/d match (50%), amount similarity within 5% (30%), fraction match (20%).

**Combined Score:** `(axis1 + axis2) / 2`

---

## Enrichment Fields

After extraction, each row is enriched with semantic metadata:

| Field | Values |
|-------|--------|
| `direction` | income / expenditure / transfer / balance_sheet / unclear |
| `category` | land_rent / ecclesiastical / maintenance / salary_stipend / administrative / educational / financial / domestic / charitable / other |
| `language` | latin / english / mixed |
| `payment_period` | half_year / annual / one_off / … |
| `is_arrears`, `is_signature` | true / false |
| `place_name`, `person_name` | normalised names |
| `english_description` | plain-English gloss |

**Status:** 1,581 pages enriched (1700–1900), stored in `experiments/results/enriched/`.

---

## Analysis Progression (v4–v8)

The versioned analysis pipeline applies a Four-Level Transformation Framework to Oxford University's financial records (1700–1900), testing whether the institution underwent a sequential institutional transformation analogous to AI adoption patterns.

| Version | Focus |
|---------|-------|
| v4 | Financial restructuring, vocabulary change, supplier networks, text trends |
| v5 | Outcome variables, OLS regression specs, robustness, income composition |
| v6 | Four-level proxy construction (L1 efficiency, L2 process, L3 capability, L4 mission) |
| v7 | ITS / RDD / Granger causality / placebo multi-method causal analysis |
| v8 | Explicit L1–L4 operationalization, sequential transformation testing, capability vs mission decomposition |

Reports generated to `experiments/reports/analysis_v{n}/`.

---

## Git Workflow

```bash
# Always branch from main for new experiments
git checkout main
git checkout -b experiments/new-feature-name

# Test on one page before full run
python -m experiments.run_experiment --pipeline <name> --pages 1700_7

# Merge to main only if improvement ≥ 0.01 on combined score
```

---

**Last Updated:** 2026-05-14
**SOTA:** v2_no_claude (0.8385 combined, 0.8515 axis2)
**Status:** Active — enrichment complete (1,581 pages), analysis v8 in progress
