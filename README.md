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
├── extraction/                    # OCR & extraction pipeline
│   ├── pipeline/                  # Production pipeline (start here)
│   │   └── run_pipeline.py        # v2_no_claude runner → Excel output
│   ├── src/                       # Core library
│   │   ├── agents/                # Extractor & supervisor agents
│   │   ├── prompts/               # System prompts
│   │   ├── evaluation/            # Axis1/Axis2 scoring
│   │   ├── clients.py             # Unified LLM client
│   │   ├── config.py              # Model registry & pipeline configs
│   │   └── validation.py          # Currency rule checks (£/s/d)
│   ├── enrichment/                # LLM enrichment of extracted rows
│   ├── experiments/               # Extraction experiments
│   │   ├── run_experiment.py      # Benchmark runner (v1–v6)
│   │   ├── robustness/            # Robustness testing
│   │   └── v6_loocv/              # CLIP-based adaptive routing
│   ├── results/                   # Extraction outputs
│   │   ├── enriched/              # 1,581 enriched JSON files
│   │   └── cache/                 # Intermediate JSONs
│   ├── tools/                     # Utility scripts (Excel export)
│   ├── run_all.py                 # One-shot: PDF → extraction
│   └── run_robustness.sh
│
├── analysis/                      # Downstream analysis (versioned)
│   ├── v1/ ~ v18/                 # Each version: script + results together
│   │   ├── analysis_v{n}.py
│   │   └── *.csv, *.png, *.html   # Outputs
│   ├── shared/                    # Common analysis scripts
│   │   ├── analysis_embeddings.py
│   │   ├── analysis_vocabulary.py
│   │   ├── analysis_text_trends.py
│   │   └── paper_figures.py
│   └── README.md                  # Script documentation
│
├── data/
│   ├── ground_truth/              # Manual annotations (33 pages)
│   └── visual_features/           # CLIP embeddings (1,581 pages)
│
└── requirements.txt
```

---

## Key Commands

### Production Extraction

```bash
# Process images → Excel
python extraction/pipeline/run_pipeline.py --images data/images/
python extraction/pipeline/run_pipeline.py --images data/images/ --use-cache

# Full pipeline: PDF conversion + extraction
python extraction/run_all.py
```

### Extraction Experiments

```bash
python -m extraction.experiments.run_experiment --pipeline v2_no_claude
python -m extraction.experiments.run_experiment --pipeline v2_no_claude --pages 1700_7
python -m extraction.experiments.run_experiment --compare-ablations
```

### Enrichment

```bash
python extraction/enrichment/enrich_supervisor_rows.py
```

### Analysis

```bash
# Run latest analysis (v18)
python analysis/v18/analysis_v18.py

# Run specific version
python analysis/v15/analysis_v15.py

# Shared scripts
python analysis/shared/analysis_embeddings.py
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

**Status:** 1,581 pages enriched (1700–1900), stored in `extraction/results/enriched/`.

---

## Analysis Versions (v4–v18)

The versioned analysis pipeline applies a Four-Level Transformation Framework to Oxford University's financial records (1700–1900).

| Version | Focus |
|---------|-------|
| v4–v8 | Foundation: Financial restructuring, OLS, four-level proxies (L1–L4) |
| v9–v11 | Event-time plots, Reform Acts, strength assessment |
| v14–v15 | L4 split, portfolio analysis, reallocation |
| v16–v17 | Spending-income gap, organizational decision rules |
| v18 | **Current** — Latest analysis |

See `analysis/README.md` for full documentation.

---

**Last Updated:** 2026-07-28
**SOTA:** v2_no_claude (0.8385 combined)
**Analysis:** v18 (latest)
