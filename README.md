# Historical Ledger Extraction Pipeline

Automated extraction and analysis of 18th–19th century English parish accounting ledgers using a multi-agent LLM architecture.

**Table of Contents:**
- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Evolution & Results](#-evolution--results)
- [Current SOTA](#-current-sota-v2_no_claude)
- [Evaluation Metrics](#-evaluation-metrics)
- [Technical Challenges](#-key-technical-challenges-solved)
- [Production Pipeline](#-production-pipeline)
- [Enrichment Pipeline](#-enrichment-pipeline)
- [Analysis](#-analysis)
- [Tools](#-tools)
- [Quick Start](#-quick-start)
- [Git Workflow](#-git-workflow)

---

## 📊 Project Overview

**Goal:** Extract structured data (£/s/d amounts, descriptions, row types) from scanned historical accounting ledgers, then enrich and analyse the extracted records to surface economic and social patterns across 1700–1900.

**Dataset:**
- **33 research pages** (ground-truth annotated, used for evaluation)
- **1,581 enriched pages** (1700–1900, full ledger corpus with semantic metadata)

Ground-truth pages contain:
- 813 entry rows with amounts
- 74 amount-less entries (section headers)
- 26 sheets with totals
- 127 fractional pence values (¼, ½, ¾)

**Current Best Performance (SOTA):**
| Metric | Value |
|--------|-------|
| **Pipeline** | v2_no_claude (gemini-flash + gpt-5-mini → Supervisor) |
| **Combined Score** | 0.8385 |
| **Axis 1 (Structure)** | 0.8255 |
| **Axis 2 (Numerical)** | 0.8515 |
| **Cost Advantage** | 33% cheaper than v2 (no Claude) |

---

## 🔧 Setup & Installation

### **Prerequisites**
- Python 3.8+
- API keys for: OpenAI, Google AI (Anthropic optional — not used in SOTA pipeline)

### **Installation**

```bash
git clone <repository-url>
cd historical_ledger_extraction

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### **Configuration**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...  # optional
```

The config is automatically loaded by `src/config.py`.

---

## 🏗️ Project Structure

```
historical_ledger_extraction/
│
├── pipeline/                          # ← PRODUCTION PIPELINE (start here)
│   ├── run_pipeline.py                # v2_no_claude runner → Excel output
│   ├── cache/                         # Intermediate extractor/supervisor JSONs
│   └── output/                        # Final Excel files
│
├── tools/
│   └── export_to_excel.py             # Convert any results dir → Excel
│
├── data/
│   ├── ground_truth/
│   │   └── ground_truth.xlsx          # Manual annotations (33 pages)
│   └── visual_features/
│       ├── clip_embeddings.json       # CLIP visual feature vectors (1,581 pages)
│       └── visual_features.json       # Additional CV features
│
├── src/                               # Core library (shared by pipeline & experiments)
│   ├── agents/
│   │   ├── standalone_extractor.py    # V2 extractor (direct from image)
│   │   ├── supervisor.py              # Row-by-row arbitration (SOTA)
│   │   ├── extractor.py               # V1 extractor (fills skeleton)
│   │   ├── corrector.py               # V1 corrector
│   │   └── structurer.py              # V1 structurer (creates skeleton)
│   ├── prompts/
│   │   ├── standalone_extractor.py    # Extractor system prompts
│   │   ├── supervisor.py              # Supervisor arbitration logic
│   │   ├── extractor.py               # V1 extractor prompts
│   │   ├── corrector.py               # V1 corrector prompts
│   │   └── structurer.py              # V1 structurer prompts
│   ├── evaluation/
│   │   ├── scorer.py                  # Axis1/Axis2 scoring logic
│   │   └── gt_converter.py            # Ground truth parser
│   ├── clients.py                     # Unified LLM client (OpenAI, Google, Anthropic)
│   ├── config.py                      # Model registry & pipeline configurations
│   ├── validation.py                  # Currency rule checks (shillings, pence, fractions)
│   └── __init__.py
│
└── experiments/                       # Research & analysis (not for production)
    ├── run_experiment.py              # Extraction experiment runner (v1–v6, ablations)
    │
    ├── enrichment/                    # Data enrichment pipeline
    │   └── enrich_supervisor_rows.py  # LLM enrichment of extracted rows
    │
    ├── analysis/                      # Downstream analysis & visualisation
    │   ├── analysis_enriched.py       # Comprehensive enriched-data analysis
    │   ├── embedding_analysis_enriched.py  # Embedding-based analysis of enriched data
    │   ├── reanalysis_ledger_yearly.py     # Yearly time-series analysis (v1)
    │   ├── reanalysis_ledger_yearly_v2.py  # Yearly time-series analysis (v2)
    │   ├── hybrid_year_text_embeddings_ledger.py  # Hybrid year+text embeddings
    │   ├── latinbert_year_embeddings.py    # LatinBERT embeddings for year docs
    │   ├── compare_unit_text_clustering_ledger.py  # Unit vs text clustering
    │   └── visualize_reanalysis_ledger.py  # Visualisation helpers
    │
    ├── robustness/                    # Robustness testing framework
    │   ├── sample_pages.py
    │   ├── measurement_validation.py
    │   ├── reliability_metrics.py
    │   ├── robustness_report.py
    │   └── rerun_enrichment.py
    │
    ├── v6_loocv/                      # Phase 6: CLIP-based adaptive routing (LOOCV)
    │   ├── README.md
    │   └── ...
    │
    ├── results/                       # Experiment outputs
    │   ├── v1/                        # Skeleton-based extraction
    │   ├── v2/                        # Multi-extractor + supervisor
    │   ├── v3/, v4/, v5/              # Archived experiments
    │   ├── enriched/                  # LLM-enriched rows (1,581 pages)
    │   ├── robustness/                # Robustness testing outputs
    │   └── sample_pdf/                # Large-scale sample PDF results
    │
    └── reports/                       # Generated analysis reports
        ├── ablation_comparison.csv
        ├── experiment_results_*.csv   # Per-pipeline scoring summaries
        ├── enriched_analysis/         # Output from analysis/analysis_enriched.py
        ├── ledger_clean/              # Output from analysis/reanalysis_ledger_yearly.py
        └── ledger_clean_v2/           # Output from analysis/reanalysis_ledger_yearly_v2.py
```

---

## 📈 Evolution & Results

### **Initial Architecture (v1)**

**Pipeline:** Structurer → Extractor → Corrector

**Approach:**
- Structurer creates skeleton (row types, counts)
- Extractor fills in amounts
- Corrector audits and fixes errors

**Results:**
- Combined: **0.7803**
- **Bottleneck:** Structurer (axis1 avg 0.7554, binding on 22/33 pages)

**Key Learning:** Skeleton-based approach fails when initial structure is wrong.

---

### **Multi-Agent Architecture (v2)**

**Pipeline:** Multiple Extractors → Supervisor

**Approach:**
- 2–3 independent extractors (gemini-flash, gpt-5-mini, claude-haiku)
- Each extracts from raw image (no skeleton)
- Supervisor picks best row-by-row using confidence scores

**Results:**
- v2 (3 extractors): **0.8295**
- v2_no_gemini (gpt + claude): **0.8394**
- v2_no_gpt (gemini + claude): **0.8205**
- **v2_no_claude (gemini + gpt): 0.8385** ← **CURRENT SOTA**

**Key Insight:** Gemini-flash + GPT-5-mini achieves best axis2 performance (0.8515)

**Ablation Study:**
- Removing Gemini → worse performance
- Removing Claude → **best results** (saves 33% cost)
- Removing GPT → worse performance

---

### **Advanced Experiments (Archived)**

#### **v3: Prompt Optimizer (Page-Specific Guidance)**
- Combined: **0.8039** — generic prompts work better

#### **v4: Agentic Debate (Collaborative Refinement)**
- No improvement, 3–4× more expensive

#### **v5: Validator (Final Quality Check)**
- No improvement — supervisor output is already clean

#### **v6: CLIP-Based Adaptive Routing (Active)**
- LOOCV experiments using visual features to route pages to different pipelines
- See `experiments/v6_loocv/README.md`

---

## 🎯 Current SOTA: v2_no_claude

**Architecture:**
```
Image
  │
  ├──────────────────────┐
  ▼                      ▼
┌─────────────┐    ┌─────────────┐
│ Extractor A │    │ Extractor B │
│ gemini-flash│    │ gpt-5-mini  │
└─────────────┘    └─────────────┘
       │                  │
       └──────────┬───────┘
                  ▼
         ┌─────────────────┐
         │ Supervisor      │
         │ (gemini-flash)  │
         │                 │
         │ Row-by-row pick │
         │ based on:       │
         │ - Confidence    │
         │ - Currency rules│
         │ - Consistency   │
         └─────────────────┘
```

**Performance:**
- **Combined:** 0.8385
- **Axis 1 (Structure):** 0.8255
- **Axis 2 (Numbers):** 0.8515

---

## 📊 Evaluation Metrics

### **Two-Axis Scoring**

**Axis 1 (Structural Accuracy):** 40% weight
- Row count match
- Row type counts (entry, header, total)
- Header text fuzzy matching (>80% similarity)

**Axis 2 (Numerical Accuracy):** 60% weight
- Exact £/s/d match: 50%
- Amount similarity (within 5%): 30%
- Fraction match (0.25/0.5/0.75): 20%

**Combined Score:** (axis1 + axis2) / 2

---

## 🔧 Key Technical Challenges Solved

### **1. Currency Column Misalignment**
**Problem:** Models often merge £/s/d columns (e.g., "25" shillings instead of "2" pounds "5" shillings)

**Solution:**
- Explicit vertical ruling line guidance in prompts
- Post-extraction validation (shillings must be 0–19, pence 0–11)
- Supervisor cross-checks candidates for rule violations

### **2. Fraction Ambiguity**
**Problem:** 'ob' (obolus = ½ penny) looks like 'd' in handwriting

**Solution:**
- Prompt guidance: "If you see 'd' or 'ob' after pence, it's 0.5, not a unit label"
- Supervisor has explicit fraction rules in prompt

### **3. Section Headers vs Entries**
**Problem:** Models skip rows with no amounts (section headers)

**Solution:**
- Explicit instruction: "Extract rows with NO amounts as row_type='section_header'"
- Structural scoring penalizes missing headers

### **4. Handwriting Digit Disambiguation (0/6/9)**
**Problem:** Handwritten 0, 6, 9 are visually similar

**Solution:**
- Prompt guidance: check loop direction, pen tail, alignment with neighbouring rows

---

## 🚀 Production Pipeline

The `pipeline/` folder contains the final, clean production pipeline.

### **Running the pipeline**

```bash
# Process all images in a directory:
python pipeline/run_pipeline.py --images data/images/

# Process specific pages:
python pipeline/run_pipeline.py --images data/images/1700_7.png

# Reuse cached intermediate results (skip API calls for already-processed pages):
python pipeline/run_pipeline.py --images data/images/ --use-cache

# Specify output path:
python pipeline/run_pipeline.py --images data/images/ --output results/ledger.xlsx
```

### **Output**

Single `.xlsx` file with:
- **Summary sheet** — page-level row counts (total, entries, headers, totals)
- **One sheet per page** — row #, type, description, £, s, d, fraction

---

## 🌱 Enrichment Pipeline

After extraction, the enrichment pipeline adds structured semantic metadata to every entry row using an LLM.

### **Running enrichment**

```bash
python experiments/enrichment/enrich_supervisor_rows.py
```

Reads all `*_supervisor_gemini-flash.json` files from `experiments/results/cache/` and writes enriched JSONs to `experiments/results/enriched/`.

### **Enrichment fields added**

| Field | Description |
|-------|-------------|
| `direction` | income / expenditure / transfer / balance_sheet / unclear |
| `category` | land_rent / ecclesiastical / maintenance / salary_stipend / administrative / educational / financial / domestic / charitable / other |
| `language` | latin / english / mixed |
| `english_description` | plain-English gloss |
| `english_description_with_amount` | gloss including £/s/d value in prose |
| `place_name` | normalised place or property name |
| `person_name` | normalised person name (tenant / payee) |
| `payment_period` | half_year / annual / sesquiannual / … / one_off / unclear |
| `is_signature` | true / false |
| `is_arrears` | true / false |

**Status:** 1,581 pages processed (1700–1900)

---

## 🔬 Analysis

Analysis scripts live in `experiments/analysis/` and operate on the enriched data or raw extraction results.

| Script | Description | Output |
|--------|-------------|--------|
| `analysis_enriched.py` | Comprehensive enriched-data analysis: direction/category time-series, place/person frequencies, change-point detection | `reports/enriched_analysis/` |
| `embedding_analysis_enriched.py` | Embedding-based analysis: entry clusters, year cosine similarity, category drift | `reports/enriched_analysis/embeddings/` |
| `reanalysis_ledger_yearly.py` | Yearly time-series from raw extraction: amount trends, term frequencies, header diversity | `reports/ledger_clean/` |
| `reanalysis_ledger_yearly_v2.py` | Improved yearly analysis with LatinBERT side-map | `reports/ledger_clean_v2/` |
| `hybrid_year_text_embeddings_ledger.py` | Hybrid year+text embeddings for clustering | `reports/ledger_clean/` |
| `latinbert_year_embeddings.py` | LatinBERT embeddings for year-level documents | `reports/ledger_clean_v2/latinbert/` |
| `compare_unit_text_clustering_ledger.py` | Compares unit-based vs text-based clustering | `reports/ledger_clean/` |
| `visualize_reanalysis_ledger.py` | Additional visualisation helpers | `reports/ledger_clean/` |

All scripts resolve the project root via `Path(__file__).resolve().parents[2]`.

---

## 🌿 Git Workflow

### **Branch Structure**

```
main                                    ← Production-ready code (v2_no_claude)
  │
  └── experiments/supervisor-improvements  ← Ablation studies (v2 variants)
        ├── experiments/v3-optimizer-retry    ← archived
        ├── experiments/v4-debate-fixed       ← archived
        └── experiments/v5-validator          ← archived
```

### **Best Practices**

```bash
# Always branch from main for new experiments
git checkout main
git checkout -b experiments/new-feature-name

# Test on 1 page first
python -m experiments.run_experiment --pipeline v2_no_claude --pages 1700_7

# Merge if improvement ≥0.01, archive if not
git checkout main
git merge experiments/new-feature-name
git tag v2-sota
```

---

## 🔬 Experiment Checklist

Before running a new experiment:

- [ ] Create branch from `main`
- [ ] Add pipeline config to `src/config.py`
- [ ] Add runner function to `experiments/run_experiment.py`
- [ ] Test on 1 page: `--pages 1700_7`
- [ ] Run full experiment on all 33 pages
- [ ] Compare to SOTA using `--compare-ablations`
- [ ] Document results in commit message
- [ ] Merge if improvement ≥0.01, archive if not

---

## 📚 Key Learnings

### **What Works**
✅ Multi-agent ensemble (2 extractors: gemini-flash + gpt-5-mini)
✅ Row-by-row arbitration (vs. wholesale selection)
✅ Explicit currency validation rules in prompts
✅ Confidence-based selection
✅ Simple, generic prompts (no page-specific optimization)

### **What Doesn't Work**
❌ Skeleton-based extraction (brittle)
❌ Page-specific prompt optimization (v3)
❌ Debate and revision loops (v4)
❌ Aggressive post-hoc validation (v5)
❌ Adding Claude as an extractor (hurts performance, saves nothing)

### **Open Questions**
❓ Can we improve axis1 (structural) without hurting axis2?
❓ Is there a better supervisor arbitration strategy than confidence-weighted?
❓ What additional features can be extracted from the enriched data? (next phase)

---

## 🛠 Tools

### **Export Existing Results to Excel**

```bash
# Export sample_pdf results (default):
python tools/export_to_excel.py

# Export from a specific results directory:
python tools/export_to_excel.py --results-dir experiments/results/sample_pdf

# Specify output path:
python tools/export_to_excel.py --output my_results.xlsx
```

Output: Excel file with a **Summary** sheet and one sheet per page, colour-coded row types (blue = headers, green = totals, white = entries).

---

## 🏃 Quick Start

### **1. Production Extraction**

```bash
python pipeline/run_pipeline.py --images data/images/
python pipeline/run_pipeline.py --images data/images/1700_7.png  # single page
python pipeline/run_pipeline.py --images data/images/ --use-cache  # skip API calls
```

### **2. Run Extraction Experiment**

```bash
python -m experiments.run_experiment --pipeline v2_no_claude
python -m experiments.run_experiment --pipeline v2_no_claude --pages 1700_7
python -m experiments.run_experiment --pipeline v2_no_claude --eval-only  # reuse cache
python -m experiments.run_experiment --compare-ablations
```

### **3. Run Enrichment**

```bash
python experiments/enrichment/enrich_supervisor_rows.py
```

### **4. Run Analysis**

```bash
python experiments/analysis/analysis_enriched.py         # enriched data analysis
python experiments/analysis/embedding_analysis_enriched.py  # embedding analysis
python experiments/analysis/reanalysis_ledger_yearly_v2.py  # yearly time-series
```

---

## 🔍 Troubleshooting

### **API Key Errors**
Ensure `.env` exists with correct keys. Check `load_dotenv()` is called in `src/config.py`.

### **Image Not Found**
Verify page identifier matches filename (e.g., `1700_7.png`, not `1700-7.png`).

### **Ground Truth Mismatch**
Check `data/ground_truth/ground_truth.xlsx` contains a sheet for the page.

### **API Rate Limits**
Reduce batch size in `run_experiment.py` or add delays between API calls.

---

## ❓ FAQ

**Q: Why is v2_no_claude the SOTA?**
A: Best combined score (0.8385) while being 33% cheaper. Ablation showed Claude hurts performance on this specific task.

**Q: Can I use custom models?**
A: Yes. Add to `MODELS` in `src/config.py`, then reference in `AGENT_ROLES`.

**Q: What does "axis1" vs "axis2" mean?**
A: Axis1 = structural correctness (row count, types, headers). Axis2 = numerical accuracy (£/s/d). Combined = (axis1 + axis2) / 2.

**Q: How do I add a new analysis script?**
A: Place it in `experiments/analysis/`, use `ROOT = Path(__file__).resolve().parents[2]` for path resolution, and write outputs to `experiments/reports/<your-analysis-name>/`.

---

## 📖 Model Versions Used

| Alias | Model Name | Provider |
|-------|-----------|----------|
| `gemini-flash` | gemini-2.5-flash | Google |
| `gpt-5-mini` | gpt-5-mini | OpenAI |
| `claude-haiku` | claude-haiku-4-5 | Anthropic |

---

## 📞 Contact & Contributors

**Lead Researcher:** Jungwoo Hong — jwhong21@korea.ac.kr — HAI Lab

---

**Last Updated:** 2026-04-11
**Current SOTA:** v2_no_claude (0.8385 combined, 0.8515 axis2)
**Status:** Active development — enrichment complete (1,581 pages), feature analysis in progress
