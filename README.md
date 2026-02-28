# Historical Ledger Extraction Pipeline

Automated extraction of 18th-19th century English parish ledgers using multi-agent LLM architecture.

**Table of Contents:**
- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Evolution & Results](#-evolution--results)
- [Current SOTA](#-current-sota-v2_no_claude)
- [Evaluation Metrics](#-evaluation-metrics)
- [Technical Challenges](#-key-technical-challenges-solved)
- [Production Pipeline](#-production-pipeline)
- [Tools](#-tools)
- [Quick Start](#-quick-start)
- [Git Workflow](#-git-workflow)

---

## 📊 Project Overview

**Goal:** Extract structured data (£/s/d amounts, descriptions, row types) from scanned historical accounting ledgers.

**Dataset:** 33 pages from 1700-1900, containing:
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
- API keys for: OpenAI, Google AI, Anthropic

### **Installation**

```bash
# Clone repository
git clone <repository-url>
cd historical_ledger_extraction

# Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Configuration**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
```

The config is automatically loaded by `src/config.py`.

---

## 🏗️ Project Structure

```
historical_ledger_extraction/
├── pipeline/                          # ← PRODUCTION PIPELINE (start here)
│   ├── run_pipeline.py                # Clean v2_no_claude runner → Excel output
│   ├── cache/                         # Intermediate extractor/supervisor JSONs
│   └── output/                        # Final Excel files
├── tools/
│   └── export_to_excel.py             # Convert existing results dir → Excel
├── data/
│   ├── images/              # Ledger page scans (.png)
│   ├── ground_truth/        # Manual annotations (.json) — ground truth labels
│   └── results/             # Deprecated (old structure)
├── src/
│   ├── agents/
│   │   ├── standalone_extractor.py   # Individual extractor agent
│   │   ├── supervisor.py             # Row-by-row arbitration (SOTA)
│   │   ├── agentic_supervisor.py     # Debate-based orchestration (v4)
│   │   ├── validator.py              # Final QA check (v5)
│   │   └── prompt_optimizer.py       # Page-specific guidance (v3)
│   ├── prompts/
│   │   ├── standalone_extractor.py   # Extractor system prompts
│   │   ├── supervisor.py             # Supervisor arbitration logic
│   │   ├── agentic_supervisor.py     # Debate prompts
│   │   ├── validator.py              # Validation rules
│   │   └── prompt_optimizer.py       # Optimizer instructions
│   ├── evaluation/
│   │   ├── scorer.py                 # Axis1/Axis2 scoring logic
│   │   └── gt_converter.py           # Ground truth parser
│   ├── clients.py           # Unified LLM client (OpenAI, Google, Anthropic)
│   ├── config.py            # Model registry & pipeline configurations
│   ├── validation.py        # Currency rule checks (shillings, pence, fractions)
│   └── __init__.py
├── experiments/             # Research & ablation experiments (not for production)
│   ├── run_experiment.py    # Experiment runner (v1–v5, ablations)
│   ├── results/             # Experiment outputs by pipeline version
│   │   ├── v1/              # Skeleton-based extraction
│   │   ├── v2/              # Multi-extractor + supervisor
│   │   ├── sample_pdf/      # Large-scale sample PDF results
│   │   └── ...
│   └── reports/
│       └── experiment_results_*.csv  # Summary results (ablations, comparisons)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .env.example             # Template for environment variables
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
- 3 independent extractors (gemini-flash, gpt-5-mini, claude-haiku)
- Each extracts from raw image (no skeleton)
- Supervisor picks best row-by-row using confidence scores

**Results:**
- v2 (3 extractors): **0.8295**
- v2_no_gemini (gpt + claude): **0.8394**
- v2_no_gpt (gemini + claude): **0.8205**
- **v2_no_claude (gemini + gpt): 0.8385** ← **CURRENT SOTA**

**Key Insight:** Gemini-flash + GPT-5-mini has best axis2 performance (0.8515)

**Ablation Study:**
- Removing Gemini → worse performance
- Removing Claude → **best results** (saves 33% cost)
- Removing GPT → worse performance

---

### **Advanced Experiments**

#### **v3: Prompt Optimizer (Page-Specific Guidance)**

**Pipeline:** Optimizer → Extractors → Supervisor

**Hypothesis:** Page-specific guidance (row count, layout hints, warnings) improves extraction.

**Results:**
- Combined: **0.8039**
- Axis 2: **0.8281**
- **Conclusion:** Optimizer **did not improve** results. Generic prompts work better.

---

#### **v4: Agentic Debate (Collaborative Refinement)**

**Pipeline:** Extractors → Agentic Supervisor (debate + retry)

**Hypothesis:** When extractors disagree, debate and revision improves accuracy.

**Approach:**
- Supervisor detects disagreements (£/s/d differ by >12 pence)
- Extractors see each other's reasoning and revise
- Supervisor arbitrates after debate

**Results:**
- **No improvement** over v2_no_claude
- **Conclusion:** Debate adds cost (3-4x) without accuracy gain.

---

#### **v5: Validator (Final Quality Check)**

**Pipeline:** Extractors → Supervisor → Validator

**Hypothesis:** Post-hoc validation fixes currency violations and missed rows.

**Approach:**
- Validator enforces rules (shillings 0-19, pence 0-11)
- Checks row count against image
- Verifies column alignment

**Results:**
- **No improvement** over v2_no_claude
- **Conclusion:** Supervisor output is already clean; aggressive validation doesn't help.

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
- Post-extraction validation (shillings must be 0-19, pence 0-11)
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

---

## 🌿 Git Workflow
=
### **Branch Structure**

```
main                                    ← Production-ready code (v2_no_claude)
  │
  └── experiments/supervisor-improvements  ← Ablation studies (v2 variants)
        │
        ├── experiments/v3-optimizer-retry    ← v3 experiments (archived)
        ├── experiments/v4-debate-fixed       ← v4 experiments (archived)
        └── experiments/v5-validator          ← v5 experiments (archived)
```

### **Best Practices**

**For new experiments:**
```bash
# Always branch from supervisor-improvements (clean base with SOTA)
git checkout experiments/supervisor-improvements
git checkout -b experiments/new-feature-name

# Work on feature
# Test results
# Commit

# If successful → merge to main
git checkout main
git merge experiments/new-feature-name

# If unsuccessful → archive
git checkout experiments/supervisor-improvements
# Delete branch locally (keep commits in history)
git branch -D experiments/new-feature-name
```

**Merging SOTA to main:**
```bash
# When ready to promote v2_no_claude as official SOTA
git checkout main
git merge experiments/supervisor-improvements -m "Promote v2_no_claude as SOTA"
git tag v2-sota
git push origin main --tags
```

---

## 🔬 Experiment Checklist

Before running a new experiment:

- [ ] Create new branch from `experiments/supervisor-improvements`
- [ ] Update `src/config.py` with new pipeline config
- [ ] Add runner function to `experiments/run_experiment.py`
- [ ] Test on 1 page: `--pages 1700_7`
- [ ] Run full experiment
- [ ] Compare to SOTA using comparison script
- [ ] Document results in commit message
- [ ] Merge if improvement ≥0.01, archive if not

---

## 📚 Key Learnings

### **What Works**
✅ Multi-agent ensemble (2-3 extractors)  
✅ Row-by-row arbitration (vs. wholesale selection)  
✅ Explicit currency validation rules in prompts  
✅ Confidence-based selection  
✅ Simple, generic prompts (no page-specific optimization)

### **What Doesn't Work**
❌ Skeleton-based extraction (brittle)  
❌ Page-specific prompt optimization (v3)  
❌ Debate and revision loops (v4)  
❌ Aggressive post-hoc validation (v5)  
❌ More than 3 extractors (diminishing returns)

### **Open Questions**
❓ Can we improve axis1 (structural) without hurting axis2?  
❓ Is there a better supervisor arbitration strategy than confidence-weighted?  
❓ Would a hybrid approach (v2 for simple pages, v4 for complex) help?

---

## 🚀 Production Pipeline

The `pipeline/` folder contains the final, clean production pipeline — completely separate from the research experiments.

### **Architecture**

```
Image(s)
  │
  ├─────────────────────────┐
  ▼                         ▼
┌──────────────┐   ┌──────────────┐
│ gemini-flash │   │ gpt-5-mini   │
│  Extractor   │   │  Extractor   │
└──────────────┘   └──────────────┘
        │                  │
        └────────┬──────────┘
                 ▼
        ┌─────────────────┐
        │ gemini-flash    │
        │   Supervisor    │
        │ (row-by-row)    │
        └─────────────────┘
                 │
                 ▼
        📊 Excel Output
        (one sheet per page)
```

### **Running the pipeline**

```bash
# Process all images in a directory:
python pipeline/run_pipeline.py --images data/images/

# Process specific pages:
python pipeline/run_pipeline.py --images data/images/1700_7.png data/images/1700_8.png

# Reuse cached intermediate results (skip API calls for already-processed pages):
python pipeline/run_pipeline.py --images data/images/ --use-cache

# Specify output path:
python pipeline/run_pipeline.py --images data/images/ --output results/ledger.xlsx
```

### **Output**

The pipeline produces a single `.xlsx` file:
- **Summary sheet** — page-level row counts (total, entries, headers, totals)
- **One sheet per page** — clean extraction: row #, type, description, £, s, d, fraction

Intermediate extractor and supervisor JSONs are saved to `pipeline/cache/` for reuse.

---

## 🛠 Tools

### **Export Existing Results to Excel**

Convert any folder of supervisor JSON results (e.g., `experiments/results/sample_pdf/`) into a single Excel file:

```bash
# Export sample_pdf results (default):
python tools/export_to_excel.py

# Export from a specific results directory:
python tools/export_to_excel.py --results-dir experiments/results/sample_pdf

# Specify output path:
python tools/export_to_excel.py --output my_results.xlsx
```

Output: one Excel file with a **Summary** sheet and one sheet per page, with colour-coded row types (blue = headers, green = totals, white = entries).

---

## 🏃 Quick Start

### **1. Production Extraction (New — Recommended)**

```bash
# Process all images → outputs a single Excel file
python pipeline/run_pipeline.py --images data/images/

# Single page test
python pipeline/run_pipeline.py --images data/images/1700_7.png

# Reuse cache (no API cost for already-processed pages)
python pipeline/run_pipeline.py --images data/images/ --use-cache
```

### **2. Export Existing Results to Excel**

```bash
# Convert the sample_pdf results folder to Excel
python tools/export_to_excel.py

# From a custom results directory
python tools/export_to_excel.py --results-dir experiments/results/sample_pdf
```

### **3. Run Experiment Pipeline (Research)**

```bash
python -m experiments.run_experiment --pipeline v2_no_claude
```

Runs extractors on all 33 research pages with full scoring against ground truth.

### **4. Test Single Page (Research)**

```bash
python -m experiments.run_experiment --pipeline v2_no_claude --pages 1700_7
```

### **5. Evaluation Only (Reuse Cached Results)**

```bash
python -m experiments.run_experiment --pipeline v2_no_claude --eval-only
```

Re-scores previous results without re-running extractors (fast, cost-free).

### **6. Compare Pipelines (Ablation Study)**

```bash
python -m experiments.run_experiment --compare-ablations
```

Compares v2, v2_no_claude, v2_no_gemini, v2_no_gpt side-by-side.

### **7. Check Available Pipelines**

See [src/config.py](src/config.py) for all available pipelines and models.

---

## � Troubleshooting

### **API Key Errors**

**Problem:** `KeyError: 'OPENAI_API_KEY'`

**Solution:** Ensure `.env` file exists with correct keys. Check that `load_dotenv()` is called in `src/config.py`.

### **Image Not Found**

**Problem:** `FileNotFoundError: data/images/<page>.png`

**Solution:** Verify page identifier matches filename in `data/images/`. Example: `1700_7.png` (not `1700-7.png`).

### **Ground Truth Mismatch**

**Problem:** Scorer reports missing ground truth for a page.

**Solution:** Check `data/ground_truth/<page>.json` exists and is valid JSON.

### **API Rate Limits**

**Problem:** `RateLimitError` from OpenAI/Google/Anthropic

**Solution:** 
- Reduce concurrent requests (modify batch size in `run_experiment.py`)
- Add delays between API calls
- Consider reducing number of pages: `--pages <subset>`

### **Out of Memory**

**Problem:** `MemoryError` when loading images

**Solution:** Process fewer pages at once, or reduce image resolution preprocessing.

---

## ❓ FAQ

**Q: Why is v2_no_claude the SOTA?**  
A: It achieves 0.8385 combined score (best axis2: 0.8515) while being 33% cheaper than v2 (which includes Claude). Ablation showed Claude actually hurts performance on this task.

**Q: Can I use custom models?**  
A: Yes. Add them to `MODELS` in `src/config.py`, then reference in `AGENT_ROLES`.

**Q: What does "axis1" vs "axis2" mean?**  
A: **Axis1** = structural correctness (row count, types, headers). **Axis2** = numerical accuracy (£/s/d amounts). Combined = (axis1 + axis2) / 2.

**Q: Can I run experiments in parallel?**  
A: Currently sequential per-page. Multi-threading support can be added (see `run_experiment.py` TODOs).

**Q: How do I add a new experiment version?**  
A: 1. Create agent in `src/agents/`, 2. Create prompts in `src/prompts/`, 3. Add pipeline config in `src/config.py`, 4. Create runner in `experiments/run_experiment.py`.

---

## �📖 References & Resources

### **Model Documentation**
- [Google Gemini API](https://ai.google.dev/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Anthropic Claude API](https://docs.anthropic.com/)

### **Model Versions Used**
| Alias | Model Name | Provider |
|-------|-----------|----------|
| `gemini-flash` | gemini-2.5-flash | Google |
| `gpt-5-mini` | gpt-5-mini | OpenAI |
| `claude-haiku` | claude-3-5-haiku | Anthropic |

### **Historical Context**
- **Project Origin:** HAI Lab
- **Domain:** 18th-19th century English parish accounting
- **Currency System:** £ (pounds), s (shillings, 0-19), d (pence, 0-11), ob (½ penny)

---

## 📞 Contact & Contributors

**Lead Researcher:**  
Jungwoo Hong  
Email: jwhong21@korea.ac.kr  
Affiliation: HAI Lab

---

**Repository Info:**  
**Last Updated:** February 28, 2026
**Current SOTA:** v2_no_claude (0.8385 combined, 0.8515 axis2)
**Status:** Active development — production pipeline added (`pipeline/run_pipeline.py`)