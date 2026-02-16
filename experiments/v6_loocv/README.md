# V6: Adaptive Routing Experiments

**Goal:** Build a prediction system that selects the optimal extraction model/ensemble for each page based on visual features, reducing API costs while preserving accuracy.

## Phase 1: Deep Dive into Existing Results ✅

Expand scoring metrics and analyze historical performance to identify oracle best per page.

### Scripts:

#### 1. `rescore_with_expanded_metrics.py`
Re-scores all cached results (v1, v2, v3) with expanded axis2 components.

**Outputs:**
- Updated `experiments/reports/experiment_results_v1.csv`
- Updated `experiments/reports/experiment_results_v2.csv`
- Updated `experiments/reports/experiment_results_v3.csv`
- Backups saved as `*_backup.csv`

**New metrics exposed:**
- `axis2_match` - Exact £/s/d match rate (with 0.5 credit for cross-type)
- `axis2_similarity` - Partial credit based on pence distance
- `axis2_fraction` - Exact fraction match (0.25/0.5/0.75)

**Usage:**
```bash
# Re-score all pipelines
python -m experiments.v6_loocv.rescore_with_expanded_metrics

# Re-score specific pipeline
python -m experiments.v6_loocv.rescore_with_expanded_metrics --pipeline v2
```

---

#### 2. `aggregate_reports.py`
Merges all experiment reports into a single unified CSV.

**Outputs:**
- `experiments/reports/unified_results.csv`

**Columns:**
- `page` - Page identifier (e.g., "1700_7")
- `pipeline` - Pipeline version (v1, v2_full, v2_no_claude, v3_optimizer, etc.)
- `model_combo` - Model configuration string
- `final_axis1`, `final_axis2`, `final_combined` - Final scores
- `axis2_match`, `axis2_similarity`, `axis2_fraction` - Detailed axis2 breakdown
- Plus all original columns from each pipeline

**Usage:**
```bash
python -m experiments.v6_loocv.aggregate_reports
```

---

#### 3. `analyze_oracle.py`
Identifies the "oracle best" model/ensemble per page and analyzes performance patterns.

**Outputs:**
- `experiments/reports/oracle_best_per_page.csv` - Best config per page with detailed scores
- `experiments/reports/oracle_analysis_summary.txt` - Readable summary with insights

**Key Questions Answered:**
- For each page, which model performed best?
- How often does ensemble beat best single model?
- What is the performance gap?
- Which axis2 components vary most across models?

**Usage:**
```bash
python -m experiments.v6_loocv.analyze_oracle
```

---

## Phase 1 Workflow

Run these in order:

```bash
# Step 1: Re-score all cached results with expanded metrics
python -m experiments.v6_loocv.rescore_with_expanded_metrics

# Step 2: Merge all reports into unified view
python -m experiments.v6_loocv.aggregate_reports

# Step 3: Analyze oracle best per page
python -m experiments.v6_loocv.analyze_oracle
```

**Outputs** (all in `experiments/v6_loocv/outputs/`):
- `unified_results.csv` - All experiments merged
- `oracle_best_per_page.csv` - Oracle analysis
- `oracle_analysis_summary.txt` - Human-readable insights

---

## Phase 2: Visual Feature Extraction (Next)

TODO: Extract lightweight visual features from each image:
- Image dimensions, aspect ratio
- Brightness histogram
- Edge density (Canny)
- Text block count

---

## Phase 3: LOOCV Prediction (After Phase 2)

TODO: Leave-one-out cross-validation with:
- Simple feature baseline (k-NN)
- CLIP embedding approach
- Performance preservation metrics

---

## Phase 4: Deployment (Final)

TODO: Integrate adaptive router into production pipeline.

---

## Success Metrics

**Prediction Accuracy:**
- % of pages where predicted model matched oracle best

**Performance Preservation:**
- Avg(predicted_model_score) / Avg(oracle_best_score) ≥ 0.95

**Cost Reduction:**
- API calls saved vs. always using ensemble

**Fallback Rate:**
- % of pages needing ensemble due to low prediction confidence