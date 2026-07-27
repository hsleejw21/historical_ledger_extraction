# V6 LOOCV: Adaptive Routing for Ledger Extraction

**Goal:** Build an adaptive routing system that selects the optimal extraction pipeline for each page based on visual features or semantic embeddings, reducing API costs while preserving accuracy.

## Overview

This directory contains experiments for intelligent pipeline selection using Leave-One-Out Cross-Validation (LOOCV) to predict which extraction pipeline will perform best on each ledger page.

### Three Routing Approaches

1. **Visual Binary** - 28 hand-crafted visual features + binary skip decisions
2. **CLIP Binary** - 512-dim CLIP embeddings + binary skip decisions (apples-to-apples comparison)
3. **CLIP Multi-Class** - 512-dim CLIP embeddings + direct pipeline selection

---

## File Structure

```
experiments/v6_loocv/
├── README.md                          ← You are here
│
├── Step 1: Data Preparation
│   ├── rescore_with_expanded_metrics.py   ← Re-score v2 cached results with expanded metrics
│   ├── aggregate_reports.py               ← Merge v2 experiment CSVs into unified view
│   └── analyze_oracle.py                  ← Identify oracle best v2 pipeline per page
│
├── Step 2: Feature Extraction
│   ├── extract_visual_features.py         ← Extract 28 visual features from images
│   ├── extract_clip_embeddings.py         ← Extract 512-dim CLIP embeddings
│   └── visualize_features.py              ← Generate feature distribution plots
│
├── Step 3: Adaptive Routing Models
│   ├── loocv_prediction.py                ← Visual features + binary skip
│   ├── loocv_clip_binary.py               ← CLIP embeddings + binary skip
│   └── loocv_clip_multiclass.py           ← CLIP embeddings + multi-class selection
│
├── Step 4: Analysis & Comparison
│   └── compare_systems.py                 ← Compare all three routing systems vs SOTA
│
└── outputs/                           ← Generated results (all v2-only)
    ├── unified_results.csv                ← 132 records: 33 pages x 4 v2 pipelines
    ├── oracle_best_per_page.csv           ← Oracle best v2 pipeline per page
    ├── oracle_analysis_summary.txt        ← Oracle analysis text report
    ├── sweep_summary.csv                  ← Visual binary sweep results
    ├── sweep_clip_binary_summary.csv      ← CLIP binary sweep results
    ├── sweep_clip_multiclass_summary.csv  ← CLIP multi-class sweep results
    ├── system_comparison.csv              ← Final comparison table
    └── system_comparison.png              ← Final comparison bar chart

data/visual_features/                  ← Feature data (not in outputs/)
    ├── visual_features.json               ← 28 features x 33 pages
    └── clip_embeddings.json               ← 512-dim embeddings x 33 pages
```

---

## Complete Workflow

### Step 1: Data Preparation (One-Time Setup)

#### 1.1 Re-Score Cached Results
```bash
python -m experiments.v6_loocv.rescore_with_expanded_metrics
```
**What it does:**
- Re-scores all cached **v2** pipeline results with the updated scorer (0.3×axis1 + 0.7×axis2)
- Creates backups before updating
- Exposes detailed axis2 components: match rate, similarity, fraction accuracy

**Outputs:**
- Updated `experiments/reports/experiment_results_v2.csv`
- Backup saved as `experiment_results_v2_backup.csv`

---

#### 1.2 Aggregate All Reports
```bash
python -m experiments.v6_loocv.aggregate_reports
```
**What it does:**
- Merges v2 experiment CSVs into a single unified view
- Tags each row with the v2 pipeline variant (v2_full, v2_no_gemini, v2_no_gpt, v2_no_claude)
- Creates a comprehensive dataset for oracle analysis and LOOCV

**Outputs:**
- `experiments/v6_loocv/outputs/unified_results.csv` - 132 records (33 pages × 4 v2 pipelines)

**Key columns:**
- `page`, `pipeline`, `model_combo`
- `final_axis1`, `final_axis2`, `final_combined`
- `axis2_match`, `axis2_similarity`, `axis2_fraction`

---

#### 1.3 Analyze Oracle Best
```bash
python -m experiments.v6_loocv.analyze_oracle
```
**What it does:**
- Identifies which pipeline performed best on each page (oracle best)
- Analyzes performance patterns and gaps
- Provides target labels for LOOCV prediction

**Outputs:**
- `experiments/v6_loocv/outputs/oracle_best_per_page.csv` - Oracle choice per page
- Console output shows oracle distribution

**Example output (current results):**
```
Oracle pipeline distribution:
  v2_no_gemini : 12 pages (36%)  ← Wins most often!
  v2_no_claude :  8 pages (24%)  ← Current SOTA (fixed policy)
  v2_no_gpt    :  8 pages (24%)
  v2_full      :  5 pages (15%)

Oracle avg:        0.8706
Full ensemble avg: 0.8380
```

---

### Step 2: Feature Extraction

#### 2.1 Extract Visual Features
```bash
python -m experiments.v6_loocv.extract_visual_features
```
**What it does:**
- Extracts 28 hand-crafted visual features from each page image
- Features include: aspect ratio, brightness, contrast, edge density, text blocks, etc.
- Lightweight features designed to capture page complexity

**Outputs:**
- `data/visual_features/visual_features.json` - 28 features × 33 pages

**Time:** ~30 seconds

---

#### 2.2 Extract CLIP Embeddings
```bash
python -m experiments.v6_loocv.extract_clip_embeddings
```
**What it does:**
- Loads OpenAI CLIP model (clip-vit-base-patch32, ~150MB download on first run)
- Extracts 512-dimensional semantic embeddings from each page image
- CLIP embeddings capture semantic content rather than just pixel statistics

**Outputs:**
- `data/visual_features/clip_embeddings.json` - 512-dim embeddings per page

**Time:** ~1-2 minutes (CPU), ~30 seconds (GPU)

**Expected output:**
```
CLIP EMBEDDING EXTRACTION
[Found] 33 images in data/images/
[Loading] CLIP model...
[Extracting embeddings...]
CLIP embedding: 100%|████████| 33/33 [00:45<00:00]
[OK] Extracted embeddings for 33/33 pages

Pairwise cosine similarity stats:
  Mean: 0.9031
  Most similar pair:  1855_7 <-> 1881_1 (sim=0.9720)
  Most dissimilar:    1700_7 <-> 1895_2 (sim=0.7876)
```

---

#### 2.3 Visualize Features (Optional)
```bash
python -m experiments.v6_loocv.visualize_features
```
**What it does:**
- Generates distribution plots for all 28 visual features
- Helps understand feature variance and outliers

**Outputs:**
- `data/visual_features/feature_distributions.png`

---

### Step 3: Train Adaptive Routing Models

#### 3.1 Visual Binary (Hand-Crafted Features)
```bash
# Full hyperparameter sweep
python -m experiments.v6_loocv.loocv_prediction --sweep

# Or single configuration
python -m experiments.v6_loocv.loocv_prediction --k 5 --skip-threshold 0.03 --conf 0.70
```
**What it does:**
- Uses k-NN on 28 visual features to predict which extractors to skip
- Binary decision per extractor: skip Gemini? skip GPT? skip Claude?
- Sweeps over k={3,5,7}, skip_threshold={0.005,0.010,0.020,0.030}, conf={0.60,0.65,0.70}

**Outputs:**
- `outputs/sweep_summary.csv` - Results for all hyperparameter configs (includes axis2 components + `min_score` + `score_std`)
- Best configuration displayed in console

**Time:** ~5 minutes for full sweep

---

#### 3.2 CLIP Binary (Embeddings + Binary Skip)
```bash
# Full hyperparameter sweep
python -m experiments.v6_loocv.loocv_clip_binary --sweep

# Or single configuration
python -m experiments.v6_loocv.loocv_clip_binary --k 5 --skip-threshold 0.03 --conf 0.65
```
**What it does:**
- Same task as visual binary, but uses CLIP embeddings instead of visual features
- Uses cosine similarity (CLIP) instead of Euclidean distance (visual)
- Direct apples-to-apples comparison of feature representations

**Outputs:**
- `outputs/sweep_clip_binary_summary.csv` (includes `min_score`, `score_std`)

**Time:** ~5 minutes for full sweep

---

#### 3.3 CLIP Multi-Class (Direct Pipeline Selection)
```bash
# Full hyperparameter sweep
python -m experiments.v6_loocv.loocv_clip_multiclass --sweep

# Or single configuration
python -m experiments.v6_loocv.loocv_clip_multiclass --k 5 --conf 0.50
```
**What it does:**
- Directly predicts which complete v2 pipeline to run (v2_full, v2_no_gemini, v2_no_gpt, v2_no_claude)
- Can recommend v2_no_gemini, which binary systems couldn't (treated Gemini as backbone)
- Falls back to v2_no_claude (current SOTA) if neighbor agreement is low

**Outputs:**
- `outputs/sweep_clip_multiclass_summary.csv` (includes `min_score`, `score_std`)

**Time:** ~3 minutes for full sweep

---

### Step 4: Compare All Systems

```bash
python -m experiments.v6_loocv.compare_systems
```
**What it does:**
- Loads best configurations from all three routing systems
- Generates comparison table and visualization plot
- Identifies winners across average score, balanced score, worst-case score, and variance

**Outputs:**
- Console table with all metrics
- `outputs/system_comparison.csv` - One row per method (SOTA + 3 routing systems) with all comparison metrics
- `outputs/system_comparison.png` - Visualization generated from `system_comparison.csv` (routing systems only)

#### How to interpret `system_comparison.csv`
- Each row is the **best hyperparameter configuration** for that method (selected by `balanced` from each sweep file)
- `final_avg`: average combined score across all 33 pages
- `min_score`: worst per-page combined score (higher is better for robustness)
- `score_std`: standard deviation across per-page combined scores (lower is more stable)
- `is_highest_min`: `True` if the method ties for the highest minimum score
- `is_lowest_std`: `True` if the method has the lowest variance among methods

#### How to interpret `system_comparison.png`
- Multi-panel bar chart built directly from **all numeric columns** in `system_comparison.csv`
- Includes all methods with non-null values for each metric (SOTA appears on metrics where it has values)
- Default panel order starts with: `accuracy`, `pres_ensemble`, `cost_reduction`, `balanced`, `final_avg`, axis metrics, `min_score`, `score_std`
- Dashed red line appears only on metrics with explicit screening targets (`accuracy`, `pres_ensemble`, `cost_reduction`)

**Example output (current results, scorer = 0.3*axis1 + 0.7*axis2):**
```
ADAPTIVE ROUTING - SYSTEM COMPARISON (vs SOTA: v2_no_claude)
===============================================================================================
  Method                      AvgScore   vs SOTA  Pres/Ens  Cost_r  Accuracy  Balanced
-----------------------------------------------------------------------------------------------
  [SOTA] v2_no_claude        0.8437     baseline   100.7%   33.3%       n/a       n/a
  visual_binary                 0.8396   -0.0041   100.2%    2.0%     81.8%    0.6523
  clip_binary                   0.8417   -0.0020   100.4%   22.2%     70.7%    0.6806 <-
  clip_multiclass               0.8435   -0.0002   100.7%   27.3%     39.4%    0.6026

AXIS2 SCORE BREAKDOWN (vs SOTA: v2_no_claude)
===============================================================================================
  Method                     CombScore   Axis1   Axis2   Match   Simil    Frac
---------------------------------------------------------------------------
  [SOTA] sota_v2_no_claude      0.8437  0.8255  0.8515  0.8279  0.9670  0.7372
  visual_binary                 0.8396  0.8113  0.8518  0.8251  0.9651  0.7486
  clip_binary                   0.8417  0.8196  0.8512  0.8282  0.9679  0.7334
  clip_multiclass               0.8435  0.8275  0.8504  0.8276  0.9697  0.7284
  Axis2 = 0.5*Match + 0.3*Similarity + 0.2*Fraction

SCORE ROBUSTNESS (per-page worst-case and spread)
===========================================================================
  Objective: highest min_score = best worst-case guarantee
             lowest score_std = most stable method

  Method                      MinScore    StdDev  AvgScore
------------------------------------------------------------
  [SOTA] sota_v2_no_claude      0.5861    0.1054    0.8437 <-- highest min
  visual_binary                 0.5861    0.1018    0.8396 <-- highest min
  clip_binary                   0.5861    0.0964    0.8417 <-- highest min, lowest std
  clip_multiclass               0.5666    0.1049    0.8435
```

---

## Key Metrics

### 1. Prediction Accuracy
**Definition:** % of pages where predicted pipeline matched oracle best

**Interpretation:** How often did the system choose the optimal pipeline?

### 2. Performance Preservation (vs Ensemble)
**Definition:** avg(predicted_pipeline_score) / avg(full_ensemble_score)

**Interpretation:** How much performance is retained compared to running all models?
- Target: ≥95%
- 100%+ means predictions sometimes beat ensemble

### 3. Cost Reduction
**Definition:** % reduction in API calls compared to full ensemble

**Interpretation:** How much money is saved by skipping models?
- Full ensemble = 3 API calls per page
- Binary skip systems can achieve 20-35% reduction
- Higher is better

### 4. Balanced Score
**Definition:** `0.3 × accuracy + 0.4 × pres_ensemble + 0.3 × cost_reduction`

**Interpretation:** Overall system quality balancing all three metrics
- Combines accuracy, preservation, and cost into single metric
- Higher is better

### 5. Worst-Case Score (`min_score`)
**Definition:** Minimum per-page combined score achieved by a method

**Interpretation:** Conservative robustness objective; if you want the highest guaranteed floor, maximize this metric.

### 6. Variance (`score_std`)
**Definition:** Standard deviation of per-page combined scores

**Interpretation:** Stability objective; lower values indicate less performance fluctuation across pages.

---

## Dependencies

```bash
pip install transformers torch pillow opencv-python pandas numpy matplotlib scipy scikit-learn tqdm
```

**For CLIP models (first run only):**
- Downloads openai/clip-vit-base-patch32 (~150MB) from HuggingFace
- Cached locally after first download

---

## Quick Reference: Common Commands

```bash
# Complete workflow from scratch
python -m experiments.v6_loocv.rescore_with_expanded_metrics
python -m experiments.v6_loocv.aggregate_reports
python -m experiments.v6_loocv.analyze_oracle
python -m experiments.v6_loocv.extract_visual_features
python -m experiments.v6_loocv.extract_clip_embeddings
python -m experiments.v6_loocv.loocv_prediction --sweep
python -m experiments.v6_loocv.loocv_clip_binary --sweep
python -m experiments.v6_loocv.loocv_clip_multiclass --sweep
python -m experiments.v6_loocv.compare_systems

# Re-run just the comparison after tweaking configs
python -m experiments.v6_loocv.compare_systems

# Test a specific configuration
python -m experiments.v6_loocv.loocv_clip_binary --k 5 --skip-threshold 0.03 --conf 0.65
```

---

## Results Summary

Based on experiments with 33 historical ledger pages using scorer: `combined = 0.3×axis1 + 0.7×axis2`

**SOTA Baseline: v2_no_claude (always)**
- Fixed policy — always runs Gemini + GPT (skips Claude)
- Combined score: **0.8437**  |  Axis1: 0.8255  |  Axis2: 0.8515
- Axis2 components: Match=0.8279, Similarity=0.9670, Fraction=0.7372
- Cost reduction: 33.3% (vs full 3-model ensemble)

---

**Best Routing System: CLIP Binary** (best balanced score)
- Best config: skip_threshold=0.03, conf=0.6, k=5
- Combined score: **0.8417** (-0.0020 vs SOTA)
- Axis1: 0.8196  |  Axis2: 0.8512  |  Match=0.8282, Sim=0.9679, Frac=0.7334
- Binary accuracy: 70.7%  |  Cost reduction: **22.2%**  |  Balanced: **0.6806**

**Closest to SOTA Score: CLIP Multi-Class**
- Best config: conf=0.3, k=3
- Combined score: **0.8435** (-0.0002 vs SOTA, nearly identical!)
- Axis1: 0.8275  |  Axis2: 0.8504  |  Match=0.8276, Sim=0.9697, Frac=0.7284
- Pipeline accuracy: 39.4%  |  Cost reduction: 27.3%
- Can recommend v2_no_gemini (oracle best on 36% of pages), which binary systems can't

**Highest Prediction Accuracy: Visual Binary**
- Best config: skip_threshold=0.03, conf=0.65, k=7
- Combined score: **0.8396** (-0.0041 vs SOTA)
- Axis1: 0.8113  |  Axis2: 0.8518  |  Match=0.8251, Sim=0.9651, Frac=0.7486
- Binary accuracy: **81.8%**  |  Cost reduction: only 2.0%
- High accuracy but conservative — tends to fall back to full ensemble

### Key Findings

1. **All three systems preserve performance** — within 0.5% of SOTA combined score
2. **Axis2 scores are nearly identical** across all systems (~0.851), differences lie in Axis1
3. **CLIP binary offers the best cost-accuracy tradeoff** (22% savings, 0.6806 balanced score)
4. **CLIP multi-class matches SOTA score most closely** (-0.0002) with 27% cost reduction
5. **Visual binary is over-conservative** — 81.8% skip accuracy but only 2% cost reduction; the k-NN predicts "skip" correctly but confidence threshold prevents most actual skips
6. **Oracle upper bound is 0.8706** — routing can theoretically improve on SOTA by 2.7%

---

## Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers torch
```

### "unified_results.csv not found"
Run data preparation steps first:
```bash
python -m experiments.v6_loocv.aggregate_reports
```

### "oracle_best_per_page.csv not found"
```bash
python -m experiments.v6_loocv.analyze_oracle
```

### "CLIP model download fails"
- Check internet connection
- Model downloads from HuggingFace (~150MB)
- Cached in `~/.cache/huggingface/` after first download

### "Low accuracy in LOOCV"
- Expected with 33 samples (small dataset)
- Visual binary achieves 81.8% skip accuracy — but conservative thresholds limit cost savings
- CLIP binary achieves 70.7% with 22% cost reduction (better overall tradeoff)
- Multi-class struggles (39.4%) due to fine-grained 4-class prediction with limited data

---

## Future Work

1. **Hybrid features** - Combine visual + CLIP embeddings
2. **Confidence-weighted fallback** - Use ensemble when prediction confidence is low
3. **Page clustering** - Group similar pages to improve k-NN with limited data
4. **Production integration** - Deploy best system (`clip_binary`) to live pipeline
5. **Expand training set** - Collect more labeled pages to improve accuracy
