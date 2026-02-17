# Phase 4 CLIP Pipeline - Execution Checklist

## Quick Start (Run These Commands In Order)

### ✅ Prerequisites Check
```bash
# 1. Check if you have the required data files
python -c "import os; print('✓ unified_results.csv' if os.path.exists('experiments/v6_loocv/outputs/unified_results.csv') else '✗ Need to run aggregate_reports'); print('✓ oracle_best_per_page.csv' if os.path.exists('experiments/v6_loocv/outputs/oracle_best_per_page.csv') else '✗ Need to run analyze_oracle')"

# 2. Install dependencies (one-time, already done based on terminal history)
# pip install transformers torch
```

---

### 🔧 Step 1: Extract CLIP Embeddings (~1-2 minutes)
```bash
python -m experiments.v6_loocv.extract_clip_embeddings
```

**Expected output:**
- Downloads CLIP model (~150MB) on first run (cached after that)
- Processes 33 images
- Creates `data/visual_features/clip_embeddings.json`
- Shows similarity statistics

**Success indicators:**
- ✅ "Extracted embeddings for 33/33 pages"
- ✅ "Embedding dimension: 512"
- ✅ File saved to clip_embeddings.json

---

### 🤖 Step 2A: CLIP Binary Skip (~5 minutes)
```bash
python -m experiments.v6_loocv.loocv_clip_binary --sweep
```

**What it does:**
- Tests 36 hyperparameter configurations
- Binary skip task (same as Phase 3, but with CLIP)
- Saves results to `outputs/sweep_clip_binary_summary.csv`

**Success indicators:**
- ✅ "CLIP BINARY SWEEP (36 configs)"
- ✅ Shows best configuration with balanced score
- ✅ CSV file created

---

### 🎯 Step 2B: CLIP Multi-Class Pipeline Selection (~3 minutes)
```bash
python -m experiments.v6_loocv.loocv_clip_multiclass --sweep
```

**What it does:**
- Tests 20 hyperparameter configurations
- Directly predicts which v2 pipeline to run
- CAN recommend v2_no_gemini (binary systems can't!)
- Saves results to `outputs/sweep_clip_multiclass_summary.csv`

**Success indicators:**
- ✅ "CLIP MULTI-CLASS SWEEP (20 configs)"
- ✅ Shows oracle distribution (v2_no_gemini should be 12/33 pages)
- ✅ CSV file created

---

### 📊 Step 3: Compare All Systems (<10 seconds)
```bash
python -m experiments.v6_loocv.compare_phase4
```

**What it does:**
- Loads best configs from Phase 3, 4A, 4B
- Generates comparison table
- Creates visualization plot
- Saves final summary

**Success indicators:**
- ✅ Table shows 3 systems (visual_binary, clip_binary, clip_multiclass)
- ✅ Winner analysis displayed
- ✅ `phase4_comparison.png` created
- ✅ `phase4_final_comparison.csv` created

---

## File Structure After Completion

```
experiments/v6_loocv/
├── outputs/
│   ├── unified_results.csv                   ← From Phase 1-3
│   ├── oracle_best_per_page.csv              ← From Phase 1-3
│   ├── sweep_summary.csv                     ← Phase 3 baseline
│   │
│   ├── sweep_clip_binary_summary.csv         ← NEW: Phase 4A results
│   ├── sweep_clip_multiclass_summary.csv     ← NEW: Phase 4B results
│   ├── phase4_final_comparison.csv           ← NEW: Final comparison
│   └── phase4_comparison.png                 ← NEW: Visualization
│
data/visual_features/
├── visual_features.json                      ← Phase 3 features
└── clip_embeddings.json                      ← NEW: CLIP embeddings
```

---

## Troubleshooting

### If Step 1 fails with "No PNG images found"
```bash
# Check image directory
ls data/images/*.png
# Should show 33 images like: 1700_7_image.png, 1704_1_image.png, etc.
```

### If Step 2A/2B fails with "CLIP embeddings not found"
```bash
# Re-run Step 1
python -m experiments.v6_loocv.extract_clip_embeddings
```

### If Step 2A/2B fails with "Oracle file not found"
```bash
# Run the prerequisite Phase 1-3 steps
python -m experiments.v6_loocv.aggregate_reports
python -m experiments.v6_loocv.analyze_oracle
```

### If Step 3 fails with "Phase 3 sweep not found"
```bash
# Run Phase 3 sweep first
python -m experiments.v6_loocv.loocv_prediction --sweep
# OR
python -m experiments.v6_loocv.sweep_thresholds
```

---

## Expected Timeline

| Step | Command | Time | Output |
|------|---------|------|--------|
| 1 | extract_clip_embeddings | 1-2 min | clip_embeddings.json |
| 2A | loocv_clip_binary --sweep | ~5 min | sweep_clip_binary_summary.csv |
| 2B | loocv_clip_multiclass --sweep | ~3 min | sweep_clip_multiclass_summary.csv |
| 3 | compare_phase4 | <10 sec | phase4_comparison.png + CSV |
| **Total** | | **~10 min** | 4 new files |

---

## Quick Test (Without Sweep)

If you want to test a single configuration before running full sweeps:

```bash
# Test binary skip with default params
python -m experiments.v6_loocv.loocv_clip_binary --k 5 --skip-threshold 0.01 --conf 0.65

# Test multi-class with default params
python -m experiments.v6_loocv.loocv_clip_multiclass --k 5 --conf 0.50
```

---

## What to Look For in Results

### Phase 4A (CLIP Binary) vs Phase 3 (Visual Features)
- **Accuracy:** Should CLIP be better at predicting which extractors to skip?
- **Preservation:** Does CLIP maintain performance better?
- **Cost:** Does CLIP enable more aggressive skipping?

### Phase 4B (Multi-Class) vs 4A (Binary)
- **Accuracy:** Can multi-class predict the right pipeline more often?
- **v2_no_gemini:** Multi-class should recommend this ~36% of the time (oracle best on 12/33 pages)
- **Performance:** Should be closer to oracle since it can pick any v2 config

### Overall Winner
- Look at **balanced score** (0.3×accuracy + 0.4×preservation + 0.3×cost_reduction)
- Check if multi-class dominates both binary systems
- Analyze failure cases to understand trade-offs

---

## Next Steps After Phase 4

1. **Review Results**
   - Check `phase4_comparison.png` for visual comparison
   - Read `phase4_final_comparison.csv` for detailed metrics
   - Identify which system wins on each metric

2. **Deploy Best System**
   - Use best config from winning system
   - Integrate into production pipeline
   - Monitor performance on new pages

3. **Optional: Deep Dive**
   - Check per-page predictions in detailed CSVs
   - Analyze pages where CLIP helps vs visual features
   - Look for patterns in failures

4. **Future Work**
   - Fine-tune CLIP on historical ledger domain
   - Combine CLIP + visual features (ensemble)
   - Test other vision models (DINOv2, ViT, etc.)

---

## Commands Summary (Copy-Paste)

```bash
# Full pipeline (run in order)
python -m experiments.v6_loocv.extract_clip_embeddings
python -m experiments.v6_loocv.loocv_clip_binary --sweep
python -m experiments.v6_loocv.loocv_clip_multiclass --sweep
python -m experiments.v6_loocv.compare_phase4
```

---

**Total execution time: ~10 minutes**
**Total new files created: 4**
**Total file size: ~1-2 MB**
