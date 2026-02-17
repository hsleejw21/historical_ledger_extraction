# Phase 4: CLIP-Based Adaptive Routing

This phase extends the v6_loocv adaptive routing system with CLIP (Contrastive Language-Image Pre-training) embeddings to improve page similarity detection and routing decisions.

## Overview

**Problem with Phase 3 (Visual Features):**
- Used 26 hand-crafted visual features (aspect ratio, brightness, contrast, etc.)
- Struggled with semantically similar pages that look visually different
- Binary skip model couldn't recommend v2_no_gemini (which is oracle best on 36% of pages)

**Phase 4 Solutions:**
- **Phase 4A**: CLIP embeddings + binary skip (apples-to-apples comparison with Phase 3)
- **Phase 4B**: CLIP embeddings + multi-class pipeline selection (can recommend any v2 config)

## File Structure

```
experiments/v6_loocv/
├── PHASE4_README.md                   ← You are here
│
├── extract_clip_embeddings.py         ← Step 1: Extract CLIP embeddings
├── loocv_clip_binary.py               ← Step 2A: Binary skip with CLIP
├── loocv_clip_multiclass.py           ← Step 2B: Multi-class with CLIP
├── compare_phase4.py                  ← Step 3: Compare all systems
│
└── outputs/
    ├── clip_embeddings.json           ← CLIP embeddings (512-dim per page)
    ├── sweep_clip_binary_summary.csv  ← Phase 4A sweep results
    ├── sweep_clip_multiclass_summary.csv ← Phase 4B sweep results
    └── phase4_comparison.png          ← Final comparison plot
```

## Prerequisites

Before running Phase 4, ensure you have:

1. **Completed Phase 1-3** (v6_loocv pipeline with visual features)
   - `unified_results.csv` exists in `outputs/`
   - `oracle_best_per_page.csv` exists in `outputs/`
   - `sweep_summary.csv` exists in `outputs/` (Phase 3 baseline)

2. **Python dependencies** (run once):
   ```bash
   pip install transformers torch
   ```

## Step-by-Step Instructions

### Step 1: Extract CLIP Embeddings

Extract 512-dimensional CLIP embeddings from all ledger page images:

```bash
python -m experiments.v6_loocv.extract_clip_embeddings
```

**What it does:**
- Loads openai/clip-vit-base-patch32 model (~150MB download on first run)
- Processes all PNG images in `data/images/`
- Saves embeddings to `data/visual_features/clip_embeddings.json`
- Prints similarity statistics

**Expected output:**
```
[Loaded] CLIP model (openai/clip-vit-base-patch32)...
[Extracting embeddings...]
CLIP embedding: 100%|████████████| 33/33 [00:45<00:00,  1.38s/img]
[OK] Extracted embeddings for 33/33 pages
     Embedding dimension: 512
[Saved] data/visual_features/clip_embeddings.json
```

**Time:** ~1 minute (CPU) or ~30 seconds (GPU)

---

### Step 2A: Binary Skip with CLIP (Apples-to-Apples)

Run the binary skip model using CLIP embeddings instead of visual features:

```bash
# Run full hyperparameter sweep
python -m experiments.v6_loocv.loocv_clip_binary --sweep

# Or test a single configuration
python -m experiments.v6_loocv.loocv_clip_binary --k 5 --skip-threshold 0.01 --conf 0.65
```

**What it does:**
- Same task as Phase 3: predict which extractors to skip (binary decision per extractor)
- Uses CLIP cosine similarity instead of Euclidean distance on visual features
- Sweeps over k={3,5,7}, skip_threshold={0.005,0.010,0.020,0.030}, conf={0.60,0.65,0.70}
- Saves best configuration to `outputs/sweep_clip_binary_summary.csv`

**Expected output:**
```
CLIP BINARY SWEEP (36 configs)
...
[Best Configuration — CLIP Binary]
  skip_threshold:     0.01
  confidence_threshold: 0.65
  k:                  5
  Binary accuracy:    XX.X%
  Preservation vs ensemble: XX.X%
  Cost reduction:     XX.X%
  Balanced score:     0.XXXX
```

**Time:** ~5 minutes

---

### Step 2B: Multi-Class Pipeline Selection (Full Vision)

Run the multi-class pipeline selector using CLIP embeddings:

```bash
# Run full hyperparameter sweep
python -m experiments.v6_loocv.loocv_clip_multiclass --sweep

# Or test a single configuration
python -m experiments.v6_loocv.loocv_clip_multiclass --k 5 --conf 0.50
```

**What it does:**
- Directly predicts which complete v2 pipeline to run (v2_full, v2_no_gemini, v2_no_gpt, v2_no_claude)
- Can recommend v2_no_gemini (which binary skip systems couldn't)
- Sweeps over k={3,5,7,9}, conf_threshold={0.30,0.40,0.50,0.60,0.70}
- Falls back to v2_no_claude (current SOTA) if confidence is low
- Saves best configuration to `outputs/sweep_clip_multiclass_summary.csv`

**Expected output:**
```
[Oracle pipeline distribution]
  v2_no_gemini   : 12 pages  ████████████    ← Wins most often!
  v2_no_claude   :  8 pages  ████████
  v2_no_gpt      :  8 pages  ████████
  v2_full        :  5 pages  █████
...
[Best Configuration — CLIP Multi-Class]
  k:                  5
  conf_threshold:     0.50
  Accuracy:           XX.X%
  Preservation vs ensemble: XX.X%
  Cost reduction:     XX.X%
  Balanced score:     0.XXXX
```

**Time:** ~3 minutes

---

### Step 3: Compare All Systems

Generate final comparison table and plot across all 3 systems:

```bash
python -m experiments.v6_loocv.compare_phase4
```

**What it does:**
- Loads best configs from Phase 3, 4A, and 4B
- Prints comparison table
- Generates comparison plot
- Saves summary to `outputs/phase4_final_comparison.csv`

**Expected output:**
```
PHASE 4 — FINAL COMPARISON
================================================================================
System                              Accuracy   Pres/Ens      Cost↓   Balanced
--------------------------------------------------------------------------------
  visual_binary                       XX.X%      XX.X%     XX.X%     0.XXXX
  clip_binary                         XX.X%      XX.X%     XX.X%     0.XXXX
  clip_multiclass                     XX.X%      XX.X%     XX.X%     0.XXXX ←
--------------------------------------------------------------------------------

[Winner Analysis]
  Highest Accuracy:     clip_multiclass (XX.X%)
  Best Preservation:    clip_multiclass (XX.X%)
  Most Cost Reduction:  clip_multiclass (XX.X%)
  Best Balanced:        clip_multiclass (0.XXXX)

[CLIP Binary vs Visual Features]
  Accuracy:     +X.X%  (CLIP better)
  Preservation: +X.X%  (CLIP better)
  Cost:         +X.X%  (CLIP better)

[CLIP Multi-Class vs CLIP Binary]
  Accuracy:     +X.X%  (Multi better)
  Preservation: +X.X%
  Cost:         +X.X%
  Note: multi-class can select v2_no_gemini (12/33 oracle pages),
        which the binary system could never recommend.

[Saved] phase4_comparison.png
[Saved] phase4_final_comparison.csv
```

**Time:** <10 seconds

---

## Troubleshooting

### "CLIP embeddings not found"
```
FileNotFoundError: CLIP embeddings not found: data/visual_features/clip_embeddings.json
Run this first: python -m experiments.v6_loocv.extract_clip_embeddings
```
**Solution:** Run Step 1 first.

### "Oracle file not found"
```
FileNotFoundError: Oracle file not found: experiments/v6_loocv/outputs/oracle_best_per_page.csv
```
**Solution:** Run Phase 1-3 pipeline first:
```bash
python -m experiments.v6_loocv.aggregate_reports
python -m experiments.v6_loocv.analyze_oracle
```

### "Missing dependencies"
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution:** Install dependencies:
```bash
pip install transformers torch
```

### GPU vs CPU
- CLIP extraction auto-detects CUDA if available
- CPU: ~1 minute for 33 pages
- GPU: ~30 seconds for 33 pages
- Both produce identical results

---

## Key Differences: Phase 3 vs Phase 4

| Aspect | Phase 3 (Visual Features) | Phase 4A (CLIP Binary) | Phase 4B (CLIP Multi-Class) |
|--------|---------------------------|------------------------|------------------------------|
| **Similarity measure** | 26 hand-crafted features | 512-dim CLIP embeddings | 512-dim CLIP embeddings |
| **Distance metric** | Euclidean (L2) | Cosine similarity | Cosine similarity |
| **Task** | Binary skip per extractor | Binary skip per extractor | Direct pipeline selection |
| **Can recommend v2_no_gemini?** | ❌ No | ❌ No | ✅ Yes |
| **Extractors learned** | 3 binary classifiers | 3 binary classifiers | 1 multi-class classifier |
| **Fallback** | Always keep all 3 | Always keep all 3 | v2_no_claude (SOTA) |

---

## Next Steps

After completing Phase 4:

1. **Analyze results:**
   - Check which system has best balanced score
   - Review per-page predictions in detailed CSV outputs
   - Identify pages where CLIP helps vs visual features

2. **Deploy winner:**
   - Integrate best-performing system into production pipeline
   - Create deployment config with optimal hyperparameters

3. **Future work:**
   - Fine-tune CLIP on historical ledger domain
   - Combine CLIP + visual features (ensemble)
   - Explore other vision models (ViT, DINOv2, etc.)

---

## Questions?

**Q: Why not just use CLIP everywhere?**
A: Phase 3 (visual features) is faster and lighter. CLIP is better but requires ~150MB model + GPU for real-time use. Phase 4 helps decide if the improvement is worth the cost.

**Q: Can I skip Phase 4A and go straight to 4B?**
A: Yes, but 4A gives the apples-to-apples comparison showing CLIP's impact independent of the task change.

**Q: What if I don't have a GPU?**
A: CPU works fine, just ~2x slower. Embeddings are cached after extraction, so you only pay the cost once.

**Q: Why does multi-class use different confidence thresholds than binary?**
A: Multi-class has a safe fallback (v2_no_claude), so lower confidence is acceptable. Binary skip with low confidence could skip all 3 models (dangerous).
