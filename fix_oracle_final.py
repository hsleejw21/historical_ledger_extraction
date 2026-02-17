#!/usr/bin/env python
# Fix validate_oracle.py v2-only references

filepath = "experiments/v6_loocv/analyze_oracle.py"

with open(filepath, 'r') as f:
    content = f.read()

# Replace old variable/concept references
replacements = {
    'avg_best_single': 'avg_best_ablation',
    "ensemble_vs_single_gap": "full_ensemble_vs_ablation_gap",
    'v6 prediction': 'v6 skip-model',
    'If v6 always picked best single model, avg would be': 'Best ablation avg is baseline if extractors skip',
    'Current SOTA (v2_no_claude): 0.8385': 'Full ensemble advantage helps on some pages',
}

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        print(f"[Fixed] {old}")

with open(filepath, 'w') as f:
    f.write(content)

print("\n[Complete] analyze_oracle.py updated for v2-only pipeline")
