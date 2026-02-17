# Quick fix for analyze_oracle.py to update references from old variables

with open("experiments/v6_loocv/analyze_oracle.py", "r") as f:
    content = f.read()

# Replace references to old variables
replacements = [
    ('avg_best_single', 'avg_best_ablation'),
    ('ensemble_vs_single_gap', 'full_ensemble_vs_ablation_gap'),
    ('v6 prediction', 'v6 skip-model'),
    ('If v6 always picked best single model', 'Best ablation avg is the baseline if extractors are skipped'),
    ("v6 always picked best single model, avg would be", "extractors are skipped, avg would be"),
    ("Current SOTA (v2_no_claude): 0", "Full ensemble advantage: {avg_gap:+.4f}"),
    ("Pages where ensemble helps most:", "Pages where full ensemble helps most:"),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"[Replaced] {old} -> {new}")
    else:
        print(f"[Not found] {old}")

# Write back
with open("experiments/v6_loocv/analyze_oracle.py", "w") as f:
    f.write(content)

print("\n[Done] Fixed analyze_oracle.py")
