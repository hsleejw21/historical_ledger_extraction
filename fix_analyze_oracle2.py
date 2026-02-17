# Comprehensive fix for analyze_oracle.py to v2-only

with open("experiments/v6_loocv/analyze_oracle.py", "r") as f:
    lines = f.readlines()

# Find and replace specific lines
updated = False
for i, line in enumerate(lines):
    if "avg_best_single" in line:
        lines[i] = line.replace("avg_best_single", "avg_best_ablation")
        updated = True
        print(f"Line {i+1}: Fixed avg_best_single -> avg_best_ablation")
    
    if "ensemble_vs_single_gap" in line and "full_ensemble" not in line:
        lines[i] = line.replace("ensemble_vs_single_gap", "full_ensemble_vs_ablation_gap")
        updated = True
        print(f"Line {i+1}: Fixed ensemble_vs_single_gap")
    
    if "If v6 always picked best single model" in line:
        lines[i] = '    summary_lines.append(f"2. Best ablation avg ({avg_best_ablation:.4f}) is the baseline if extractors are skipped")\n'
        updated = True
        print(f"Line {i+1}: Fixed line 2 insight")
    
    if "Current SOTA (v2_no_claude)" in line:
        lines[i] = '    summary_lines.append(f"3. Full ensemble advantage: {avg_gap:+.4f} (useful on {oracle_is_full_ensemble} pages)")\n'
        updated = True
        print(f"Line {i+1}: Fixed line 3/4 insight")

# Write back
with open("experiments/v6_loocv/analyze_oracle.py", "w") as f:
    f.writelines(lines)

if updated:
    print("\n[Done] Fixed analyze_oracle.py")
else:
    print("[No changes needed]")
