"""
experiments/v6_loocv/compare_phase4.py

Produces the final comparison table across all three systems:

  System 1: Visual features + binary skip  (Phase 3, your existing results)
  System 2: CLIP + binary skip             (Phase 4A, apples-to-apples)
  System 3: CLIP + multi-class pipeline    (Phase 4B, full vision)

Reads the best-config sweep CSV from each system and builds one
unified comparison table + plot.

Usage:
    python -m experiments.v6_loocv.compare_phase4

Requires (run these first):
    python -m experiments.v6_loocv.loocv_clip_binary    --sweep
    python -m experiments.v6_loocv.loocv_clip_multiclass --sweep
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import BASE_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "v6_loocv", "outputs")


# ── Load results ───────────────────────────────────────────────────────────

def load_phase3_best():
    """
    Read the Phase 3 sweep summary and extract the best config row.
    Normalises column names to a shared schema.
    """
    path = os.path.join(OUTPUT_DIR, "sweep_summary.csv")
    if not os.path.exists(path):
        print(f"[Warning] Phase 3 sweep not found: {path}")
        return None

    df = pd.read_csv(path)

    # Column name variants produced by different sweep script versions
    rename = {
        "bin_acc":       "accuracy",
        "binary_acc":    "accuracy",
        "pres_ens":      "pres_ensemble",
        "pres_ensemble": "pres_ensemble",
        "cost_reduction":"cost_reduction",
        "cost↓":         "cost_reduction",
        "balanced_score":"balanced",
        "balanced":      "balanced",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    best = df.sort_values("balanced", ascending=False).iloc[0]

    return {
        "system":        "Visual Features\n+ Binary Skip\n(Phase 3)",
        "method":        "visual_binary",
        "accuracy":      best.get("accuracy",      best.get("bin_acc", float("nan"))),
        "pres_ensemble": best.get("pres_ensemble", best.get("pres_ens", float("nan"))),
        "cost_reduction":best.get("cost_reduction",float("nan")),
        "balanced":      best.get("balanced",      float("nan")),
        "best_config":   f"skip={best.get('skip_threshold', best.get('skip','?'))}, "
                         f"conf={best.get('conf_threshold', best.get('conf','?'))}, "
                         f"k={best.get('k','?')}",
    }


def load_clip_binary_best():
    path = os.path.join(OUTPUT_DIR, "sweep_clip_binary_summary.csv")
    if not os.path.exists(path):
        print(f"[Warning] CLIP binary sweep not found: {path}")
        return None

    df  = pd.read_csv(path)
    best = df.sort_values("balanced", ascending=False).iloc[0]

    return {
        "system":        "CLIP Embeddings\n+ Binary Skip\n(Phase 4A)",
        "method":        "clip_binary",
        "accuracy":      best["bin_acc"],
        "pres_ensemble": best["pres_ensemble"],
        "cost_reduction":best["cost_reduction"],
        "balanced":      best["balanced"],
        "best_config":   f"skip={best['skip_threshold']}, "
                         f"conf={best['conf_threshold']}, k={best['k']}",
    }


def load_clip_multiclass_best():
    path = os.path.join(OUTPUT_DIR, "sweep_clip_multiclass_summary.csv")
    if not os.path.exists(path):
        print(f"[Warning] CLIP multiclass sweep not found: {path}")
        return None

    df  = pd.read_csv(path)
    best = df.sort_values("balanced", ascending=False).iloc[0]

    return {
        "system":        "CLIP Embeddings\n+ Multi-Class\n(Phase 4B)",
        "method":        "clip_multiclass",
        "accuracy":      best["accuracy"],
        "pres_ensemble": best["pres_ensemble"],
        "cost_reduction":best["cost_reduction"],
        "balanced":      best["balanced"],
        "best_config":   f"conf={best['conf_threshold']}, k={best['k']}",
    }


# ── Print comparison table ─────────────────────────────────────────────────

def print_comparison(systems):
    print(f"\n{'='*80}")
    print("PHASE 4 — FINAL COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'System':<35s} {'Accuracy':>10s} {'Pres/Ens':>10s} {'Cost↓':>8s} {'Balanced':>10s}")
    print("-"*80)

    best_balanced = max(s["balanced"] for s in systems)

    for s in systems:
        marker = " ←" if s["balanced"] == best_balanced else ""
        print(f"  {s['method']:<33s} {s['accuracy']:>9.1%} {s['pres_ensemble']:>9.1%} "
              f"{s['cost_reduction']:>7.1%} {s['balanced']:>9.4f}{marker}")

    print("-"*80)

    # Winner analysis
    print(f"\n[Winner Analysis]")
    best_acc  = max(systems, key=lambda s: s["accuracy"])
    best_pres = max(systems, key=lambda s: s["pres_ensemble"])
    best_cost = max(systems, key=lambda s: s["cost_reduction"])
    best_bal  = max(systems, key=lambda s: s["balanced"])

    print(f"  Highest Accuracy:     {best_acc['method']}  ({best_acc['accuracy']:.1%})")
    print(f"  Best Preservation:    {best_pres['method']}  ({best_pres['pres_ensemble']:.1%})")
    print(f"  Most Cost Reduction:  {best_cost['method']}  ({best_cost['cost_reduction']:.1%})")
    print(f"  Best Balanced:        {best_bal['method']}  ({best_bal['balanced']:.4f})")

    # CLIP vs visual features analysis
    visual = next((s for s in systems if s["method"] == "visual_binary"), None)
    clip_b = next((s for s in systems if s["method"] == "clip_binary"),   None)

    if visual and clip_b:
        acc_delta  = clip_b["accuracy"]      - visual["accuracy"]
        pres_delta = clip_b["pres_ensemble"] - visual["pres_ensemble"]
        cost_delta = clip_b["cost_reduction"]- visual["cost_reduction"]

        print(f"\n[CLIP Binary vs Visual Features]")
        print(f"  Accuracy:     {acc_delta:+.1%}  "
              f"({'CLIP better' if acc_delta > 0 else 'Visual better'})")
        print(f"  Preservation: {pres_delta:+.1%}  "
              f"({'CLIP better' if pres_delta > 0 else 'Visual better'})")
        print(f"  Cost:         {cost_delta:+.1%}  "
              f"({'CLIP better' if cost_delta > 0 else 'Visual better'})")

    # Multi-class analysis
    clip_m = next((s for s in systems if s["method"] == "clip_multiclass"), None)
    if clip_b and clip_m:
        print(f"\n[CLIP Multi-Class vs CLIP Binary]")
        print(f"  Accuracy:     {clip_m['accuracy'] - clip_b['accuracy']:+.1%}  "
              f"({'Multi better' if clip_m['accuracy'] > clip_b['accuracy'] else 'Binary better'})")
        print(f"  Preservation: {clip_m['pres_ensemble'] - clip_b['pres_ensemble']:+.1%}")
        print(f"  Cost:         {clip_m['cost_reduction'] - clip_b['cost_reduction']:+.1%}")
        print(f"  Note: multi-class can select v2_no_gemini (12/33 oracle pages),")
        print(f"        which the binary system could never recommend.")

    print(f"\n[Best Configurations]")
    for s in systems:
        print(f"  {s['method']:<20s}: {s['best_config']}")


# ── Visualisation ──────────────────────────────────────────────────────────

def plot_comparison(systems):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Phase 4: Adaptive Routing — System Comparison", fontsize=15, fontweight="bold")

    colors = {
        "visual_binary":   "#4C72B0",
        "clip_binary":     "#DD8452",
        "clip_multiclass": "#55A868",
    }
    labels = {
        "visual_binary":   "Visual Features\n+ Binary Skip",
        "clip_binary":     "CLIP\n+ Binary Skip",
        "clip_multiclass": "CLIP\n+ Multi-Class",
    }

    metrics = [
        ("accuracy",       "Prediction Accuracy",        0.5, 1.0,  0.65, "Target: 65%"),
        ("pres_ensemble",  "Performance Preservation\nvs Full Ensemble", 0.8, 1.1, 0.95, "Target: 95%"),
        ("cost_reduction", "API Cost Reduction",          0.0, 0.6,  0.30, "Target: 30%"),
    ]

    for ax, (metric, title, ymin, ymax, target, target_label) in zip(axes, metrics):
        bars = []
        for s in systems:
            val = s.get(metric, 0)
            bar = ax.bar(
                labels[s["method"]],
                val,
                color=colors[s["method"]],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                width=0.5,
            )
            bars.append(bar)
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                val + (ymax - ymin) * 0.02,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.axhline(target, color="red", linestyle="--", linewidth=1.5,
                   label=target_label, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(title.split("\n")[0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "phase4_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] phase4_comparison.png")
    plt.close()


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loaders = [load_phase3_best, load_clip_binary_best, load_clip_multiclass_best]
    systems = [r for loader in loaders if (r := loader()) is not None]

    if not systems:
        print("[Error] No sweep results found. Run the sweeps first:")
        print("  python -m experiments.v6_loocv.loocv_clip_binary    --sweep")
        print("  python -m experiments.v6_loocv.loocv_clip_multiclass --sweep")
        sys.exit(1)

    print_comparison(systems)
    plot_comparison(systems)

    # Save summary CSV
    summary_df = pd.DataFrame([
        {k: v for k, v in s.items() if k != "system"} for s in systems
    ])
    summary_path = os.path.join(OUTPUT_DIR, "phase4_final_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] phase4_final_comparison.csv")