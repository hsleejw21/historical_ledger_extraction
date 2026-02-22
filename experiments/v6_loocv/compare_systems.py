"""
experiments/v6_loocv/compare_systems.py

Compares three adaptive routing approaches:

  System 1: Visual features + binary skip
  System 2: CLIP embeddings + binary skip (apples-to-apples comparison)
  System 3: CLIP embeddings + multi-class pipeline selection

Reads the best-config sweep CSV from each system and builds one
unified comparison table + plot.

Usage:
    python -m experiments.v6_loocv.compare_systems

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

def load_visual_binary_best():
    """
    Read the visual features binary sweep summary and extract the best config row.
    Normalises column names to a shared schema.
    """
    path = os.path.join(OUTPUT_DIR, "sweep_summary.csv")
    if not os.path.exists(path):
        print(f"[Warning] Visual binary sweep not found: {path}")
        return None

    df = pd.read_csv(path)

    # Column name variants produced by different sweep script versions
    rename = {
        "overall_binary_accuracy": "accuracy",
        "bin_acc":                 "accuracy",
        "binary_acc":              "accuracy",
        "performance_preservation_vs_ensemble": "pres_ensemble",
        "pres_ens":                "pres_ensemble",
        "pres_ensemble":           "pres_ensemble",
        "cost_reduction":          "cost_reduction",
        "cost↓":                   "cost_reduction",
        "balanced_score":          "balanced",
        "balanced":                "balanced",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    best = df.sort_values("balanced", ascending=False).iloc[0]

    return {
        "system":          "Visual Features\n+ Binary Skip",
        "method":          "visual_binary",
        "accuracy":        best.get("accuracy",      best.get("bin_acc", float("nan"))),
        "pres_ensemble":   best.get("pres_ensemble", best.get("pres_ens", float("nan"))),
        "cost_reduction":  best.get("cost_reduction",float("nan")),
        "balanced":        best.get("balanced",      float("nan")),
        "final_avg":       best.get("final_avg",     float("nan")),
        "avg_axis1":       best.get("avg_axis1",     float("nan")),
        "avg_axis2":       best.get("avg_axis2",     float("nan")),
        "avg_axis2_match": best.get("avg_axis2_match", float("nan")),
        "avg_axis2_sim":   best.get("avg_axis2_sim", float("nan")),
        "avg_axis2_frac":  best.get("avg_axis2_frac", float("nan")),
        "best_config":     f"skip={best.get('skip_threshold', best.get('skip','?'))}, "
                           f"conf={best.get('confidence_threshold', best.get('conf_threshold', best.get('conf','?')))}, "
                           f"k={best.get('k','?')}",
        "min_score":       best.get("min_score",  float("nan")),
        "score_std":       best.get("score_std",  float("nan")),
    }


def load_sota_baseline():
    """Load v2_no_claude average combined score as the fixed SOTA baseline."""
    path = os.path.join(OUTPUT_DIR, "unified_results.csv")
    if not os.path.exists(path):
        print(f"[Warning] unified_results.csv not found - cannot compute SOTA baseline")
        return None

    df   = pd.read_csv(path)
    sota = df[df["pipeline"] == "v2_no_claude"]
    full = df[df["pipeline"] == "v2_full"]
    if sota.empty:
        return None

    avg_combined  = sota["final_combined"].mean()
    full_avg      = full["final_combined"].mean() if not full.empty else float("nan")
    pres_ensemble = avg_combined / full_avg if full_avg else float("nan")

    return {
        "system":          "SOTA Baseline\n(v2_no_claude, always)",
        "method":          "sota_v2_no_claude",
        "accuracy":        float("nan"),
        "pres_ensemble":   pres_ensemble,
        "cost_reduction":  1 / 3,
        "balanced":        float("nan"),
        "final_avg":       avg_combined,
        "best_config":     "fixed policy - always run v2_no_claude",
        "avg_axis1":       sota["final_axis1"].mean()      if "final_axis1"      in sota.columns else float("nan"),
        "avg_axis2":       sota["final_axis2"].mean()      if "final_axis2"      in sota.columns else float("nan"),
        "avg_axis2_match": sota["axis2_match"].mean()      if "axis2_match"      in sota.columns else float("nan"),
        "avg_axis2_sim":   sota["axis2_similarity"].mean() if "axis2_similarity" in sota.columns else float("nan"),
        "avg_axis2_frac":  sota["axis2_fraction"].mean()   if "axis2_fraction"   in sota.columns else float("nan"),
        "min_score":       sota["final_combined"].min()    if "final_combined"   in sota.columns else float("nan"),
        "score_std":       sota["final_combined"].std()    if "final_combined"   in sota.columns else float("nan"),
    }


def load_clip_binary_best():
    path = os.path.join(OUTPUT_DIR, "sweep_clip_binary_summary.csv")
    if not os.path.exists(path):
        print(f"[Warning] CLIP binary sweep not found: {path}")
        return None

    df  = pd.read_csv(path)
    best = df.sort_values("balanced", ascending=False).iloc[0]

    return {
        "system":          "CLIP Embeddings\n+ Binary Skip",
        "method":          "clip_binary",
        "accuracy":        best["bin_acc"],
        "pres_ensemble":   best["pres_ensemble"],
        "cost_reduction":  best["cost_reduction"],
        "balanced":        best["balanced"],
        "final_avg":       best.get("final_avg",     float("nan")),
        "avg_axis1":       best.get("avg_axis1",     float("nan")),
        "avg_axis2":       best.get("avg_axis2",     float("nan")),
        "avg_axis2_match": best.get("avg_axis2_match", float("nan")),
        "avg_axis2_sim":   best.get("avg_axis2_sim", float("nan")),
        "avg_axis2_frac":  best.get("avg_axis2_frac", float("nan")),
        "best_config":     f"skip={best['skip_threshold']}, "
                           f"conf={best['conf_threshold']}, k={best['k']}",
        "min_score":       best.get("min_score",  float("nan")),
        "score_std":       best.get("score_std",  float("nan")),
    }


def load_clip_multiclass_best():
    path = os.path.join(OUTPUT_DIR, "sweep_clip_multiclass_summary.csv")
    if not os.path.exists(path):
        print(f"[Warning] CLIP multiclass sweep not found: {path}")
        return None

    df  = pd.read_csv(path)
    best = df.sort_values("balanced", ascending=False).iloc[0]

    return {
        "system":          "CLIP Embeddings\n+ Multi-Class",
        "method":          "clip_multiclass",
        "accuracy":        best["accuracy"],
        "pres_ensemble":   best["pres_ensemble"],
        "cost_reduction":  best["cost_reduction"],
        "balanced":        best["balanced"],
        "final_avg":       best.get("final_avg",     float("nan")),
        "avg_axis1":       best.get("avg_axis1",     float("nan")),
        "avg_axis2":       best.get("avg_axis2",     float("nan")),
        "avg_axis2_match": best.get("avg_axis2_match", float("nan")),
        "avg_axis2_sim":   best.get("avg_axis2_sim", float("nan")),
        "avg_axis2_frac":  best.get("avg_axis2_frac", float("nan")),
        "best_config":     f"conf={best['conf_threshold']}, k={best['k']}",
        "min_score":       best.get("min_score",  float("nan")),
        "score_std":       best.get("score_std",  float("nan")),
    }


# ── Print comparison table ─────────────────────────────────────────────────

def _fmt(val, pct=True):
    if isinstance(val, float) and np.isnan(val):
        return "n/a"
    return f"{val:.1%}" if pct else f"{val:.4f}"


def print_comparison(systems):
    print(f"\n{'='*95}")
    print("ADAPTIVE ROUTING - SYSTEM COMPARISON (vs SOTA: v2_no_claude)")
    print(f"{'='*95}")

    sota_score = next(
        (s["final_avg"] for s in systems if s["method"] == "sota_v2_no_claude"),
        float("nan")
    )
    routing = [s for s in systems if s["method"] != "sota_v2_no_claude"]
    best_balanced_val = max(
        (s["balanced"] for s in routing if not np.isnan(s["balanced"])),
        default=float("nan")
    )

    print(f"\n  {'Method':<26s} {'AvgScore':>9s} {'vs SOTA':>9s} "
          f"{'Pres/Ens':>9s} {'Cost_r':>7s} {'Accuracy':>9s} {'Balanced':>9s}")
    print("-"*95)

    for s in systems:
        if s["method"] == "sota_v2_no_claude":
            print(f"  {'[SOTA] v2_no_claude':<26s} {s['final_avg']:.4f}    {'baseline':>9s} "
                  f"{_fmt(s['pres_ensemble']):>9s} {_fmt(s['cost_reduction']):>7s} "
                  f"{'n/a':>9s} {'n/a':>9s}")
            print("  " + "-"*93)
            continue
        marker = " <-" if not np.isnan(s["balanced"]) and s["balanced"] == best_balanced_val else ""
        delta  = s["final_avg"] - sota_score if not (np.isnan(s["final_avg"]) or np.isnan(sota_score)) else float("nan")
        score_str = f"{s['final_avg']:.4f}" if not np.isnan(s["final_avg"]) else "   n/a"
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "   n/a"
        print(f"  {s['method']:<26s} {score_str:>9s} {delta_str:>9s} "
              f"{_fmt(s['pres_ensemble']):>9s} {_fmt(s['cost_reduction']):>7s} "
              f"{_fmt(s['accuracy']):>9s} {_fmt(s['balanced'], pct=False):>9s}{marker}")

    print("-"*95)

    # Winner analysis (routing systems only)
    print(f"\n[Winner Analysis - Routing Systems]")
    best_acc  = max(routing, key=lambda s: s["accuracy"] if not np.isnan(s["accuracy"]) else -1)
    best_pres = max(routing, key=lambda s: s["pres_ensemble"] if not np.isnan(s["pres_ensemble"]) else -1)
    best_cost = max(routing, key=lambda s: s["cost_reduction"] if not np.isnan(s["cost_reduction"]) else -1)
    best_bal  = max(routing, key=lambda s: s["balanced"] if not np.isnan(s["balanced"]) else -1)
    best_score= max(routing, key=lambda s: s["final_avg"] if not np.isnan(s["final_avg"]) else -1)

    print(f"  Highest Avg Score:    {best_score['method']}  ({best_score['final_avg']:.4f})")
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
        print(f"  Note: multi-class directly selects the full pipeline (4-class);")
        print(f"        binary independently predicts skip/keep per extractor (3 binary decisions).")

    print(f"\n[Best Configurations]")
    for s in systems:
        print(f"  {s['method']:<20s}: {s['best_config']}")

    # Axis2 breakdown section
    print(f"\n{'='*95}")
    print("AXIS2 SCORE BREAKDOWN (vs SOTA: v2_no_claude)")
    print(f"{'='*95}")
    print(f"\n  {'Method':<26s} {'CombScore':>9s} {'Axis1':>7s} {'Axis2':>7s} "
          f"{'Match':>7s} {'Simil':>7s} {'Frac':>7s}")
    print("-"*75)
    for s in systems:
        method = f"[SOTA] {s['method']}" if s["method"] == "sota_v2_no_claude" else s["method"]
        print(f"  {method:<26s} "
              f"{_fmt(s.get('final_avg',     float('nan')), pct=False):>9s} "
              f"{_fmt(s.get('avg_axis1',     float('nan')), pct=False):>7s} "
              f"{_fmt(s.get('avg_axis2',     float('nan')), pct=False):>7s} "
              f"{_fmt(s.get('avg_axis2_match',float('nan')),pct=False):>7s} "
              f"{_fmt(s.get('avg_axis2_sim', float('nan')), pct=False):>7s} "
              f"{_fmt(s.get('avg_axis2_frac',float('nan')), pct=False):>7s}")
    print("-"*75)
    print(f"  Axis2 = 0.5*Match + 0.3*Similarity + 0.2*Fraction")

    # Min score and variance section (robustness objective)
    print(f"\n{'='*75}")
    print("SCORE ROBUSTNESS (per-page worst-case and spread)")
    print(f"{'='*75}")
    print(f"  Objective: highest min_score = best worst-case guarantee")
    print(f"             lowest score_std = most stable method")
    print(f"\n  {'Method':<26s} {'MinScore':>9s} {'StdDev':>9s} {'AvgScore':>9s}")
    print("-"*60)
    best_min = max(
        (s.get("min_score", float("nan")) for s in systems if not np.isnan(s.get("min_score", float("nan")))),
        default=float("nan")
    )
    best_std = min(
        (s.get("score_std", float("nan")) for s in systems if not np.isnan(s.get("score_std", float("nan")))),
        default=float("nan")
    )
    for s in systems:
        method = f"[SOTA] {s['method']}" if s["method"] == "sota_v2_no_claude" else s["method"]
        min_s  = s.get("min_score", float("nan"))
        std_s  = s.get("score_std", float("nan"))
        avg_s  = s.get("final_avg",  float("nan"))
        flags = []
        if not np.isnan(min_s) and min_s == best_min:
            flags.append("highest min")
        if not np.isnan(std_s) and std_s == best_std:
            flags.append("lowest std")
        marker = f" <-- {', '.join(flags)}" if flags else ""
        print(f"  {method:<26s} "
              f"{_fmt(min_s, pct=False):>9s} "
              f"{_fmt(std_s, pct=False):>9s} "
              f"{_fmt(avg_s, pct=False):>9s}{marker}")


# ── Visualisation ──────────────────────────────────────────────────────────

def plot_comparison(summary_df):
    preferred_order = [
        "accuracy", "pres_ensemble", "cost_reduction", "balanced", "final_avg",
        "avg_axis1", "avg_axis2", "avg_axis2_match", "avg_axis2_sim", "avg_axis2_frac",
        "min_score", "score_std",
    ]

    excluded_cols = {"method", "best_config", "is_highest_min", "is_lowest_std"}
    numeric_cols = [
        col for col in summary_df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(summary_df[col])
    ]

    ordered_metrics = [col for col in preferred_order if col in numeric_cols]
    ordered_metrics += [col for col in numeric_cols if col not in ordered_metrics]

    n_metrics = len(ordered_metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.6 * n_rows))
    fig.suptitle("Adaptive Routing — All Metrics from system_comparison.csv", fontsize=15, fontweight="bold")
    axes = np.array(axes).reshape(-1)

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

    metric_labels = {
        "accuracy": "Prediction Accuracy",
        "pres_ensemble": "Performance Preservation vs Full Ensemble",
        "cost_reduction": "API Cost Reduction",
        "balanced": "Balanced Score",
        "final_avg": "Average Combined Score",
        "avg_axis1": "Average Axis1 Score",
        "avg_axis2": "Average Axis2 Score",
        "avg_axis2_match": "Average Axis2 Match",
        "avg_axis2_sim": "Average Axis2 Similarity",
        "avg_axis2_frac": "Average Axis2 Fraction",
        "min_score": "Worst-Case Score (min_score)",
        "score_std": "Score Variability (score_std, lower better)",
    }

    target_lines = {
        "accuracy": (0.65, "Target: 65%"),
        "pres_ensemble": (0.95, "Target: 95%"),
        "cost_reduction": (0.30, "Target: 30%"),
    }

    pct_metrics = {"accuracy", "pres_ensemble", "cost_reduction"}

    for i, metric in enumerate(ordered_metrics):
        ax = axes[i]
        metric_df = summary_df[summary_df[metric].notna()].copy()
        if metric_df.empty:
            ax.axis("off")
            continue

        bars = []
        for _, row in metric_df.iterrows():
            method = row["method"]
            val = float(row[metric])
            bar = ax.bar(
                labels.get(method, method),
                val,
                color=colors.get(method, "#777777"),
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                width=0.5,
            )
            bars.append(bar)
            value_label = f"{val:.1%}" if metric in pct_metrics else f"{val:.4f}"

            values = metric_df[metric].to_numpy(dtype=float)
            value_max = np.max(values)
            value_min = np.min(values)
            span = max(value_max - value_min, 1e-6)

            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                val + span * 0.03,
                value_label,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        title = metric_labels.get(metric, metric)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(title)

        if metric in target_lines:
            target, target_label = target_lines[metric]
            ax.axhline(target, color="red", linestyle="--", linewidth=1.5,
                       label=target_label, alpha=0.8)

        values = metric_df[metric].to_numpy(dtype=float)
        value_max = np.max(values)
        value_min = np.min(values)
        span = max(value_max - value_min, 1e-6)
        padding = max(span * 0.15, 0.02 if metric in pct_metrics else 0.005)
        ax.set_ylim(value_min - padding, value_max + padding)

        if metric in pct_metrics:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))

        if metric in target_lines:
            ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)

    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "system_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] system_comparison.png")
    plt.close()


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loaders = [load_sota_baseline, load_visual_binary_best,
               load_clip_binary_best, load_clip_multiclass_best]
    systems = [r for loader in loaders if (r := loader()) is not None]

    if not systems:
        print("[Error] No sweep results found. Run the sweeps first:")
        print("  python -m experiments.v6_loocv.loocv_clip_binary    --sweep")
        print("  python -m experiments.v6_loocv.loocv_clip_multiclass --sweep")
        sys.exit(1)

    print_comparison(systems)

    # Save summary CSV
    valid_min_scores = [s.get("min_score", float("nan")) for s in systems if not np.isnan(s.get("min_score", float("nan")))]
    valid_std_scores = [s.get("score_std", float("nan")) for s in systems if not np.isnan(s.get("score_std", float("nan")))]
    best_min = max(valid_min_scores) if valid_min_scores else float("nan")
    best_std = min(valid_std_scores) if valid_std_scores else float("nan")

    summary_df = pd.DataFrame([
        {k: v for k, v in s.items() if k != "system"} for s in systems
    ])
    summary_df["is_highest_min"] = summary_df["min_score"].apply(
        lambda v: bool(not np.isnan(v) and not np.isnan(best_min) and v == best_min)
    )
    summary_df["is_lowest_std"] = summary_df["score_std"].apply(
        lambda v: bool(not np.isnan(v) and not np.isnan(best_std) and v == best_std)
    )
    summary_path = os.path.join(OUTPUT_DIR, "system_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] system_comparison.csv")

    # Build PNG directly from the saved summary CSV (single source of truth)
    summary_df_from_csv = pd.read_csv(summary_path)
    plot_comparison(summary_df_from_csv)