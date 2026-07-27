#!/usr/bin/env python
"""
paper_figures.py — Research-grade figures for the v10 LaTeX working paper.

Distinct from the v9/v10 HTML report figures: those bake explanatory suptitles and
multi-line subplot titles into the image (appropriate for a browseable report). For a
paper, the LaTeX caption carries all description, so these figures are deliberately
clean — no suptitles, minimal titles, consistent muted palette, serif fonts, 300 dpi.

Reuses already-computed numbers by reading the v10 CSV outputs + the v6 panel; performs
no statistical re-estimation. Run standalone, or via analysis_v10.generate paths.

  python experiments/analysis/paper_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
V6   = ROOT / "experiments/reports/analysis_v6"
V10  = ROOT / "experiments/reports/analysis_v10"

CUT1, CUT2 = 1854, 1877

# Consistent, slightly desaturated project palette (L1 blue, L2 green, L3 orange, L4 purple)
COLORS = {
    "L1_inv": "#3b6ea5",
    "L2":     "#2e8b57",
    "L3":     "#cc7a30",
    "L4":     "#7d5ba6",
}
SHORT = {"L1_inv": "L1", "L2": "L2", "L3": "L3", "L4": "L4"}
LABEL = {
    "L1_inv": "L1: Efficiency",
    "L2":     "L2: Process",
    "L3":     "L3: Capability",
    "L4":     "L4: Mission",
}
LEVEL_COLS = ["L1_inv", "L2", "L3", "L4"]

ERA_COLORS = {
    "pre_industrial":   "#eaf2f8",
    "transition":       "#eafaf1",
    "early_industrial": "#fef5e7",
    "late_industrial":  "#fdedec",
}


def _apply_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "dejavuserif",
        "font.size":          11,
        "axes.titlesize":     11,
        "axes.labelsize":     11,
        "axes.titleweight":   "normal",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.labelsize":    9.5,
        "ytick.labelsize":    9.5,
        "legend.frameon":     False,
        "legend.fontsize":    9,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.35,
        "grid.linestyle":     ":",
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })


def _reform_lines(ax, label1=True):
    ax.axvline(CUT1, color="#7b241c", ls="--", lw=0.9, alpha=0.75,
               label=f"{CUT1} Reform Act" if label1 else None)
    ax.axvline(CUT2, color="#7b241c", ls=":", lw=0.9, alpha=0.75,
               label=f"{CUT2} Reform Act" if label1 else None)


def _era_bands(ax, df):
    if "era" not in df.columns:
        return
    for era in ["pre_industrial", "transition", "early_industrial", "late_industrial"]:
        sub = df[df["era"] == era]["year"]
        if len(sub):
            ax.axvspan(sub.min(), sub.max(), alpha=1.0,
                       color=ERA_COLORS.get(era, "#f4f4f4"), zorder=0)


def _save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data loading (mirror of v9.load_main_panel, kept self-contained)
# ---------------------------------------------------------------------------

def _load_panel():
    df = pd.read_csv(V6 / "four_level_proxies.csv").sort_values("year").reset_index(drop=True)
    df["L1_inv"]      = 1.0 - df["L1"]
    df["L1_inv_10yr"] = df["L1_inv"].rolling(10, center=True, min_periods=5).mean()
    for col in ["L2", "L3", "L4"]:
        df[f"{col}_10yr"] = df[col].rolling(10, center=True, min_periods=5).mean()
    return df


# ---------------------------------------------------------------------------
# Figure 1 — four dimensions (2x2)
# ---------------------------------------------------------------------------

def fig1_dimensions(panel, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), sharex=True)
    axes = axes.flatten()
    roll = {"L1_inv": "L1_inv_10yr", "L2": "L2_10yr", "L3": "L3_10yr", "L4": "L4_10yr"}
    for ax, col in zip(axes, LEVEL_COLS):
        _era_bands(ax, panel)
        ax.scatter(panel["year"], panel[col], s=5, alpha=0.30, color=COLORS[col],
                   edgecolors="none", zorder=2)
        ax.plot(panel["year"], panel[roll[col]], color=COLORS[col], lw=1.8, zorder=3)
        _reform_lines(ax, label1=False)
        ax.set_title(LABEL[col])
        ax.set_ylabel("Expenditure share")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y")
    for ax in axes[2:]:
        ax.set_xlabel("Year")
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_dimensions.png")


# ---------------------------------------------------------------------------
# Figure 2 — normalised trajectories
# ---------------------------------------------------------------------------

def fig2_normalised(out_dir):
    df = pd.read_csv(V10 / "normalised_trajectories.csv")
    df = df[df["year"] >= 1820]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for col in LEVEL_COLS:
        roll = df[f"{col}_norm"].rolling(10, center=True, min_periods=5).mean()
        ax.plot(df["year"], roll, color=COLORS[col], lw=2.0, label=LABEL[col], zorder=3)
        ax.scatter(df["year"], df[f"{col}_norm"], s=4, alpha=0.18,
                   color=COLORS[col], edgecolors="none", zorder=2)
    ax.axhline(1.0, color="#555555", lw=0.8, ls="--", alpha=0.7)
    _reform_lines(ax, label1=True)
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalised index (1820--1853 mean $=1$)")
    ax.grid(axis="y")
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_normalised.png")


# ---------------------------------------------------------------------------
# Figure 3 — normalised ITS effect sizes (1854 vs 1877)
# ---------------------------------------------------------------------------

def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fig3_its_effects(out_dir):
    df = pd.read_csv(V10 / "effect_sizes.csv")
    order = LEVEL_COLS  # L1..L4 bottom-to-top
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, reform in zip(axes, [CUT1, CUT2]):
        sub = df[df["Reform"] == reform].set_index("Code").loc[order]
        vals = sub["Norm. effect size"].values
        errs = (sub["SE"] / sub["Pre-reform mean"]).values
        ypos = np.arange(len(order))
        ax.barh(ypos, vals, xerr=errs, color=[COLORS[c] for c in order],
                alpha=0.9, error_kw={"ecolor": "#444444", "capsize": 3, "lw": 0.8})
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([SHORT[c] for c in order])
        ax.set_xlabel("Effect size ($\\times$ pre-reform mean)")
        ax.set_title(f"{reform} Reform Act")
        ax.grid(axis="x")
        # x-limits with headroom so significance stars never collide with the spine
        lo = np.nanmin(vals - np.nan_to_num(errs))
        hi = np.nanmax(vals + np.nan_to_num(errs))
        pad = 0.18 * (hi - lo)
        ax.set_xlim(min(lo - pad, -pad), hi + pad)
        for i, c in enumerate(order):
            s = _stars(float(sub.loc[c, "p"]))
            if s:
                offset = (errs[i] if not np.isnan(errs[i]) else 0.02) + 0.03 * (hi - lo)
                xp = vals[i] + (offset if vals[i] >= 0 else -offset)
                ha = "left" if vals[i] >= 0 else "right"
                ax.text(xp, i, s, va="center", ha=ha, fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_its_effects.png")


# ---------------------------------------------------------------------------
# Figure 4 — relative break magnitudes
# ---------------------------------------------------------------------------

def fig4_breaks(out_dir):
    df = pd.read_csv(V10 / "break_magnitudes.csv").set_index("Code").loc[LEVEL_COLS]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    vals = df["Rel. magnitude"].values
    bars = ax.bar([SHORT[c] for c in LEVEL_COLS], vals,
                  color=[COLORS[c] for c in LEVEL_COLS], alpha=0.9, width=0.62)
    ax.set_ylabel(r"$|\mathrm{post}-\mathrm{pre}|\,/\,\mathrm{pre\ mean}$")
    ax.grid(axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                f"{v:.2f}$\\times$", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_breaks.png")


# ---------------------------------------------------------------------------
# Figure 5 — signal-to-noise of 1854 break
# ---------------------------------------------------------------------------

def fig5_snr(out_dir):
    df = pd.read_csv(V10 / "signal_to_noise.csv").set_index("Code").loc[LEVEL_COLS]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    vals = df["Signal-to-noise ratio"].values
    bars = ax.bar([SHORT[c] for c in LEVEL_COLS], vals,
                  color=[COLORS[c] for c in LEVEL_COLS], alpha=0.9, width=0.62)
    ax.axhline(1.0, color="#7b241c", ls="--", lw=0.9, alpha=0.8)
    ax.text(3.45, 1.02, "noise level", color="#7b241c", fontsize=8.5,
            ha="right", va="bottom")
    ax.set_ylabel("Break magnitude / pre-reform std.")
    ax.grid(axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_snr.png")


# ---------------------------------------------------------------------------
# Figure 6 — sharpness (observed vs linear expectation)
# ---------------------------------------------------------------------------

def fig6_sharpness(out_dir):
    df = pd.read_csv(V10 / "sharpness.csv").set_index("Code").loc[LEVEL_COLS]
    obs = df["Sharpness (5yr/total)"].values
    exp = df["Linear expected sharpness"].values
    x = np.arange(len(LEVEL_COLS))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(x - 0.19, obs, width=0.36, color=[COLORS[c] for c in LEVEL_COLS],
           alpha=0.9, label="Observed")
    ax.bar(x + 0.19, exp, width=0.36, color="#9aa0a6", alpha=0.7,
           label="Linear-trend expectation")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in LEVEL_COLS])
    ax.set_ylabel("Share of total gain in first 5 years")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_sharpness.png")


# ---------------------------------------------------------------------------
# Figure A1 — post-reform acceleration
# ---------------------------------------------------------------------------

def figA1_acceleration(out_dir):
    df = pd.read_csv(V10 / "post_reform_acceleration.csv")
    windows = ["1854–1877", "1877–1900"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=False)
    for ax, w in zip(axes, windows):
        sub = df[df["Window"] == w].set_index("Code").loc[LEVEL_COLS]
        vals = sub["% change (10yr rolling)"].values
        bars = ax.bar([SHORT[c] for c in LEVEL_COLS], vals,
                      color=[COLORS[c] for c in LEVEL_COLS], alpha=0.9, width=0.62)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(w.replace("–", "--"))
        ax.set_ylabel("\\% change in 10yr rolling mean")
        ax.grid(axis="y")
        for bar, v in zip(bars, vals):
            ypos = bar.get_height() + (np.sign(v) * max(abs(vals)) * 0.03)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{v:.0f}\\%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    return _save(fig, out_dir, "figA1_acceleration.png")


# ---------------------------------------------------------------------------
# Figure A2 — event-time response (±15 yr) around each reform
# ---------------------------------------------------------------------------

def figA2_eventtime(panel, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    window = 15
    for ax, cut in zip(axes, [CUT1, CUT2]):
        trange = list(range(-window, window + 1))
        for col in LEVEL_COLS:
            ev = []
            for t in trange:
                row = panel[panel["year"] == cut + t]
                ev.append(row[col].iloc[0] if len(row) else np.nan)
            ax.plot(trange, ev, color=COLORS[col], lw=1.6, label=LABEL[col])
            ax.scatter(trange, ev, s=7, color=COLORS[col], alpha=0.45, edgecolors="none")
        ax.axvline(0, color="#7b241c", ls="--", lw=1.0, alpha=0.8)
        ax.set_xlabel(f"Years relative to {cut}")
        ax.set_title(f"{cut} Reform Act")
        ax.grid(axis="y")
    axes[0].set_ylabel("Expenditure share")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "figA2_eventtime.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all(out_fig_dir):
    out_fig_dir = Path(out_fig_dir)
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    panel = _load_panel()

    paths = [
        fig1_dimensions(panel, out_fig_dir),
        fig2_normalised(out_fig_dir),
        fig3_its_effects(out_fig_dir),
        fig4_breaks(out_fig_dir),
        fig5_snr(out_fig_dir),
        fig6_sharpness(out_fig_dir),
        figA1_acceleration(out_fig_dir),
        figA2_eventtime(panel, out_fig_dir),
    ]
    return paths


def main():
    out_fig_dir = V10 / "latex" / "figures"
    paths = generate_all(out_fig_dir)
    print(f"Figures written to: {out_fig_dir}")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
