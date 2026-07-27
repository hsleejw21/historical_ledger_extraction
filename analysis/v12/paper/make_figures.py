#!/usr/bin/env python
"""
make_figures.py — journal-grade figures for the v12 two-author paper.

Upgrades the v10 `paper_figures.py` aesthetic to publication grade:
  - vector PDF output (editable text, pdf.fonttype 42),
  - Okabe–Ito colourblind-safe palette,
  - serif fonts matching the LaTeX body, no in-figure titles where the caption carries them,
  - consistent sizing.

Adds four figures for the integrated story: regime conditioning, persistence/durability,
the efficiency-trap counterfactual (all v11), and the L4 textual triangulation (v12).

Reuses already-computed CSVs — performs NO statistical re-estimation.

  cd experiments/reports/analysis_v12/paper && python make_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
V6   = ROOT / "experiments/reports/analysis_v6"
V7   = ROOT / "experiments/reports/analysis_v7"
V10  = ROOT / "experiments/reports/analysis_v10"
V11  = ROOT / "experiments/reports/analysis_v11"
V12  = ROOT / "experiments/reports/analysis_v12"
OUT  = Path(__file__).resolve().parent / "figures"

CUT1, CUT2 = 1854, 1877

# Restrained two-tone scheme: colour encodes only the operational (grey) vs strategic (navy)
# axis; line style separates the two depths within each axis. This keeps every figure
# near-monochrome and professional, with the mission depth (L4) the single most salient series.
COLORS = {"L1_inv": "#a6a6a6", "L2": "#6f6f6f", "L3": "#90a4ba", "L4": "#1f3b5c"}
LINESTYLE = {"L1_inv": (0, (1, 1)), "L2": (0, (5, 2)), "L3": (0, (6, 1, 1, 1)), "L4": "-"}
REFORM = "#333333"
GREY   = "#8a8a8a"
GREY_LT = "#cccccc"
SHORT  = {"L1_inv": "L1", "L2": "L2", "L3": "L3", "L4": "L4"}
LABEL  = {"L1_inv": "L1: Task / efficiency", "L2": "L2: Process",
          "L3": "L3: Capability", "L4": "L4: Mission"}
LEVEL_COLS = ["L1_inv", "L2", "L3", "L4"]
CODE_MAP = {"L1": "L1_inv", "L2": "L2", "L3": "L3", "L4": "L4"}  # v11 'Code' -> palette key

# Neutral alternating era bands (no hue)
ERA_COLORS = {"pre_industrial": "#ffffff", "transition": "#f2f2f2",
              "early_industrial": "#ffffff", "late_industrial": "#f2f2f2"}


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
        "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.frameon": False, "legend.fontsize": 11,
        "grid.linewidth": 0.5, "grid.alpha": 0.35, "grid.linestyle": ":",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _reform_lines(ax, label=True):
    ax.axvline(CUT1, color=REFORM, ls="--", lw=0.9, alpha=0.8,
               label=f"{CUT1} Reform Act" if label else None)
    ax.axvline(CUT2, color=REFORM, ls=":", lw=1.0, alpha=0.8,
               label=f"{CUT2} Reform Act" if label else None)


def _era_bands(ax, df):
    if "era" not in df.columns:
        return
    for era in ["pre_industrial", "transition", "early_industrial", "late_industrial"]:
        sub = df[df["era"] == era]["year"]
        if len(sub):
            ax.axvspan(sub.min(), sub.max(), color=ERA_COLORS.get(era, "#f4f4f4"), zorder=0)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _panel():
    df = pd.read_csv(V6 / "four_level_proxies.csv").sort_values("year").reset_index(drop=True)
    df["L1_inv"] = 1.0 - df["L1"]
    for col in LEVEL_COLS:
        df[f"{col}_10yr"] = df[col].rolling(10, center=True, min_periods=5).mean()
    return df


# --- Fig 1: four depths -----------------------------------------------------

def fig_dimensions(panel):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True)
    for ax, col in zip(axes.flatten(), LEVEL_COLS):
        _era_bands(ax, panel)
        ax.scatter(panel["year"], panel[col], s=5, alpha=0.28, color=COLORS[col],
                   edgecolors="none", zorder=2)
        ax.plot(panel["year"], panel[f"{col}_10yr"], color=COLORS[col], lw=1.8, zorder=3)
        _reform_lines(ax, label=False)
        ax.set_title(LABEL[col]); ax.set_ylabel("Expenditure share")
        ax.set_ylim(0, 1.0); ax.grid(axis="y")
    for ax in axes[1]:
        ax.set_xlabel("Year")
    fig.tight_layout()
    return _save(fig, "fig_dimensions.png")


# --- Fig 2: normalised leapfrogging -----------------------------------------

def fig_normalised():
    df = pd.read_csv(V10 / "normalised_trajectories.csv")
    df = df[df["year"] >= 1820]
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for col in LEVEL_COLS:
        roll = df[f"{col}_norm"].rolling(10, center=True, min_periods=5).mean()
        ax.plot(df["year"], roll, color=COLORS[col], lw=2.0, ls=LINESTYLE[col],
                label=LABEL[col], zorder=3)
    ax.axhline(1.0, color=GREY, lw=0.8, ls="--", alpha=0.7)
    _reform_lines(ax, label=True)
    ax.set_xlabel("Year"); ax.set_ylabel(r"Normalised index (1820--1853 mean $=1$)")
    ax.grid(axis="y"); ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig_normalised.png")


# --- Fig 3: normalised ITS effects ------------------------------------------

def fig_its_effects():
    df = pd.read_csv(V10 / "effect_sizes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, reform in zip(axes, [CUT1, CUT2]):
        sub = df[df["Reform"] == reform].set_index("Code").loc[LEVEL_COLS]
        vals = sub["Norm. effect size"].values
        errs = (sub["SE"] / sub["Pre-reform mean"]).values
        ypos = np.arange(len(LEVEL_COLS))
        ax.barh(ypos, vals, xerr=errs, color=[COLORS[c] for c in LEVEL_COLS],
                alpha=0.92, error_kw={"ecolor": "#444444", "capsize": 3, "lw": 0.8})
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ypos); ax.set_yticklabels([SHORT[c] for c in LEVEL_COLS])
        ax.set_xlabel(r"Effect size ($\times$ pre-reform mean)")
        ax.set_title(f"{reform} Reform Act"); ax.grid(axis="x")
        lo = np.nanmin(vals - np.nan_to_num(errs)); hi = np.nanmax(vals + np.nan_to_num(errs))
        pad = 0.18 * (hi - lo); ax.set_xlim(min(lo - pad, -pad), hi + pad)
        for i, c in enumerate(LEVEL_COLS):
            s = _stars(float(sub.loc[c, "p"]))
            if s:
                off = (errs[i] if not np.isnan(errs[i]) else 0.02) + 0.03 * (hi - lo)
                xp = vals[i] + (off if vals[i] >= 0 else -off)
                ax.text(xp, i, s, va="center", ha="left" if vals[i] >= 0 else "right", fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig_its_effects.png")


# --- Fig 4: relative break magnitudes ---------------------------------------

def fig_breaks():
    df = pd.read_csv(V10 / "break_magnitudes.csv").set_index("Code").loc[LEVEL_COLS]
    vals = df["Rel. magnitude"].values
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    bars = ax.bar([SHORT[c] for c in LEVEL_COLS], vals,
                  color=[COLORS[c] for c in LEVEL_COLS], alpha=0.92, width=0.62)
    ax.set_ylabel(r"$|\mathrm{post}-\mathrm{pre}|\,/\,\mathrm{pre\ mean}$"); ax.grid(axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                rf"{v:.2f}$\times$", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    return _save(fig, "fig_breaks.png")


# --- Fig 5: discontinuity (SNR + sharpness) ---------------------------------

def fig_discontinuity():
    snr = pd.read_csv(V10 / "signal_to_noise.csv").set_index("Code").loc[LEVEL_COLS]
    sh  = pd.read_csv(V10 / "sharpness.csv").set_index("Code").loc[LEVEL_COLS]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

    v = snr["Signal-to-noise ratio"].values
    bars = axes[0].bar([SHORT[c] for c in LEVEL_COLS], v,
                       color=[COLORS[c] for c in LEVEL_COLS], alpha=0.92, width=0.62)
    axes[0].axhline(1.0, color=REFORM, ls="--", lw=0.9, alpha=0.85)
    axes[0].text(3.45, 1.03, "noise level", color=REFORM, fontsize=8.5, ha="right", va="bottom")
    axes[0].set_ylabel("Break magnitude / pre-reform std."); axes[0].grid(axis="y")
    axes[0].set_title("(a) Signal-to-noise of 1854 break")
    for b, val in zip(bars, v):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9.5)
    axes[0].set_ylim(0, max(v) * 1.2)

    obs = sh["Sharpness (5yr/total)"].values; exp = sh["Linear expected sharpness"].values
    x = np.arange(len(LEVEL_COLS))
    axes[1].bar(x - 0.19, obs, width=0.36, color=[COLORS[c] for c in LEVEL_COLS],
                alpha=0.92, label="Observed")
    axes[1].bar(x + 0.19, exp, width=0.36, color=GREY_LT, alpha=0.9,
                label="Linear-trend expectation")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([SHORT[c] for c in LEVEL_COLS])
    axes[1].set_ylabel("Share of total gain in first 5 yr"); axes[1].grid(axis="y")
    axes[1].set_title("(b) Sharpness of 1854 response"); axes[1].legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig_discontinuity.png")


# --- Fig 6: regime conditioning (calm vs disruption) [v11] ------------------

def fig_conditioning():
    df = pd.read_csv(V11 / "conditioning_by_regime.csv").set_index("Code").loc[["L1", "L2", "L3", "L4"]]
    calm = df["Norm. response calm"].values
    disr = df["Norm. response disrupt"].values
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.bar(x - 0.2, calm, width=0.38, color=GREY_LT, alpha=0.9, label="Calm years")
    ax.bar(x + 0.2, disr, width=0.38,
           color=[COLORS[CODE_MAP[c]] for c in df.index], alpha=0.92, label="Disruption years ($\\pm$8 yr)")
    ax.set_xticks(x); ax.set_xticklabels(list(df.index))
    ax.set_ylabel("Movement relative to own scale"); ax.grid(axis="y"); ax.legend(loc="upper left")
    for xi, (c, d) in enumerate(zip(calm, disr)):
        ax.text(xi + 0.2, d + 0.03 * max(disr), f"{d:.2f}", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "fig_conditioning.png")


# --- Fig 7: persistence / durability [v11] ----------------------------------

def fig_persistence():
    df = pd.read_csv(V11 / "persistence_decomposition.csv").set_index("Code").loc[["L1", "L2", "L3", "L4"]]
    held = df["Held to 1891–1900"].values
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    bars = ax.bar(x, held, color=[COLORS[CODE_MAP[c]] for c in df.index], alpha=0.92, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{p}" for c, p in zip(df.index, df["Pattern"])], fontsize=8.5)
    ax.set_ylabel("Durable gain retained to 1891--1900"); ax.grid(axis="y")
    for b, v in zip(bars, held):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.01),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    pad = max(abs(held)) * 0.25
    ax.set_ylim(min(held) - pad, max(held) + pad)
    fig.tight_layout()
    return _save(fig, "fig_persistence.png")


# --- Fig 8: efficiency-trap counterfactual [v11] ----------------------------

def fig_efficiency_trap():
    df = pd.read_csv(V11 / "lockin_counterfactual.csv").sort_values("year")
    ya = df["T_actual"].rolling(10, center=True, min_periods=4).mean()
    yl = df["T_low_only"].rolling(10, center=True, min_periods=4).mean()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(df["year"], ya, color=COLORS["L4"], lw=2.0, label="Actual (all depths)")
    ax.plot(df["year"], yl, color=GREY, lw=2.0, ls="--", label="Counterfactual: L1/L2 only")
    ax.fill_between(df["year"], yl, ya, where=(ya >= yl), color=COLORS["L4"], alpha=0.12,
                    label="Higher-order contribution (L3/L4)")
    _reform_lines(ax, label=True)
    ax.set_xlabel("Year"); ax.set_ylabel("Transformation index (10-yr rolling)")
    ax.grid(axis="y"); ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    return _save(fig, "fig_efficiency_trap.png")


# --- Fig 9: L4 textual triangulation [v12] ----------------------------------

def fig_l4_triangulation():
    df = pd.read_csv(V12 / "l4_triangulation_panel.csv").sort_values("year")

    def norm(s):
        s = s.rolling(10, min_periods=1).mean()
        return (s - s.min()) / ((s.max() - s.min()) or 1.0)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(df["year"], norm(df["L4"]), color=COLORS["L4"], lw=2.2,
            label="L4: educational-spend share (main proxy)")
    ax.plot(df["year"], norm(df["textual_mission_index"]), color="#444444", lw=1.8, ls="--",
            label="Textual mission index (scholarships, exams, science)")
    _reform_lines(ax, label=True)
    ax.set_xlabel("Year"); ax.set_ylabel("Normalised (10-yr rolling)")
    ax.grid(axis="y"); ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig_l4_triangulation.png")


# --- Fig 10: causal robustness (v7) -----------------------------------------

def fig_causal_robustness():
    its = pd.read_csv(V7 / "its_results.csv").sort_values("year")
    rdd = pd.read_csv(V7 / "rdd_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

    # (a) higher-order axis (MFS = L3+L4) vs no-reform ITS counterfactual
    mfs_roll = its["modern_function_share"].rolling(10, center=True, min_periods=5).mean()
    axes[0].scatter(its["year"], its["modern_function_share"], s=5, alpha=0.25,
                    color=COLORS["L4"], edgecolors="none")
    axes[0].plot(its["year"], mfs_roll, color=COLORS["L4"], lw=2.0, label="Actual (L3+L4)")
    axes[0].plot(its["year"], its["counterfactual"], color=GREY, lw=1.8, ls="--",
                 label="No-reform counterfactual")
    _reform_lines(axes[0], label=False)
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Modern-function share (L3 + L4)")
    axes[0].set_title("(a) Higher-order axis vs ITS counterfactual")
    axes[0].grid(axis="y"); axes[0].legend(loc="upper left")

    # (b) RDD level shift at each reform, across bandwidths
    cut_color = {1854: COLORS["L4"], 1877: COLORS["L3"]}
    for cut in (1854, 1877):
        sub = rdd[rdd["cutoff"] == cut].sort_values("bandwidth")
        yerr = np.vstack([sub["tau"] - sub["ci_lo"], sub["ci_hi"] - sub["tau"]])
        axes[1].errorbar(sub["bandwidth"], sub["tau"], yerr=yerr, fmt="o", ms=6,
                         color=cut_color[cut], capsize=3, lw=1.0, label=f"{cut} reform")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Bandwidth (years)"); axes[1].set_ylabel(r"RDD level shift $\tau$")
    axes[1].set_title("(b) Regression-discontinuity estimates"); axes[1].grid(axis="y")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "fig_causal_robustness.png")


# --- Appendix figures -------------------------------------------------------

def figA_acceleration():
    df = pd.read_csv(V10 / "post_reform_acceleration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, w in zip(axes, ["1854–1877", "1877–1900"]):
        sub = df[df["Window"] == w].set_index("Code").loc[LEVEL_COLS]
        vals = sub["% change (10yr rolling)"].values
        bars = ax.bar([SHORT[c] for c in LEVEL_COLS], vals,
                      color=[COLORS[c] for c in LEVEL_COLS], alpha=0.92, width=0.62)
        ax.axhline(0, color="black", lw=0.8); ax.set_title(w.replace("–", "--"))
        ax.set_ylabel(r"\% change in 10yr rolling mean"); ax.grid(axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + np.sign(v) * max(abs(vals)) * 0.03,
                    f"{v:.0f}\\%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    return _save(fig, "figA_acceleration.png")


def figA_eventtime(panel):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    window = 15
    for ax, cut in zip(axes, [CUT1, CUT2]):
        trange = list(range(-window, window + 1))
        for col in LEVEL_COLS:
            ev = [panel.loc[panel["year"] == cut + t, col].iloc[0]
                  if (panel["year"] == cut + t).any() else np.nan for t in trange]
            ax.plot(trange, ev, color=COLORS[col], lw=1.7, ls=LINESTYLE[col], label=LABEL[col])
        ax.axvline(0, color=REFORM, ls="--", lw=1.0, alpha=0.8)
        ax.set_xlabel(f"Years relative to {cut}"); ax.set_title(f"{cut} Reform Act"); ax.grid(axis="y")
    axes[0].set_ylabel("Expenditure share"); axes[1].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "figA_eventtime.png")


def main():
    _style()
    panel = _panel()
    paths = [
        fig_dimensions(panel), fig_normalised(), fig_its_effects(), fig_breaks(),
        fig_discontinuity(), fig_conditioning(), fig_persistence(),
        fig_efficiency_trap(), fig_l4_triangulation(), fig_causal_robustness(),
        figA_acceleration(), figA_eventtime(panel),
    ]
    print(f"Wrote {len(paths)} figures to {OUT}:")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
