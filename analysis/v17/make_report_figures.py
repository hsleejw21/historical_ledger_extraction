#!/usr/bin/env python
"""make_report_figures.py -- report/manuscript figures for v17, built from the committed CSVs.

Produces:
  fig_headline.png : two panels -- (L) the coupling adjustment speed lambda collapsing at 1820
                     (from v16 adjustment_speed.csv), (R) the spending/income volatility ratio flipping
                     through 1.0 at 1820 with bootstrap 95% CIs (from v17 t2_relative_volatility.csv).
  fig_timeline.png : the mechanism -> institution -> portfolio sequence (1820 / 1854 / 1870).
Run: cd experiments/reports/analysis_v17 && python make_report_figures.py
"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parent
V16 = OUT.parent / "analysis_v16"
NAVY, ORANGE, TEAL, GREY = "#1f3b5c", "#b5530f", "#2b7a72", "#9aa4ad"


def fig_headline():
    # --- left: adjustment speed lambda by era (v16) ---
    asp = pd.read_csv(V16 / "adjustment_speed.csv")
    asp = asp[~asp.period.str.contains(r"\[imposed\]")].copy()
    labels_l = ["1700-1749\n(baseline)", "1800-1819", "1820-1853", "1854-1900"]
    lam = [0.463, 0.948, 0.159, 0.185]                    # from adjustment_speed.csv
    colors_l = [GREY, NAVY, ORANGE, ORANGE]

    # --- right: smoothing ratio by era with bootstrap CI (v17) ---
    rv = pd.read_csv(OUT / "t2_relative_volatility.csv")
    labels_r = ["1700-1749\n(baseline)", "1800-1819", "1820-1853", "1854-1900"]
    ratio = rv.smoothing_ratio.values
    lo = rv.ci_lo.values; hi = rv.ci_hi.values
    colors_r = [GREY if r < 1 else NAVY for r in ratio]
    colors_r[1] = NAVY  # tight era >1

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    a = ax[0]
    bars = a.bar(range(4), lam, color=colors_l, width=0.62)
    a.axvspan(1.5, 3.5, color=ORANGE, alpha=0.06)
    a.set_xticks(range(4)); a.set_xticklabels(labels_l, fontsize=9)
    a.set_ylabel("adjustment speed  $\\lambda$", fontsize=11)
    a.set_ylim(0, 1.05)
    for i, v in enumerate(lam):
        a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    a.set_title("A.  How tightly spending tracks income\n(share of the gap closed within a year)",
                fontsize=11)
    a.annotate("coupling breaks\n~1820", xy=(2, 0.159), xytext=(2.35, 0.62),
               fontsize=9, color=ORANGE, ha="left",
               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))

    a = ax[1]
    x = np.arange(4)
    yerr = np.vstack([ratio - lo, hi - ratio])
    hatches = [None, None, None, "///"]                   # flag the contaminated late-era bar
    for i in range(4):
        a.bar(i, ratio[i], color=colors_r[i], width=0.62, hatch=hatches[i],
              edgecolor="white" if hatches[i] else colors_r[i])
    a.errorbar(x, ratio, yerr=yerr, fmt="none", ecolor="#333", elinewidth=1.2, capsize=4)
    a.axhline(1.0, color="#333", ls="--", lw=1.2)
    a.text(-0.35, 1.05, "matching", fontsize=8, color="#333", va="bottom", ha="left")
    a.text(-0.35, 0.95, "smoothing", fontsize=8, color="#333", va="top", ha="left")
    a.set_xticks(x); a.set_xticklabels(labels_r, fontsize=9)
    a.set_ylabel("spending volatility / income volatility", fontsize=11)
    a.set_ylim(0, 2.45)
    for i, v in enumerate(ratio):
        lab = f"{v:.2f}" if i != 3 else f"{v:.2f}*"
        a.text(i, hi[i] + 0.06, lab, ha="center", fontsize=10, fontweight="bold")
    a.set_title("B.  Is spending smoothed relative to income?\n(>1 chases income; <1 held on a plan)",
                fontsize=11)
    a.annotate("flips through 1.0\nat ~1820", xy=(2, 0.72), xytext=(2.25, 1.75),
               fontsize=9, color=ORANGE, ha="left",
               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    a.text(0.5, -0.19, "*1854–1900 raw shown; clean value 0.92 after removing 2 incomplete-income "
           "years (1859, 1862).\nThe artefact-free comparison is 1800–1819 vs 1820–1853.",
           transform=a.transAxes, ha="center", va="top", fontsize=7.6, color="#666", style="italic")

    fig.suptitle("The decision rule changed at ~1820: from matching spending to income, to smoothing over it",
                 fontsize=12.5, y=1.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_headline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_timeline():
    fig, ax = plt.subplots(figsize=(12.5, 2.9))
    ax.axis("off")
    ax.set_xlim(1795, 1905); ax.set_ylim(0, 1)
    ax.plot([1800, 1900], [0.5, 0.5], color=GREY, lw=2, zorder=1)
    # (marker_year, label_x, year_y, text_y_mid, text_y_small, big, mid, small, colour)
    events = [(1820, 1820, 0.72, 0.36, 0.20, "~1820", "The RULE changes",
               "matching → smoothing\n(this analysis, v17)", ORANGE),
              (1854, 1849, 0.72, 0.36, 0.20, "1854", "The INSTITUTION changes",
               "Oxford University Act\n(external reform)", NAVY),
              (1870, 1876, 0.90, 0.90 - 0.14, 0.90 - 0.28, "1870", "The PORTFOLIO changes",
               "L3/L4 spending mix\nvisibly shifts", TEAL)]
    for yr, lx, yy, ym, ys, big, mid, small, col in events:
        ax.scatter([yr], [0.5], s=140, color=col, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(big, xy=(yr, 0.5), xytext=(lx, yy), ha="center", fontsize=13,
                    fontweight="bold", color=col,
                    arrowprops=(dict(arrowstyle="-", color=col, lw=0.8) if lx != yr else None))
        ax.text(lx, ym, mid, ha="center", fontsize=9.2, fontweight="bold", color="#222")
        ax.text(lx, ys, small, ha="center", fontsize=8.1, color="#444")
    # arrows between the three, showing the lead
    for x0, x1 in [(1824, 1850), (1858, 1866)]:
        ax.add_patch(FancyArrowPatch((x0, 0.5), (x1, 0.5), arrowstyle="-|>", mutation_scale=14,
                                     color="#888", lw=0, zorder=2))
    ax.annotate("30–50 years", xy=(1845, 0.5), xytext=(1845, 0.60), ha="center", fontsize=8.5,
                color="#888", style="italic")
    ax.set_title("The mechanism moves first, the institution second, the visible portfolio last",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_headline()
    fig_timeline()
    print("Wrote fig_headline.png and fig_timeline.png to", OUT)
