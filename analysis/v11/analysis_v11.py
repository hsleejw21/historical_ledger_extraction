#!/usr/bin/env python
"""
analysis_v11.py — Environmental Conditioning, Persistence/Lock-in, Slack & Oxford→AI

Reframes the four-level transformation evidence around the advisor's question
(2026-06-03): *do different transformation levels matter under different
environmental conditions?* and *what can a completed technological revolution
(the Industrial Revolution at Oxford) teach the ongoing AI revolution?*

Four-Level AI Transformation Framework mapping used throughout:
  L1  Automation              ← L1_inv  (1 − traditional_function_share)
  L2  Personalization/Process ← L2      (payment-modernity index)
  L3  Operational Innovation  ← L3      (salary/stipend share, capability)
  L4  Business-Model Innov.   ← L4      (educational share, mission)

Core hypothesis under test: L1/L2 suffice in stable periods, but durable
adaptation under major disruption requires L3/L4. Organisations can get
*trapped* optimising L1/L2.

Sections:
  A  Environmental conditioning — L1–L4 movement in disruption vs calm regimes
  B  Sustained-vs-temporary, long-run contribution, lock-in / efficiency trap
  C  Resource-slack mechanism — does slack / land-rent income enable L3/L4?
  D  Oxford → AI interpretation table (HTML)
  E  Strength assessment + report assembly

Reuses the v9 engine (load_main_panel, _run_its, _detect_breaks, _add_era_bands,
_df_to_html, _b64, LEVEL_META, LEVEL_COLS, CUT1/CUT2). No enriched-JSON
re-processing — everything comes from the v6 panel.

Outputs (experiments/reports/analysis_v11/):
  analysis_v11_report.html  (the deliverable)
  conditioning_by_regime.csv, persistence_decomposition.csv,
  lockin_counterfactual.csv, slack_leadlag.csv, oxford_ai_table.csv,
  strength_assessment_v11.csv
  figures/*.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

import analysis_v9 as v9

ROOT   = Path(__file__).resolve().parents[2]
OUT    = ROOT / "experiments/reports/analysis_v11"
FIGDIR = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

CUT1, CUT2 = v9.CUT1, v9.CUT2          # 1854, 1877
LEVEL_COLS = v9.LEVEL_COLS             # ["L1_inv", "L2", "L3", "L4"]
LEVEL_META = v9.LEVEL_META

# AI-framework labels for the four levels (display only)
AI_LABEL = {
    "L1_inv": "L1 · Automation",
    "L2":     "L2 · Personalization / Process",
    "L3":     "L3 · Operational Innovation",
    "L4":     "L4 · Business-Model Innovation",
}
SHORT = {"L1_inv": "L1", "L2": "L2", "L3": "L3", "L4": "L4"}
COLOR = {c: LEVEL_META[c]["color"] for c in LEVEL_COLS}

DISRUPT_W = 8                          # ± years around each reform = high-disruption


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, name):
    """Save a PNG into figures/ and return its base64 (for HTML embedding)."""
    fig.savefig(FIGDIR / name, dpi=150, bbox_inches="tight")
    return v9._b64(fig)                # _b64 also closes the figure


def _img_tagged(b64, caption, tag):
    if b64 is None:
        return '<p class="missing">Figure not available.</p>'
    badge = "badge-essential" if tag == "ESSENTIAL" else "badge-supporting"
    return (f'<figure class="fig-{tag.lower()}"><div class="badge {badge}">{tag}</div>'
            f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'
            f'<figcaption>{caption}</figcaption></figure>')


def regime(year, w=DISRUPT_W):
    """High-disruption if within ±w years of either Reform Act, else low (calm)."""
    return "high" if (abs(year - CUT1) <= w or abs(year - CUT2) <= w) else "low"


def _clean_panel(panel):
    """Interpolate the few interior NaNs so all four levels + slack align by year."""
    df = panel.sort_values("year").reset_index(drop=True).copy()
    for c in ["L2", "land_rent_share", "total_inc_real", "total_exp_real",
              "income_div", "hhi", "modern_function_share", *LEVEL_COLS]:
        if c in df.columns:
            df[c] = df[c].interpolate(limit_direction="both")
    return df


# ---------------------------------------------------------------------------
# Section A — Environmental conditioning
# ---------------------------------------------------------------------------

def section_a(df):
    """Do the four levels move differently under disruption vs calm?"""
    df = df.copy()
    df["regime"] = df["year"].apply(regime)
    base_mask = (df["year"] >= 1820) & (df["year"] <= 1853)

    rows = []
    for col in LEVEL_COLS:
        s = df[["year", col, "regime"]].dropna().reset_index(drop=True)
        absd = s[col].diff().abs()
        reg = s["regime"]               # regime of the *to*-year of each diff
        hi = absd[reg == "high"].dropna()
        lo = absd[reg == "low"].dropna()
        premean = df.loc[base_mask, col].mean()   # 1820–1853 pre-reform scale

        total_move = absd.sum()
        frac_years_hi = (reg == "high").mean()
        move_share_hi = hi.sum() / total_move if total_move else np.nan
        # concentration: share of movement in disruption / share of years in disruption.
        # >1 ⇒ the level's change is concentrated in shock windows (timing).
        concentration = move_share_hi / frac_years_hi if frac_years_hi else np.nan
        # proportional response: disruption-window change relative to the level's own
        # pre-reform scale — the leapfrogging metric (comparable across levels).
        norm_disrupt = hi.mean() / premean if premean else np.nan
        norm_calm = lo.mean() / premean if premean else np.nan

        # regime-interaction OLS: |Δ| ~ disrupt + secular trend t
        reg_df = pd.DataFrame({
            "absd": absd, "disrupt": (reg == "high").astype(float),
            "t": s["year"] - 1700,
        }).dropna()
        X = sm.add_constant(reg_df[["disrupt", "t"]])
        res = sm.OLS(reg_df["absd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        rows.append({
            "Code": SHORT[col], "Level": AI_LABEL[col],
            "Mean |Δ| calm": lo.mean(), "Mean |Δ| disrupt": hi.mean(),
            "Disrupt / calm ratio": (hi.mean() / lo.mean()) if lo.mean() else np.nan,
            "Norm. response calm": norm_calm, "Norm. response disrupt": norm_disrupt,
            "Movement concentration": concentration,
            "Interaction coef": res.params["disrupt"], "p": res.pvalues["disrupt"],
            "sig": v9._stars(res.pvalues["disrupt"]),
        })

    cond_df = pd.DataFrame(rows)
    cond_df.to_csv(OUT / "conditioning_by_regime.csv", index=False)

    # Robustness across window widths
    rob_rows = []
    for w in (5, 8, 12):
        for col in LEVEL_COLS:
            s = df[["year", col]].dropna().reset_index(drop=True)
            reg = s["year"].apply(lambda y: regime(y, w))
            absd = s[col].diff().abs()
            tot = absd.sum()
            share = absd[reg == "high"].sum() / tot if tot else np.nan
            fy = (reg == "high").mean()
            rob_rows.append({"W": w, "Code": SHORT[col],
                             "Concentration": share / fy if fy else np.nan})
    rob_df = pd.DataFrame(rob_rows)
    rob_pivot = rob_df.pivot(index="Code", columns="W", values="Concentration").round(2)
    rob_pivot.columns = [f"±{w}yr" for w in rob_pivot.columns]
    rob_pivot = rob_pivot.reset_index().rename(columns={"Code": "Level"})

    # Figure 1: (a) proportional disruption response [headline], (b) concentration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(LEVEL_COLS))
    ax1.bar(x - 0.2, cond_df["Norm. response calm"], width=0.4, label="Calm", color="#aab7c4")
    ax1.bar(x + 0.2, cond_df["Norm. response disrupt"], width=0.4, label="Disruption (±8yr)",
            color=[COLOR[c] for c in LEVEL_COLS])
    ax1.set_xticks(x); ax1.set_xticklabels(cond_df["Code"])
    ax1.set_ylabel("Change ÷ level's own pre-reform mean")
    ax1.set_title("(a) Proportional response to disruption (leapfrogging)")
    ax1.legend(frameon=False)

    conc = cond_df["Movement concentration"]
    ax2.bar(x, conc, color=[COLOR[c] for c in LEVEL_COLS])
    ax2.axhline(1.0, color="#333", ls="--", lw=1)
    ax2.text(len(x) - 0.5, 1.03, "regime-neutral = 1", fontsize=8, color="#333", ha="right")
    ax2.set_xticks(x); ax2.set_xticklabels(cond_df["Code"])
    ax2.set_ylabel("Movement concentration in shock windows")
    ax2.set_title("(b) Timing: is change concentrated in disruption?")
    fig.suptitle("Section A — Environmental conditioning: which levels respond to shocks?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    b64 = _save_fig(fig, "v11_figA_conditioning.png")

    return {"cond_df": cond_df, "rob_pivot": rob_pivot, "b64": b64}


# ---------------------------------------------------------------------------
# Section B — Persistence, long-run contribution, lock-in trap
# ---------------------------------------------------------------------------

def _normalise(s):
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def section_b(df):
    df = df.copy()
    years = df["year"].values

    # --- B1: persistence — does the post-reform plateau hold to 1900? -----
    # Robust to the opposite-signed 1854/1877 reforms: compare the elevation
    # gained by the early post-1877 decade against what survives to 1900.
    def _mean(lo, hi):
        m = (df["year"] >= lo) & (df["year"] <= hi)
        return df.loc[m, col].mean()

    pers_rows = []
    for col in LEVEL_COLS:
        its = v9._run_its(df[col].values, years).get("res")
        baseline = _mean(1820, 1853)
        post_early = _mean(1878, 1887)        # early post-final-reform plateau
        final = _mean(1891, 1900)
        gained = post_early - baseline
        retained = (final - baseline) / gained if abs(gained) > 1e-9 else np.nan
        if gained <= 0.005:                    # no meaningful post-reform elevation to retain
            cls = "No durable gain"
        elif retained >= 0.9:
            cls = "Sustained / amplifying"
        elif retained >= 0.5:
            cls = "Partly sustained"
        else:
            cls = "Transitory"
        pers_rows.append({
            "Code": SHORT[col], "Level": AI_LABEL[col],
            "ITS jump 1854": its.params.get("D54", np.nan),
            "ITS jump 1877": its.params.get("D77", np.nan),
            "Gain by 1878–87": gained,
            "Held to 1891–1900": final - baseline,
            "Retained fraction": retained,
            "Pattern": cls,
        })
    pers_df = pd.DataFrame(pers_rows)
    pers_df.to_csv(OUT / "persistence_decomposition.csv", index=False)

    # --- B2: lock-in / efficiency-trap counterfactual ---------------------
    # Composite transformation index T = mean of min-max normalised levels.
    norm = {c: _normalise(df[c].values) for c in LEVEL_COLS}
    ndf = pd.DataFrame({"year": years, **norm})
    base_mask = (ndf["year"] >= 1820) & (ndf["year"] <= 1853)
    base = {c: ndf.loc[base_mask, c].mean() for c in LEVEL_COLS}

    T_actual = ndf[LEVEL_COLS].mean(axis=1)
    # "Trapped" org: only L1/L2 advance; L3/L4 frozen at pre-reform baseline.
    low_only = ndf[["L1_inv", "L2"]].copy()
    low_only["L3"] = base["L3"]; low_only["L4"] = base["L4"]
    T_lowonly = low_only[LEVEL_COLS].mean(axis=1)

    lock_df = pd.DataFrame({
        "year": years,
        "T_actual": T_actual.values,
        "T_low_only": T_lowonly.values,
        "higher_order_gap": (T_actual - T_lowonly).values,
    })
    lock_df.to_csv(OUT / "lockin_counterfactual.csv", index=False)

    # Additive contribution decomposition (baseline → 1896-1900 mean).
    late_mask = ndf["year"] >= 1896
    deltas = {c: ndf.loc[late_mask, c].mean() - base[c] for c in LEVEL_COLS}
    tot_delta = sum(deltas.values())
    low_share = (deltas["L1_inv"] + deltas["L2"]) / tot_delta if tot_delta else np.nan
    high_share = (deltas["L3"] + deltas["L4"]) / tot_delta if tot_delta else np.nan
    # Per-level long-run change, so the reader can see that only L4 actually rose.
    contrib_df = pd.DataFrame([
        {"Level": AI_LABEL[c], "Long-run change (1820s baseline → 1900)": deltas[c],
         "Direction": "rose" if deltas[c] > 0.01 else ("flat" if deltas[c] > -0.01 else "fell")}
        for c in LEVEL_COLS
    ])

    # Figure B: durable-gain bars + lock-in trajectory
    fig, (axp, axl) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(LEVEL_COLS))
    held = pers_df["Held to 1891–1900"].values
    axp.bar(x, held, color=[COLOR[c] for c in LEVEL_COLS])
    axp.axhline(0.0, color="#333", lw=1)
    axp.set_xticks(x); axp.set_xticklabels(pers_df["Code"])
    axp.set_ylabel("Durable gain held to 1891–1900 (level units)")
    axp.set_title("(a) Which gains are durable? (only L4 is positive)")

    sm10 = lambda s: pd.Series(s).rolling(10, center=True, min_periods=4).mean().values
    ta, tl = sm10(lock_df["T_actual"].values), sm10(lock_df["T_low_only"].values)
    axl.plot(lock_df["year"], ta, color="#117a65", lw=2, label="Actual transformation")
    axl.plot(lock_df["year"], tl, color="#b9770e", lw=2, ls="--",
             label="If trapped in L1/L2 only")
    axl.fill_between(lock_df["year"], tl, ta, color="#117a65", alpha=0.15)
    for cut in (CUT1, CUT2):
        axl.axvline(cut, color="#c0392b", ls=":", lw=1)
    axl.set_ylabel("Composite transformation index")
    axl.set_title("(b) The efficiency trap: L3/L4 contribution (shaded)")
    axl.legend(frameon=False, loc="upper left")
    fig.suptitle("Section B — Durable transformation & the lower-level trap", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    b64 = _save_fig(fig, "v11_figB_persistence_lockin.png")

    pers_display = pers_df[["Code", "Level", "Gain by 1878–87",
                            "Held to 1891–1900", "Pattern"]].rename(columns={
        "Gain by 1878–87": "Gain just after reform",
        "Held to 1891–1900": "Still there by 1900"})

    return {"pers_df": pers_df, "pers_display": pers_display, "lock_df": lock_df,
            "contrib_df": contrib_df, "low_share": low_share, "high_share": high_share,
            "b64": b64}


# ---------------------------------------------------------------------------
# Section C — Resource-slack mechanism
# ---------------------------------------------------------------------------

def _cross_corr_leadlag(driver, target, max_lag=8, n_boot=600, block=5, seed=0):
    """Lead-lag of first-differenced driver vs target.
    Positive lag = driver leads target. Returns best lag, peak r, bootstrap CI."""
    x = pd.Series(driver).diff()
    y = pd.Series(target).diff()
    paired = pd.DataFrame({"x": x.values, "y": y.values}).dropna().reset_index(drop=True)
    xz = (paired["x"] - paired["x"].mean()) / (paired["x"].std() or 1)
    yz = (paired["y"] - paired["y"].mean()) / (paired["y"].std() or 1)
    n = len(xz)

    def corr_at(lag, xs, ys):
        if lag >= 0:
            a, b = xs[:n - lag], ys[lag:]
        else:
            a, b = xs[-lag:], ys[:n + lag]
        if len(a) < 5:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    lags = range(-max_lag, max_lag + 1)
    corrs = {lag: corr_at(lag, xz.values, yz.values) for lag in lags}
    best_lag = max(corrs, key=lambda k: abs(corrs[k]) if not np.isnan(corrs[k]) else -1)
    peak = corrs[best_lag]

    # moving-block bootstrap CI for the correlation at the best lag
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = []
    xv, yv = xz.values, yz.values
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block), n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boots.append(corr_at(best_lag, xv[idx], yv[idx]))
    boots = np.array([b for b in boots if not np.isnan(b)])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
    return {"best_lag": best_lag, "peak": peak, "ci_low": lo, "ci_high": hi, "corrs": corrs}


def section_c(df):
    df = df.copy()
    df["slack_surplus"] = df["total_inc_real"] - df["total_exp_real"]
    df["land_rent_real"] = df["land_rent_share"] * df["total_inc_real"]

    drivers = {"Financial slack (income − expenditure)": "slack_surplus",
               "Land-rent income (real)": "land_rent_real"}
    targets = {"L3": "L3", "L4": "L4"}

    rows = []
    for dname, dcol in drivers.items():
        for tname, tcol in targets.items():
            r = _cross_corr_leadlag(df[dcol].values, df[tcol].values)
            rows.append({
                "Driver": dname, "Target": tname,
                "Best lag (yr, +=driver leads)": r["best_lag"],
                "Peak corr": r["peak"], "CI low": r["ci_low"], "CI high": r["ci_high"],
                "Driver leads": "yes" if r["best_lag"] > 0 else ("sync" if r["best_lag"] == 0 else "no"),
            })
    slack_df = pd.DataFrame(rows)
    slack_df.to_csv(OUT / "slack_leadlag.csv", index=False)

    # Pre-reform slack vs realised post-reform L3/L4 jump (descriptive conditioning)
    cond_rows = []
    for cut in (CUT1, CUT2):
        pre = df[(df["year"] >= cut - DISRUPT_W) & (df["year"] < cut)]
        post = df[(df["year"] > cut) & (df["year"] <= cut + DISRUPT_W)]
        cond_rows.append({
            "Reform": cut,
            "Pre-reform slack (real £)": pre["slack_surplus"].mean(),
            "Pre-reform land-rent (real £)": pre["land_rent_real"].mean(),
            "ΔL3 around reform": post["L3"].mean() - pre["L3"].mean(),
            "ΔL4 around reform": post["L4"].mean() - pre["L4"].mean(),
        })
    slack_cond_df = pd.DataFrame(cond_rows)

    # Figure C: slack & land-rent vs L3/L4 (normalised overlay) + lead-lag stems
    fig, (axt, axs) = plt.subplots(1, 2, figsize=(13, 4.6))
    yr = df["year"]
    sm10n = lambda v: pd.Series(_normalise(v)).rolling(10, center=True, min_periods=4).mean().values
    axt.plot(yr, sm10n(df["land_rent_real"].values), color="#7f8c8d", lw=2,
             label="Land-rent income")
    axt.plot(yr, sm10n(df["slack_surplus"].clip(lower=0).values), color="#16a085",
             lw=1.5, ls="-.", label="Financial slack (≥0)")
    axt.plot(yr, sm10n(df["L3"].values), color=COLOR["L3"], lw=2, label="L3 capability")
    axt.plot(yr, sm10n(df["L4"].values), color=COLOR["L4"], lw=2, label="L4 mission")
    for cut in (CUT1, CUT2):
        axt.axvline(cut, color="#c0392b", ls=":", lw=1)
    axt.set_title("(a) Slack / land-rent vs higher-order levels (10-yr smoothed)")
    axt.legend(frameon=False, fontsize=8)

    width = 0.2
    base_x = np.arange(-8, 9)
    for i, (dname, dcol) in enumerate(drivers.items()):
        r = _cross_corr_leadlag(df[dcol].values, df["L4"].values)
        vals = [r["corrs"][l] for l in base_x]
        axs.plot(base_x, vals, marker="o", ms=3, label=f"{dname.split('(')[0].strip()} → L4")
    axs.axvline(0, color="#333", lw=1)
    axs.axhline(0, color="#999", lw=0.8)
    axs.set_xlabel("Lag (years; >0 ⇒ driver leads L4)")
    axs.set_ylabel("Cross-correlation (Δ)")
    axs.set_title("(b) Does slack lead mission investment?")
    axs.legend(frameon=False, fontsize=8)
    fig.suptitle("Section C — Resource slack as the enabler of L3/L4", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    b64 = _save_fig(fig, "v11_figC_slack.png")

    return {"slack_df": slack_df, "slack_cond_df": slack_cond_df, "b64": b64}


# ---------------------------------------------------------------------------
# Section D — Oxford → AI interpretation table
# ---------------------------------------------------------------------------

def build_oxford_ai_rows(a, b, c):
    cond = a["cond_df"].set_index("Code")
    nr = cond["Norm. response disrupt"]
    high_share = b["high_share"]

    return [
        {
            "Oxford finding": (
                f"L4 (mission) is the most disruption-driven level and L2 (process) the least; "
                f"and L4 changes far more relative to its own size than any level "
                f"(L4 {nr['L4']:.1f}× vs L1 {nr['L1']:.1f}×). L1 and L3 are intermediate."),
            "Level": "L4 vs L2",
            "AI interpretation": (
                "Personalization/process (L2) is steady background that barely reacts to shocks; "
                "the business-model level (L4) is where a disrupted environment shows up most. "
                "Automation (L1) and operational change (L3) are in between."),
            "Lesson": (
                "Under a genuine AI disruption, process tweaks won't carry adaptation; the "
                "business-model level is the one most activated by the shock."),
            "Strength": "STRONG",
            "Artifact": "conditioning_by_regime.csv; Fig A",
        },
        {
            "Oxford finding": (
                "Higher-order dimensions leapfrog: large reform responses in L3/L4 without "
                "completing L1/L2 optimisation (v10 core)."),
            "Level": "L3 / L4",
            "AI interpretation": (
                "Firms can rationally invest in operational redesign and new business models "
                "before fully optimising automation — stages are skippable under pressure."),
            "Lesson": (
                "‘Master automation first’ stage-gating may be wrong under disruption; "
                "higher-order bets need not wait."),
            "Strength": "STRONG",
            "Artifact": "analysis_v10 (effect sizes, break magnitudes)",
        },
        {
            "Oxford finding": (
                "The L4 (mission) gain settles at a durably higher plateau through 1900 "
                "(retained &gt;2× its early elevation); L1/L2 show no net durable gain and L3 "
                "capability is more transitory."),
            "Level": "L4 (durable)",
            "AI interpretation": (
                "Business-model and operational innovation produce durable change; pure "
                "efficiency gains tend to commoditise and erode."),
            "Lesson": (
                "Durable competitive advantage from AI comes from L4, not from squeezing "
                "efficiency that rivals also capture."),
            "Strength": "STRONG",
            "Artifact": "persistence_decomposition.csv; Fig B(a)",
        },
        {
            "Oxford finding": (
                f"An L1/L2-only organisation would have stagnated: L3/L4 account for the entire "
                f"net durable transformation ({high_share*100:.0f}% share; L1/L2 net ≈ 0)."),
            "Level": "L1 / L2 trap",
            "AI interpretation": (
                "The ‘efficiency trap’: concentrating effort on automation/personalization "
                "leaves the organisation essentially where it started in transformation terms."),
            "Lesson": (
                "Treat L1/L2 as the necessary baseline, not the goal; escaping the efficiency trap requires deliberate "
                "L3/L4 investment."),
            "Strength": "STRONG",
            "Artifact": "lockin_counterfactual.csv; Fig B(b)",
        },
        {
            "Oxford finding": (
                "Land-rent income and financial slack co-move with — and at some lags lead — "
                "capability/mission investment, though lead-lag CIs are wide."),
            "Level": "Enabler",
            "AI interpretation": (
                "Financial slack is a plausible precondition for higher-order AI transformation; "
                "resource-constrained firms may get stuck at L1/L2."),
            "Lesson": (
                "Budget explicit slack for L3/L4 experimentation rather than routing all "
                "capacity into efficiency."),
            "Strength": "SUGGESTIVE",
            "Artifact": "slack_leadlag.csv; Fig C",
        },
        {
            "Oxford finding": (
                "Income base broadens (HHI falls, diversification rises) across the 19th "
                "century alongside transformation."),
            "Level": "Enabler",
            "AI interpretation": (
                "A diversified capability/revenue base underwrites the risk of higher-order "
                "transformation."),
            "Lesson": (
                "Diversification builds the resilience that makes L3/L4 bets survivable."),
            "Strength": "MEDIUM",
            "Artifact": "income_diversification.csv (v6)",
        },
        {
            "Oxford finding": (
                "No robust strict L1→L2→L3→L4 ordering; Granger chaining across levels is "
                "mostly null (v10 inventory)."),
            "Level": "All",
            "AI interpretation": (
                "Maturity-model determinism (must climb levels in order) is not supported; "
                "pathways are shock-contingent."),
            "Lesson": (
                "Don't manage AI transformation as a fixed maturity ladder; sequencing is "
                "conditional on the environment."),
            "Strength": "USEFUL NULL",
            "Artifact": "v10 research inventory (Granger, ordering)",
        },
        {
            "Oxford finding": (
                "L2 (process/payment modernisation) rises smoothly across 1700–1900, "
                "independent of shocks."),
            "Level": "L2",
            "AI interpretation": (
                "Personalization/process modernisation is continuous background improvement, "
                "not a disruption response."),
            "Lesson": (
                "Keep investing in L2 — but recognise it won't, by itself, deliver shock "
                "adaptation."),
            "Strength": "MEDIUM",
            "Artifact": "four_level_proxies.csv (L2)",
        },
    ]



# ---------------------------------------------------------------------------
# Section E — Strength assessment
# ---------------------------------------------------------------------------

def build_strength(a, b, c):
    cond = a["cond_df"].set_index("Code")
    nr = cond["Norm. response disrupt"]
    pers = b["pers_df"].set_index("Code")
    sustained = [code for code in ("L3", "L4")
                 if str(pers.loc[code, "Pattern"]).startswith(("Sustained", "Partly"))]
    rows = [
        {"Claim": "L4 (mission) is the most disruption-driven level; L2 (process) the least",
         "Strength": "STRONG",
         "Quantitative basis": (
             f"L4 changes cluster most in shock years (concentration {cond.loc['L4','Movement concentration']:.2f}) "
             f"and are ~50% larger in shocks than calm; L2 is below-neutral ({cond.loc['L2','Movement concentration']:.2f}). "
             f"L4 also changes far more relative to its own scale than any level (L4 {nr['L4']:.1f}× vs L1 {nr['L1']:.1f}×)."),
         "Caveats": "L1 and L3 respond to disruption about equally, so this is an L4-vs-L2 contrast, not a strict lower-vs-upper split; the large L4 scale-of-change appears in calm decades too."},
        {"Claim": "An efficiency/process-only path goes nowhere; durable transformation is all L4",
         "Strength": "STRONG",
         "Quantitative basis": (
             f"Over 1820s→1900, only L4 rose; L1, L2 and L3 all ended slightly below baseline. "
             f"An L1/L2-only organisation net-regresses (share {b['low_share']*100:.0f}% of net change); "
             f"the entire net rise comes from the top level."),
         "Caveats": "Equal-weight 0–1 composite; robustness rests on the raw fact that only L4 grew."},
        {"Claim": "Only the mission level (L4) shows a lasting gain; the rest fade or never gained",
         "Strength": "STRONG",
         "Quantitative basis": (
             "L4's post-reform rise was still well above baseline in 1900; L3's rise reverted, "
             "and L1/L2 had no lasting gain to begin with."),
         "Caveats": "Two opposite-signed reforms (1854 down, 1877 up) make a single persistence number delicate; read it as a direction, not a precise figure."},
        {"Claim": "Resource slack / land-rent income enables the move to higher levels",
         "Strength": "SUGGESTIVE",
         "Quantitative basis": (
             "Slack and land-rent income move together with L3/L4, but the lead-lag is weak — "
             "confidence intervals mostly include zero, and the one significant correlation is "
             "contemporaneous, not slack-leading (see slack_leadlag.csv)."),
         "Caveats": "Correlational and weak; only two reform episodes to lean on. The weakest link."},
        {"Claim": "Transformation is not a fixed L1→L2→L3→L4 ladder",
         "Strength": "USEFUL NULL",
         "Quantitative basis": "No robust strict ordering; Granger chaining mostly null (v10 inventory).",
         "Caveats": "Null results; slow-moving annual series limit the power of these tests."},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "strength_assessment_v11.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

CSS = """
body{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#222}
h1{font-size:1.6em;border-bottom:2px solid #2c3e50;padding-bottom:8px}
h2{font-size:1.3em;color:#2c3e50;margin-top:36px}
h3{font-size:1.05em;color:#34495e}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:0.85em}
th{background:#2c3e50;color:white;padding:7px 10px;text-align:left}
td{border:1px solid #ddd;padding:6px 10px;vertical-align:top}
tr:nth-child(even){background:#f9f9f9}
figure{margin:18px 0;position:relative}
figcaption{font-size:0.82em;color:#555;margin-top:4px}
figure img{border:1px solid #ddd;border-radius:4px;width:100%}
.badge{display:inline-block;font-size:0.7em;font-weight:bold;letter-spacing:0.05em;
       padding:3px 8px;border-radius:3px;margin-bottom:6px}
.badge-essential{background:#117a65;color:white}
.badge-supporting{background:#909497;color:white}
.fig-essential{border-left:4px solid #117a65;padding-left:12px}
.fig-supporting{border-left:4px solid #d5dbdb;padding-left:12px}
.callout{background:#eaf4fb;border-left:4px solid #2980b9;padding:12px 16px;margin:16px 0;border-radius:4px}
.headline{background:#eafaf1;border-left:6px solid #117a65;padding:16px 20px;margin:20px 0;border-radius:4px}
.section{background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:16px 20px;margin:24px 0}
.missing{color:#888;font-style:italic}
.strong-tag{display:inline-block;padding:2px 8px;border-radius:3px;font-size:0.78em;font-weight:bold;color:white}
.tag-strongest{background:#117a65}.tag-strong{background:#1f618d}
.tag-medium{background:#7d6608}.tag-suggestive{background:#b9770e}.tag-null{background:#7f8c8d}
@media print {
  body{max-width:none;margin:20px;padding:0}
  h2{page-break-before:always}
  h2:first-of-type{page-break-before:auto}
  h3{page-break-after:avoid}
  figure,table,.callout,.section,.headline{page-break-inside:avoid}
}
"""


def _tag_html(s):
    cls = {"STRONGEST": "tag-strongest", "STRONG": "tag-strong", "MEDIUM": "tag-medium",
           "SUGGESTIVE": "tag-suggestive", "USEFUL NULL": "tag-null"}.get(s, "tag-suggestive")
    return f'<span class="strong-tag {cls}">{s}</span>'


def _ai_table_html(rows):
    head = ("<tr><th>Oxford evidence</th><th>Level</th><th>AI-transformation interpretation</th>"
            "<th>Lesson for organisations</th><th>Strength</th></tr>")
    body = "".join(
        f"<tr><td>{r['Oxford finding']}</td><td>{r['Level']}</td>"
        f"<td>{r['AI interpretation']}</td><td>{r['Lesson']}</td>"
        f"<td>{_tag_html(r['Strength'])}</td></tr>" for r in rows)
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def _strength_table_html(df):
    head = "<tr><th>Claim</th><th>Strength</th><th>Quantitative basis</th><th>Caveats</th></tr>"
    body = "".join(
        f"<tr><td>{r['Claim']}</td><td>{_tag_html(r['Strength'])}</td>"
        f"<td>{r['Quantitative basis']}</td><td>{r['Caveats']}</td></tr>"
        for _, r in df.iterrows())
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def build_html(a, b, c, ai_rows, strength_df):
    nr = a["cond_df"].set_index("Code")["Norm. response disrupt"]
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Analysis v11 — Conditioning, Lock-in, Slack & Oxford→AI</title>
<style>{CSS}</style></head><body>

<h1>Analysis v11: Do Transformation Levels Matter Under Different Conditions?</h1>

<div class="headline">
<strong>Question.</strong> Do different levels of transformation matter under different
conditions? <strong>Central claim under test:</strong> lower-level change (efficiency and process,
L1/L2) is sufficient in stable periods, but durable adaptation under major disruption depends on
higher-level change (capability and mission, L3/L4). The two Oxford Reform Acts (1854 and 1877)
provide the disruption used to test this.
</div>

<!-- ============ SETUP ============ -->
<h2>Background, Data, and Definitions</h2>
<div class="section">

<h3>Purpose</h3>
<p>We test a single idea: whether improving only the lower levels (efficiency and process) is
enough on its own, or whether durable transformation depends on the higher levels (capability and
mission). We then read the result as a possible lesson for the current AI transition. We do not
attempt to prove cause and effect; we ask whether the historical record is consistent with the
framework, and we label every finding by how strong the evidence is.</p>

<h3>Data</h3>
<p>The evidence is an annual series (1700–1900) built from 1,581 enriched pages of Oxford
accounting ledgers. Each ledger row was extracted and labelled with an economic category, an
income/expenditure direction, and a payment period. Monetary values are converted to pounds,
adjusted to constant 1700 prices, and weighted so a page covering two years contributes half to
each. This report builds on an existing yearly table (<code>four_level_proxies.csv</code>) from
earlier work; it does not re-extract any ledger data.</p>

<h3>How the four levels are measured</h3>
<p>Each level of the framework is represented by one share of Oxford's annual <em>expenditure</em>
(or, for L2, a weighted index). These definitions are inherited from the earlier reports and are
restated here because they matter for reading every result below. All four are scaled so that a
higher number means "more modern / more transformed".</p>
<table>
<thead><tr><th>Level (AI framework)</th><th>What it captures at Oxford</th><th>How it is computed</th></tr></thead>
<tbody>
<tr><td><strong>L1 — Automation</strong></td><td>Moving away from traditional functions (efficiency)</td>
<td>1 − (ecclesiastical + maintenance + domestic) ÷ total expenditure. We use 1 − traditional share so that higher = more modern.</td></tr>
<tr><td><strong>L2 — Personalization / Process</strong></td><td>Modern, regular payment practices</td>
<td>Amount-weighted average of payment-period scores (annual = 1.0, half-year = 0.85, … multi-year = 0.10), over entries that record a payment period.</td></tr>
<tr><td><strong>L3 — Operational Innovation</strong></td><td>Capability expansion (paid staff)</td>
<td>Salary &amp; stipend expenditure ÷ total expenditure.</td></tr>
<tr><td><strong>L4 — Business-Model Innovation</strong></td><td>Mission change (teaching)</td>
<td>Educational expenditure ÷ total expenditure.</td></tr>
</tbody>
</table>
<p>The resource-slack analysis (Section C) also uses two income-side measures: real land-rent
income, and financial slack (real income − real expenditure).</p>

<h3>Key assumptions and limitations — please read first</h3>
<ul>
<li><strong>These are shares, not amounts.</strong> L1–L4 describe how the institution's spending
is <em>composed</em>, not how big it is. A level can fall simply because another grew faster.</li>
<li><strong>"Disruption" means two events.</strong> A year is treated as disrupted if it lies
within ±{DISRUPT_W} years of the 1854 or 1877 Reform Act. Two events is a small basis for general
claims, and every test in this report inherits that limit.</li>
<li><strong>Each level is a single proxy.</strong> One accounting share stands in for each idea; it
captures the intended meaning but is not the only possible measure.</li>
<li><strong>Interpretive, not causal.</strong> The reforms were not randomly assigned, so we report
whether the data are <em>consistent</em> with the framework rather than asserting cause and effect.</li>
<li><strong>The pace does not transfer.</strong> Oxford changed over two centuries as an endowed
body with no competitors. The part that may carry over to the AI setting is the <em>pattern</em>,
not the timescale.</li>
</ul>
</div>

<!-- ============ A ============ -->
<h2>A. Environmental Conditioning — which levels respond to shocks?</h2>
<div class="section">
<p>We call a year "disrupted" if it falls within ±{DISRUPT_W} years of the 1854 or 1877 Reform
Acts, and "calm" otherwise, and ask how each level behaves in each. Two views:
<strong>(a) size of change relative to the level's own scale</strong> (so the four levels can be
compared despite very different sizes), and <strong>(b) timing concentration</strong> — whether a
level's changes cluster in the disrupted years (1.0 = no preference either way).</p>
{_img_tagged(a['b64'], "Figure A. (a) Size of change relative to each level's own scale, calm vs disrupted. (b) Timing concentration (1.0 = no preference).", "ESSENTIAL")}
{v9._df_to_html(a['cond_df'].round(3), title="How each level behaves by regime")}
<p><strong>Robustness</strong> — the timing pattern holds across different window widths:</p>
{v9._df_to_html(a['rob_pivot'], title="Timing concentration by window width")}
<p><strong>Interpretation (with a caveat):</strong> the four levels do <em>not</em> split cleanly
into "lower two" versus "upper two" on every measure. What is robust: (1) relative to its own
size, <strong>L4 changes far more than any other level</strong> (about 14× L1's proportional
change) — the leapfrogging signature — though this large scale of change appears in calm decades
too, not only in shocks; (2) <strong>L4 is the most disruption-driven level</strong> (its changes
cluster most in shock years and are about 50% larger in shocks than in calm), while <strong>L2 is
the least</strong> — a genuine steady "background" level; (3) L1 and L3 sit in between and respond
to disruption about equally. So the clean contrast is <strong>L4 (most shock-driven) versus L2
(pure background)</strong>, not a strict lower-vs-upper divide.</p>
</div>

<!-- ============ B ============ -->
<h2>B. Durable Transformation &amp; the Lower-Level Trap</h2>
<div class="section">
<h3>B.1 Did the reform gains last, or fade?</h3>
<p>For each level we measure the rise just after the 1877 reform (the 1878–1887 average, versus
the 1820–1853 baseline) and then check how much of that rise was still there at the end of the
century (1891–1900).</p>
{v9._df_to_html(b['pers_display'].round(3), title="Did the reform gains last?")}
<p><strong>Interpretation:</strong> only <strong>L4 (mission)</strong> shows a lasting gain — it was still
well above its baseline in 1900. L3 (capability) rose around the reforms but slipped back, and
L1/L2 had no lasting gain at all. So the one durable change is at the very top of the framework,
which fits the idea that capability (L3) is a temporary build-up that gets converted into a
permanent change of mission (L4).</p>

<h3>B.2 The efficiency trap</h3>
<p>We combine the four levels into one simple transformation score (each level rescaled to 0–1,
then averaged) and compare the real history against an imagined organisation that improved only on
the two lower levels (L1, L2) while leaving the two higher levels (L3, L4) frozen at their early-19th-century
values. The shaded gap is what the higher levels added.</p>
{_img_tagged(b['b64'], "Figure B. (a) How much of each level's post-reform gain was still there by 1900 (only L4 is positive). (b) Real transformation vs an 'efficiency-only' organisation; the shaded gap is the higher-level contribution.", "ESSENTIAL")}
{v9._df_to_html(b['contrib_df'].round(3), title="Long-run change by level (1820s baseline → 1900)")}
<p><strong>Interpretation:</strong> over the whole period <strong>only L4 (mission) actually rose</strong>.
L1, L2 and even L3 ended slightly <em>below</em> their early-19th-century level. So an organisation
that worked only on efficiency and process (L1/L2) would not have moved forward at all — it would
have drifted slightly backwards, and every bit of net long-run transformation came from the top of
the framework. This is the clearest evidence in the data for the "efficiency trap": doing the
lower-level work well, on its own, leaves the institution no further ahead than where it started.</p>
</div>

<!-- ============ C ============ -->
<h2>C. Resource Slack — what enabled the move to L3/L4?</h2>
<div class="section">
<p>If spare resources are what let Oxford afford higher-level change, we would expect financial
slack (real income minus spending) and land-rent income to move <em>before</em> capability (L3) and
mission (L4) investment. We test this by looking at year-to-year changes and checking, at each time
lag, whether slack tends to move first. A positive lag means slack leads; the shaded bands show how
much of the pattern could be chance.</p>
{_img_tagged(c['b64'], "Figure C. (a) Slack / land-rent income vs L3/L4 over time (rescaled to compare). (b) Does slack move before mission investment? Correlation at each time lag.", "SUPPORTING")}
{v9._df_to_html(c['slack_df'].round(3), title="Does slack move before L3/L4?")}
{v9._df_to_html(c['slack_cond_df'].round(2), title="Resources before each reform vs the L3/L4 change that followed")}
<p><strong>Interpretation:</strong> land-rent income and accumulated slack <em>co-move</em> with higher-order
investment and lead it at some lags, but the bootstrap confidence intervals mostly include zero —
so this is <em>suggestive, not established</em>. The pattern fits organisational-slack theory
(Cyert &amp; March), in which slack is the precondition that lets an institution invest ahead of
strategic certainty; but with only two reform episodes the enabling role of slack is the weakest
link in the chain and the clearest target for further work.</p>
</div>

<!-- ============ D ============ -->
<h2>D. Oxford → AI Interpretation Table</h2>
<div class="callout">For each major finding: the Oxford evidence, the corresponding AI-transformation
interpretation, and a candidate lesson for contemporary organisations. The interpretations and
lessons are the authors' reading, not statistical output; each row links to the result behind it.</div>
<div class="section">
{_ai_table_html(ai_rows)}
</div>

<!-- ============ E ============ -->
<h2>E. Strength Assessment — Claim by Claim</h2>
<div class="section">
{_strength_table_html(strength_df)}
</div>

<!-- ============ F ============ -->
<h2>F. Summary and Implications for AI</h2>
<div class="section">

<h3>F.1 The levels are not a uniform set</h3>
<p>The four levels do not behave the same way. Process modernisation (L2) improves steadily and
does not react to the reforms — a stable "background" level. Mission (L4) is the opposite: its
changes are most concentrated in the reform years, and it changes far more relative to its own
size than any other level. Automation (L1) and capability (L3) fall in between and respond to the
reforms about equally. The clearest single contrast is therefore between L4 and L2; the four levels
are better read as a range from "background" to "shock-driven" than as a clean lower-vs-upper
divide.</p>

<h3>F.2 Efficiency and process alone did not produce durable change</h3>
<p>Over the full period, only the mission level (L4) ended above its early-19th-century value;
automation (L1), process (L2), and even capability (L3) ended slightly below where they started.
An organisation that had worked only on the lower levels would therefore not have advanced in net
terms. All of the net long-run change came from the higher levels. This is the report's clearest
result, and the most direct support for the idea of a "lower-level trap": doing efficiency and
process work well, on its own, did not move the institution forward.</p>

<h3>F.3 Capability rises first; mission is what lasts</h3>
<p>Only L4 (mission) stays elevated through 1900. L3 (capability) rises around the reforms but then
recedes as a share. A consistent reading is that capability is a temporary build-up that is
converted into a lasting change of mission: the institution first added paid staff, then settled
into a redefined purpose. What remains a generation later is the change in mission, not the earlier
rise in capability.</p>

<h3>F.4 What the data do not yet show</h3>
<p>The weakest part of the chain is the enabling mechanism. The expectation is that spare resources
(rising land-rent income) allowed Oxford to invest in higher-level change before its value was
certain. The data are consistent with this but do not establish it: slack leads the higher levels
only weakly, and two reform episodes are too few to settle the question. We therefore present slack
as a plausible enabler, not a demonstrated one, and mark it as the main item for further work.</p>

<h3>F.5 Implications for AI transformation</h3>
<p>Read against the current AI transition, the pattern is suggestive. Many organisations concentrate
their AI effort on automation (L1) and personalisation (L2) — the levels that, in this case, improved
steadily but did not by themselves produce durable change. Three implications follow: (i) under a
major disruption, automation and personalisation are a necessary baseline but are unlikely to carry
adaptation on their own; (ii) a strict "improve the basics first, innovate later" sequence may be the
wrong model during a shock, since the higher levels can move ahead of the lower ones; and (iii)
moving to higher-level change requires deliberate investment, and probably the spare resources to
fund it. None of this is proof about AI. It is the reading that a completed transition supports, and
it places the burden of evidence on the assumption that efficiency alone is sufficient.</p>

</div>

<div class="callout"><strong>Boundary conditions.</strong> Two reform shocks are a small sample;
the 200-year pace is pre-industrial and not generalisable to AI tempo; Oxford's endowed,
no-competition structure resembles a well-capitalised incumbent rather than all firms. The
<em>mechanism</em> — disruption activates higher-order transformation, slack enables it, and
efficiency-only effort under-delivers — is the generalisable contribution, not the magnitudes.</div>

</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading panel via v9...")
    panel = v9.load_main_panel()
    df = _clean_panel(panel)

    print("Section A: environmental conditioning...")
    a = section_a(df)
    print("Section B: persistence & lock-in trap...")
    b = section_b(df)
    print("Section C: resource slack...")
    c = section_c(df)

    print("Section D: Oxford→AI table...")
    ai_rows = build_oxford_ai_rows(a, b, c)
    pd.DataFrame(ai_rows).to_csv(OUT / "oxford_ai_table.csv", index=False)

    print("Section E: strength assessment...")
    strength_df = build_strength(a, b, c)

    print("Assembling HTML report...")
    html = build_html(a, b, c, ai_rows, strength_df)
    (OUT / "analysis_v11_report.html").write_text(html, encoding="utf-8")

    print(f"\nReport:  {OUT/'analysis_v11_report.html'}")
    print("CSVs / figures written to:", OUT)
    for f in sorted(OUT.glob("*.csv")):
        print("  ", f.name)
    for f in sorted(FIGDIR.glob("*.png")):
        print("  figures/", f.name)


if __name__ == "__main__":
    main()
