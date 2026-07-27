#!/usr/bin/env python
"""
build_proxies_v13.py — essay-aligned L1-L4 proxies, beside the old ones, for comparison.

Each proxy is re-derived from the professor's essay definitions (see MAPPING.md). Rationale is
restated in the docstring of every compute_* function so no choice is unexplained:

  L1 Automation        = cost of the *unchanged* recurring operation   -> maintenance+domestic share
  L2 Personalisation   = differentiated provision to *individuals*      -> individual award/fee spend
  L3 Operational Innov.= redesign of the operating/coordination machine -> admin + accounting structure
  L4 Business Innov.   = *new* revenue business that did not exist before-> securities + fee income

The old proxies (L1=1-traditional, L2=payment-modernity, L3=salary share, L4=educational share)
are computed alongside so `proxy_comparison.png` shows exactly what changed and why.

Outputs (this folder): proxies_v13.csv, proxy_comparison.png, l4_first_appearance.csv
Run: cd experiments/reports/analysis_v13 && python build_proxies_v13.py
"""

from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
ENRICHED = ROOT / "experiments/results/enriched"
OUT = Path(__file__).resolve().parent
CUT1, CUT2 = 1854, 1877

# Phelps Brown-Hopkins price index (1700=100), copied from analysis_v6 for self-containment.
_PBH = {1700:100.0,1710:103.7,1720:101.3,1730:93.6,1740:100.0,1750:104.7,1760:115.7,1770:125.4,
        1780:138.3,1790:145.2,1800:203.4,1810:269.0,1820:213.7,1830:175.4,1840:170.3,1850:161.2,
        1860:175.4,1870:193.0,1880:182.4,1890:160.0,1900:169.7}
_PY = sorted(_PBH); _PV = [_PBH[y] for y in _PY]
def price_index(y): return _PV[0] if y<=_PY[0] else _PV[-1] if y>=_PY[-1] else float(np.interp(y,_PY,_PV))
def era_of_year(y): return "pre_industrial" if y<1780 else "transition" if y<1820 else "early_industrial" if y<1860 else "late_industrial"

PAYMENT_SCORES = {"annual":1.0,"half_year":0.85,"sesquiannual":0.70,"one_off":0.60,"biennial":0.40,
                  "triennial":0.25,"quadrennial":0.20,"quinquennial":0.15,"multi_year":0.10}

L1_TRAD = {"ecclesiastical","maintenance","domestic"}   # old-L1 traditional set
L1_UPKEEP = {"maintenance","domestic"}                  # new-L1 recurring upkeep

# Keyword rules (documented so the classification is not a black box).
RX_AWARD = re.compile(r"\b(?:scholar|exhibition|exhibitioner|prize|bachelor|commoner|servitor)", re.I)   # L2 individual awards
RX_FEE   = re.compile(r"\b(?:composition|tuition|caution|battels|matriculat|degree fee|entrance)", re.I)  # L4 education-as-revenue
# L4 securities-investment: specific instruments only (bare 'stock/funds/shares' over-match corn/livestock).
RX_SEC   = re.compile(r"\b(?:consol|dividend|per cent|per\.? ct\b|canal|railway|navigation|annuit|debenture|exchequer|bank stock|india stock)", re.I)


def _amt(r):
    def m(v):
        try: return float(v or 0)
        except (TypeError, ValueError): return 0.0
    frac = {"¼":.25,"½":.5,"¾":.75,"1/4":.25,"1/2":.5,"3/4":.75}.get(str(r.get("amount_pence_fraction")).strip(), None)
    f = frac if frac is not None else m(r.get("amount_pence_fraction"))
    return m(r.get("amount_pounds")) + m(r.get("amount_shillings"))/20.0 + (m(r.get("amount_pence_whole"))+f)/240.0

def _year(pid, fname):
    for t in (str(pid).split("_")[0], Path(fname).name.split("_")[0]):
        if t.isdigit() and 1600 < int(t) < 2000: return int(t)
    return None


def load_entries():
    """Entry-level econ records with the extra fields the new proxies need
    (person_name, description text). One row per (entry, year)."""
    rows = []
    for fp in sorted(ENRICHED.glob("*_enriched.json")):
        d = json.loads(fp.read_text())
        y = _year(d.get("page_id"), fp.name)
        if y is None: continue
        pidx = price_index(y)
        for r in d.get("rows", []):
            if not isinstance(r, dict) or str(r.get("row_type","")).lower() != "entry": continue
            dirn = str(r.get("direction") or "").lower()
            if dirn not in ("expenditure","income"): continue
            a = _amt(r)
            txt = f"{r.get('english_description') or ''} {r.get('description') or ''}"
            rows.append({"year":y, "amount_real": a/(pidx/100.0), "direction":dirn,
                         "category": r.get("category"), "payment_period": r.get("payment_period"),
                         "has_person": bool(r.get("person_name")), "text": txt})
    df = pd.DataFrame(rows)
    df["era"] = df["year"].map(era_of_year)
    return df


def load_structure():
    """Per-year accounting-structure metrics from ALL rows (entry/total/header).
    Rationale: L3 (operational innovation) is workflow/coordination redesign; in a ledger its
    trace is a more elaborate, reconciled operating model -> more distinct managed accounts
    (section_header), more subtotalling (total rows), denser pages."""
    per = {}
    for fp in sorted(ENRICHED.glob("*_enriched.json")):
        d = json.loads(fp.read_text())
        y = _year(d.get("page_id"), fp.name)
        if y is None: continue
        rows = [r for r in d.get("rows", []) if isinstance(r, dict)]
        if not rows: continue
        rt = [str(r.get("row_type","")).lower() for r in rows]
        secs = {r.get("section_header") for r in rows if r.get("section_header")}
        s = per.setdefault(y, {"pages":0,"rows":0,"entries":0,"totals":0,"sections":0})
        s["pages"]+=1; s["rows"]+=len(rows)
        s["entries"]+=rt.count("entry"); s["totals"]+=rt.count("total"); s["sections"]+=len(secs)
    out = []
    for y, s in per.items():
        out.append({"year":y,
                    "rows_per_page": s["rows"]/s["pages"],
                    "sections_per_page": s["sections"]/s["pages"],
                    "subtotal_ratio": s["totals"]/max(s["entries"],1)})
    return pd.DataFrame(out).sort_values("year")


def compute_old(df):
    """Old proxies, for comparison (the v6/v8-v11 definitions)."""
    exp = df[df.direction=="expenditure"]
    tot = exp.groupby("year").amount_real.sum().rename("texp")
    trad = exp[exp.category.isin(L1_TRAD)].groupby("year").amount_real.sum()
    sal  = exp[exp.category=="salary_stipend"].groupby("year").amount_real.sum()
    edu  = exp[exp.category=="educational"].groupby("year").amount_real.sum()
    o = pd.concat([tot], axis=1)
    o["L1_old"] = 1 - (trad/tot).reindex(o.index).fillna(0)      # efficiency = 1 - traditional
    o["L3_old"] = (sal/tot).reindex(o.index).fillna(0)
    o["L4_old"] = (edu/tot).reindex(o.index).fillna(0)
    pm = exp[exp.payment_period.isin(PAYMENT_SCORES)].copy()
    pm["s"] = pm.payment_period.map(PAYMENT_SCORES)
    num = (pm.s*pm.amount_real).groupby(pm.year).sum(); den = pm.groupby("year").amount_real.sum()
    o["L2_old"] = (num/den).reindex(o.index).fillna(np.nan)
    return o.drop(columns="texp").reset_index()


def compute_new(df, struct):
    """Essay-aligned proxies. See MAPPING.md for the per-level justification."""
    exp = df[df.direction=="expenditure"]; inc = df[df.direction=="income"]
    texp = exp.groupby("year").amount_real.sum().rename("texp")
    tinc = inc.groupby("year").amount_real.sum().rename("tinc")
    idx = texp.index

    # L1 Automation = cost weight of the unchanged recurring operation (maintenance+domestic).
    up = exp[exp.category.isin(L1_UPKEEP)].groupby("year").amount_real.sum()
    L1 = (up/texp).reindex(idx).fillna(0).rename("L1_new")

    # L2 Personalisation = expenditure differentiating provision to individuals
    # (educational spend + scholarship/exhibition/prize awards, esp. where a person is named).
    is_award = ((exp.category=="educational") | exp.text.str.contains(RX_AWARD))
    l2 = exp[is_award].groupby("year").amount_real.sum()
    L2 = (l2/texp).reindex(idx).fillna(0).rename("L2_new")

    # L3 Operational Innovation = administrative coordination + accounting-structure formalisation.
    adm = exp[exp.category=="administrative"].groupby("year").amount_real.sum()
    adm_sh = (adm/texp).reindex(idx).fillna(0)
    st = struct.set_index("year").reindex(idx)
    def z01(s):
        s = s.rolling(10, center=True, min_periods=3).mean()
        return (s - s.min())/((s.max()-s.min()) or 1.0)
    struct_idx = pd.concat([z01(st.sections_per_page), z01(st.subtotal_ratio)], axis=1).mean(axis=1)
    L3 = ((z01(adm_sh) + struct_idx.fillna(0))/2.0).rename("L3_new")

    # L4 Business Innovation = NEW revenue businesses (securities investment + education-as-fee),
    # i.e. income that did not exist as a business before (the "existed before?" test).
    sec = inc[(inc.category=="financial") & inc.text.str.contains(RX_SEC)].groupby("year").amount_real.sum()
    fee = inc[(inc.category=="educational") | inc.text.str.contains(RX_FEE)].groupby("year").amount_real.sum()
    sec_sh = (sec/tinc).reindex(idx).fillna(0).rename("L4_securities_inc")
    fee_sh = (fee/tinc).reindex(idx).fillna(0).rename("L4_fee_inc")
    L4 = (sec_sh + fee_sh).rename("L4_new")

    out = pd.concat([texp, tinc, L1, L2, adm_sh.rename("L3_admin_share"),
                     struct_idx.rename("L3_struct_idx"), L3, sec_sh, fee_sh, L4], axis=1)
    return out.reset_index()


def first_appearance(df):
    """First year each new-revenue signal appears (enforces the L4 'did it exist before?' test)."""
    inc = df[df.direction=="income"]
    rec = []
    for name, mask in [("securities_investment", (inc.category=="financial") & inc.text.str.contains(RX_SEC)),
                       ("education_fee_income", (inc.category=="educational") | inc.text.str.contains(RX_FEE))]:
        sub = inc[mask & (inc.amount_real>0)]
        if len(sub): rec.append({"signal":name, "first_year": int(sub.year.min()),
                                  "first_material_year": int(sub.groupby("year").amount_real.sum()
                                                              .pipe(lambda s: s[s>s.max()*0.05]).index.min())})
    return pd.DataFrame(rec)


def main():
    df = load_entries()
    struct = load_structure()
    old = compute_old(df); new = compute_new(df, struct)
    panel = old.merge(new, on="year", how="outer").sort_values("year")
    panel["era"] = panel.year.map(era_of_year)
    panel.to_csv(OUT/"proxies_v13.csv", index=False)
    fa = first_appearance(df); fa.to_csv(OUT/"l4_first_appearance.csv", index=False)

    # comparison figure: old vs new shape per level (each min-max normalised, 10yr rolling)
    pairs = [("L1","L1_old","L1_new","Automation / cost"),
             ("L2","L2_old","L2_new","Personalisation / revenue"),
             ("L3","L3_old","L3_new","Operational innovation / process"),
             ("L4","L4_old","L4_new","Business innovation / new markets")]
    def n01(s):
        s = s.rolling(10, center=True, min_periods=4).mean()
        return (s-s.min())/((s.max()-s.min()) or 1.0)
    fig, axes = plt.subplots(2,2, figsize=(10,6.8))
    h_old = h_new = None
    for ax,(lv,co,cn,title) in zip(axes.flatten(), pairs):
        h_old, = ax.plot(panel.year, n01(panel[co]), color="#a6a6a6", lw=1.8, ls=(0,(5,2)))
        h_new, = ax.plot(panel.year, n01(panel[cn]), color="#1f3b5c", lw=2.0)
        for c in (CUT1,CUT2): ax.axvline(c, color="#888", lw=0.8, ls=":")
        ax.set_title(f"{lv}: {title}", fontsize=10, pad=6); ax.set_ylim(-0.05,1.12)
        ax.set_xlim(1700,1900); ax.grid(axis="y", alpha=.3)
        ax.spines[["top","right"]].set_visible(False)
    fig.legend([h_old,h_new], ["old measure","v13 (essay-aligned)"], loc="upper center",
               ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5,0.945))
    fig.suptitle("Old vs essay-aligned measures (shape, each scaled to the same height)",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0,0,1,0.90]); fig.savefig(OUT/"proxy_comparison.png", dpi=150); plt.close(fig)

    print("wrote proxies_v13.csv, proxy_comparison.png, l4_first_appearance.csv")
    print(fa.to_string(index=False))
    print("\nera means (new proxies):")
    print(panel.groupby("era")[["L1_new","L2_new","L3_new","L4_new"]].mean()
          .reindex(["pre_industrial","transition","early_industrial","late_industrial"]).round(3).to_string())


if __name__ == "__main__":
    main()
