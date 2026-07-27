# Analysis Scripts

Analysis scripts for the historical ledger extraction project. All scripts read from `experiments/results/enriched/` and output to `experiments/reports/`.

## Main Analysis (Versioned)

Weekly analysis iterations. **v18 is the latest.**

| Script | Date | Focus |
|--------|------|-------|
| `analysis_v4.py` | Apr 23 | Initial novelty-phase analysis |
| `analysis_v5.py` | Apr 21 | Outcome variable, OLS specs, income composition |
| `analysis_v6.py` | Apr 27 | Four-level framework structure |
| `analysis_v7.py` | May 11 | Extended analysis |
| `analysis_v8.py` | May 18 | Section-based structure |
| `analysis_v9.py` | May 25 | Event-time plots, Reform Acts |
| `analysis_v10.py` | May 31 | Strongest-vs-Suggestive assessment |
| `analysis_v11.py` | Jun 8 | Strength assessment + report assembly |
| `analysis_v14.py` | Jun 29 | L4 split analysis, lead-lag |
| `analysis_v15.py` | Jul 6 | Revenue/investment portfolio, reallocation |
| `analysis_v16.py` | Jul 13 | Spending adjustment to income gap (~1820) |
| `analysis_v17.py` | Jul 20 | Organizational decision rule change |
| `analysis_v18.py` | Jul 27 | **Current** - Latest analysis |

## Topic-Specific Scripts

| Script | Purpose |
|--------|---------|
| `analysis_embeddings.py` | Embedding-based analysis |
| `analysis_vocabulary.py` | Innovation vocabulary analysis |
| `analysis_text_trends.py` | Text trend analysis over time |
| `analysis_supplier_networks.py` | Supplier network analysis |
| `analysis_hypotheses.py` | Hypothesis testing report (May 4) |

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `paper_figures.py` | Generate figures for paper/manuscript |
| `reanalysis_ledger_yearly_v2.py` | Yearly re-analysis |
| `variable_validation.py` | Variable validation (from v12) |
| `build_proxies_v13.py` | Build proxy variables (from v13) |
| `finer_unit.py` | Finer unit analysis (from v13) |
| `make_report_figures.py` | Report figures (from v17) |

## misc/

Early exploratory scripts (embeddings, clustering).

## Usage

```bash
# Run latest analysis
python experiments/analysis/analysis_v18.py

# Run specific version
python experiments/analysis/analysis_v15.py
```

Outputs go to `experiments/reports/analysis_v*/`.
