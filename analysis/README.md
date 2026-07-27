# Analysis Scripts

Analysis scripts for the historical ledger extraction project. Each version folder contains the script and its outputs together.

## Structure

```
analysis/
├── v1/ ~ v18/           # Versioned analysis (script + outputs)
├── shared/              # Common utility scripts
├── hypothesis_report/   # Hypothesis testing report
└── README.md
```

## Main Analysis (Versioned)

Weekly analysis iterations. **v18 is the latest.**

| Version | Date | Focus |
|---------|------|-------|
| v4 | Apr 23 | Initial novelty-phase analysis |
| v5 | Apr 21 | Outcome variable, OLS specs, income composition |
| v6 | Apr 27 | Four-level framework structure |
| v7 | May 11 | Extended analysis |
| v8 | May 18 | Section-based structure |
| v9 | May 25 | Event-time plots, Reform Acts |
| v10 | May 31 | Strongest-vs-Suggestive assessment |
| v11 | Jun 8 | Strength assessment + report assembly |
| v14 | Jun 29 | L4 split analysis, lead-lag |
| v15 | Jul 6 | Revenue/investment portfolio, reallocation |
| v16 | Jul 13 | Spending adjustment to income gap (~1820) |
| v17 | Jul 20 | Organizational decision rule change |
| v18 | Jul 27 | **Current** — Latest analysis |

## Shared Scripts (`shared/`)

Common analysis utilities:

| Script | Purpose |
|--------|---------|
| `analysis_embeddings.py` | Embedding-based analysis |
| `analysis_vocabulary.py` | Innovation vocabulary analysis |
| `analysis_text_trends.py` | Text trend analysis over time |
| `analysis_supplier_networks.py` | Supplier network analysis |
| `paper_figures.py` | Generate figures for paper/manuscript |
| `reanalysis_ledger_yearly_v2.py` | Yearly re-analysis |
| `misc/` | Early exploratory scripts |

## Version-Specific Utilities

| Script | Location | Purpose |
|--------|----------|---------|
| `variable_validation.py` | v12/ | Variable validation |
| `build_proxies_v13.py` | v13/ | Build proxy variables |
| `finer_unit.py` | v13/ | Finer unit analysis |
| `make_report_figures.py` | v17/ | Report figures |
| `analysis_hypotheses.py` | hypothesis_report/ | Hypothesis testing (May 4) |

## Usage

```bash
# Run latest analysis
python analysis/v18/analysis_v18.py

# Run specific version
python analysis/v15/analysis_v15.py

# Run shared script
python analysis/shared/analysis_embeddings.py
```

## Data Sources

- **Enriched data:** `extraction/results/enriched/` (1,581 JSON files)
- **Ground truth:** `data/ground_truth/ground_truth.xlsx`

## Output

Each version's outputs (CSV, PNG, HTML) are saved in the same folder as the script.
