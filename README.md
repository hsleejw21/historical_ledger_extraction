# Historical Ledger Extraction Pipeline

A multi-agent pipeline for extracting structured data from scans of 18th–19th century English parish ledgers. Three specialised agents work in sequence: a **Structurer** maps the page layout, an **Extractor** reads the numbers, and a **Corrector** audits and fixes errors using chain-of-thought reasoning.

The experiment harness benchmarks every agent role across multiple LLM providers (Google Gemini, OpenAI GPT, Anthropic Claude) to find the most robust model for each stage.

---

## Architecture

```
Image
  │
  ▼
┌─────────────┐     skeleton JSON      ┌─────────────┐    filled JSON     ┌─────────────┐
│  Agent 1    │  ─────────────────►    │  Agent 2    │ ─────────────────► │  Agent 3    │
│ Structurer  │                         │  Extractor  │                    │  Corrector  │
└─────────────┘                         └─────────────┘                    └─────────────┘
  • Layout classify (standard/complex)    • Column-position OCR              • Chain-of-thought audit
  • Row count + type skeleton             • £ / s / d + fraction fill        • Rule-violation fixes
  • Description transcription             • Confidence scoring               • Visual re-verification
```

**Why three agents?**

The hardest failure modes on these documents are *structural* (missing rows, wrong row types) and *numerical* (misaligned columns). Separating these concerns lets each agent focus on one hard problem with a prompt tailored to it. The Structurer's skeleton acts as a binding contract — the Extractor cannot skip or merge rows that the Structurer already identified.

---

## Project Layout

```
historical-ledger-extraction/
├── src/
│   ├── config.py                  # Model registry, agent role assignments, paths
│   ├── clients.py                 # Unified LLM client (OpenAI / Google / Anthropic)
│   ├── agents/
│   │   ├── structurer.py          # Agent 1
│   │   ├── extractor.py           # Agent 2
│   │   └── corrector.py           # Agent 3
│   ├── prompts/
│   │   ├── structurer.py          # All prompts for Agent 1
│   │   ├── extractor.py           # All prompts for Agent 2
│   │   └── corrector.py           # All prompts for Agent 3
│   └── evaluation/
│       ├── scorer.py              # Two-axis scoring (structure + numbers)
│       └── gt_converter.py        # Converts ground_truth.xlsx → per-page JSON
├── data/
│   ├── images/                    # Ledger page scans (PNG/JPG)
│   └── ground_truth/              # ground_truth.xlsx + exported per-page JSONs
├── experiments/
│   ├── run_experiment.py          # Main benchmark runner
│   ├── results/                   # Cached per-stage outputs (gitignored)
│   └── reports/                   # Generated CSV reports (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Clone and create a virtual environment
git clone <your-repo-url>
cd historical-ledger-extraction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with your API keys (never committed)
# .env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=anthropic-...

# 4. Place your ledger images in data/images/
# Naming convention: <sheet_name>_image.png
# Example: 1889_4_image.png  (matches sheet "1889_4" in ground_truth.xlsx)

# 5. Export ground truth JSONs (run once, or re-run after editing the xlsx)
python -m src.evaluation.gt_converter
```

---

## Running Experiments

```bash
# Full benchmark: all pages × all model combinations
python -m experiments.run_experiment

# Subset of pages only
python -m experiments.run_experiment --pages 1889_4 1873_5 1881_1

# Isolate a single stage (useful for fast iteration on prompts)
python -m experiments.run_experiment --stage structurer
python -m experiments.run_experiment --stage extractor
```

Results are cached in `experiments/results/` — if you re-run, completed stages are loaded from disk instead of hitting the API again. Delete that folder to force a fresh run.

The final comparison report is written to `experiments/reports/experiment_results.csv`.

---

## Evaluation: Two-Axis Scoring

Every output is scored on two independent axes, each producing a value in [0, 1].

### Axis 1 — Structural Accuracy
| Sub-score | What it measures |
|---|---|
| Row count score | How close the predicted row count is to ground truth |
| Type count score | Per-type (header/entry/total) count accuracy |
| Header text score | Fuzzy match (Levenshtein ≥ 0.8) on header descriptions |

### Axis 2 — Numerical Accuracy *(entry + total rows only)*
| Sub-score | What it measures |
|---|---|
| Amount match score | Fraction of GT rows whose £/s/d triplet is matched exactly |
| Fraction match score | Fraction of GT rows with fractions (ob/q/3q) matched exactly |

The **combined score** is the simple average of Axis 1 and Axis 2.

Entry rows with no amounts (sub-items in braced groups) are excluded from Axis 2 scoring but still counted in Axis 1.

---

## Adding a New Model

1. Add it to the `MODELS` registry in `src/config.py`.
2. Add the model key to the appropriate role list(s) in `AGENT_ROLES`.
3. Run the experiment — no other changes needed.

## Adjusting Which Models Run for Each Role

Edit `AGENT_ROLES` in `src/config.py`:

```python
AGENT_ROLES = {
    "structurer": ["gemini-flash", "claude-haiku"],   # cheap, fast models
    "extractor":  ["claude-sonnet", "gpt-4.1-mini"],  # accuracy-focused
    "corrector":  ["claude-sonnet"],                   # strong reasoning
}
```
