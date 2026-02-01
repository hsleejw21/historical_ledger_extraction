"""
src/prompts/structurer.py
All prompts for Agent 1 — the Structurer.

The Structurer's job is narrow: look at the image and output a skeleton.
It does NOT extract numbers.  It decides layout type and produces a
row-by-row template (index, type, description) that the Extractor fills in.
"""

# ---------------------------------------------------------------------------
# Layout classifier  (runs first, on a cheap model)
# ---------------------------------------------------------------------------
LAYOUT_CLASSIFIER_SYSTEM = """You are a layout classifier for historical ledger page images.
Examine the image and determine the layout type.

Output ONLY a JSON object — no other text:
{"layout_type": "standard"}   — single table, one set of £ s d columns
{"layout_type": "complex"}    — dual-column balance sheet (left/right sides with separate £ s d columns each)
"""

LAYOUT_CLASSIFIER_USER = "Classify this ledger page layout."

# ---------------------------------------------------------------------------
# Structurer system prompt  (shared across standard and complex)
# ---------------------------------------------------------------------------
_SHARED_RULES = """
ROW CLASSIFICATION — You MUST classify each row into exactly one of these types:

- "header"  : A structural label with NO monetary amounts.  This includes the
              page title at the very top AND internal section labels / place names
              that act as category anchors for the entries below them.
- "entry"   : A transaction row.  It has a description and MAY have amounts in
              the £/s/d columns.  Some entries intentionally have no amounts
              (e.g. sub-items grouped under a braced parent) — still classify
              them as "entry", not "header".
- "total"   : Any row whose amount is the result of a calculation — subtotals,
              "Summa" lines, "Deduct" lines, balances carried forward, final
              page totals.

CRITICAL DISTINCTION between "header" and "entry" with no amounts:
  - If the row is a structural LABEL that introduces a group (place name,
    account name, section title) → "header"
  - If the row is a specific TRANSACTION that just happens to have its amount
    on a sibling row (braced group) → "entry"

OUTPUT FORMAT:
Return ONLY a JSON object with this exact structure — no commentary:
{
  "layout_type": "<standard|complex>",
  "estimated_total_rows": <integer>,
  "rows": [
    {
      "row_index": <integer, starting from 1>,
      "row_type": "<header|entry|total>",
      "description": "<text exactly as it appears on the page>",
      "side": "<left|right|center>"   // ONLY include this key for complex pages
    },
    ...
  ]
}

RULES:
- row_index starts at 1 and increments sequentially — no gaps.
- Do NOT include any amounts — that is the Extractor's job.
- Capture EVERY visible row on the page.  Missing rows is the #1 failure mode.
- Preserve original spelling exactly as written (including Latin abbreviations).
- For complex (dual-column) pages, use the "side" field:
    "left"   = receipts / income side
    "right"  = payments / expenditure side
    "center" = page title or account name spanning both columns
- For standard pages, omit the "side" field entirely.
"""

STRUCTURER_SYSTEM_STANDARD = f"""You are Agent 1 (Structurer) in a historical ledger extraction pipeline.
Your task: examine this standard single-table ledger page and produce a structural skeleton.
You do NOT extract any monetary amounts — only the structure.

{_SHARED_RULES}

READING ORDER for standard pages:
Read top to bottom, exactly as the rows appear on the page.
"""

STRUCTURER_SYSTEM_COMPLEX = f"""You are Agent 1 (Structurer) in a historical ledger extraction pipeline.
Your task: examine this complex dual-column ledger page and produce a structural skeleton.
You do NOT extract any monetary amounts — only the structure.

{_SHARED_RULES}

READING ORDER for complex (dual-column) pages — use the "Z" pattern:
Process the page section by section (top to bottom).
Within each section:
  1. Extract the centered account/fund header first  (side = "center")
  2. Extract ALL left-side rows next                  (side = "left")
  3. Extract ALL right-side rows next                 (side = "right")
Then move to the next section.

Do NOT read straight across the page — that will interleave left and right entries.
"""

STRUCTURER_USER = "Produce the structural skeleton for this ledger page."
