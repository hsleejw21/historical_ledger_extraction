"""
experiments/enrich_supervisor_rows.py

Reads every *_supervisor_gemini-flash.json from experiments/results/cache/,
sends the entry rows to gpt-5-mini (via OpenAI API) and asks it to add:

  direction          – income | expenditure | transfer | balance_sheet | unclear
  category           – land_rent | ecclesiastical | maintenance | salary_stipend |
                       administrative | educational | financial | domestic |
                       charitable | other
  language           – latin | english | mixed
  english_description             – plain English gloss, no monetary amounts
  english_description_with_amount – same gloss including the £/s/d value in prose
  place_name         – normalised place / property name or null
  person_name        – normalised person name (tenant / payee) or null
  payment_period     – half_year | annual | sesquiannual | biennial | triennial |
                       quadrennial | quinquennial | multi_year | one_off | unclear
  is_signature       – true | false
  is_arrears         – true | false

Results are written per page to experiments/results/enriched/
  e.g. experiments/results/enriched/1700_1_image_enriched.json

Usage:
    python -m experiments.enrich_supervisor_rows              # all pages
    python -m experiments.enrich_supervisor_rows --pages 1700_1 1735_3
    python -m experiments.enrich_supervisor_rows --dry-run    # print first batch, no API call
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CACHE_DIR  = ROOT / "experiments" / "results" / "cache"
OUTPUT_DIR = ROOT / "experiments" / "results" / "enriched"

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert in historical Oxford college financial ledgers (1700–1900).
Entries span Latin, abbreviated English, and mixed language across 200 years.

--- LATIN GLOSSARY ---
The glossary below covers the most frequent terms. It is NOT exhaustive. For any Latin word or
phrase not listed here, use your own knowledge of medieval and early-modern Latin to translate and
classify it. When confident, apply the appropriate field values; when genuinely uncertain, use
"unclear" / "mixed" as appropriate — do not default to null.

Time periods:
  pro anno / pro an / pro ann           = for the year (annual)
  pro dimidio / pro dim / dimid         = for half a year
  pro sesquianno                        = for 1.5 years
  pro biennio / pro bien / pro bien     = for 2 years
  pro triennio / pro trien              = for 3 years
  pro quadriennio                       = for 4 years
  pro quinquennio / pro quing           = for 5 years
  pro [N] annis                         = for N years (multi-year arrears)

Feast-day calendar markers (indicates ecclesiastical time system):
  ad festum Michaelis / mich / michs    = at Michaelmas (29 Sep)
  ad festum Annunciationis / annun      = at Lady Day / Annunciation (25 Mar)
  ad festum Nativitatis / natio / xmas  = at Christmas (25 Dec)
  ad festum Jo. Bapt. / mids            = at Midsummer / St John Baptist (24 Jun)

Section header meanings (use to infer direction):
  Recepta / Arrearagia recepta          = receipts / income (→ direction: income)
  Soluta / Soluta Ordinaria / Soluta Varia = payments made (→ direction: expenditure)
  Stipendia                             = stipend payments (→ direction: expenditure)
  Arrearagia / Arrearagia recepta       = arrears section (→ is_arrears: true; direction usually income)
  Dr / Cr columns on a balance-sheet page → direction: balance_sheet

Named trust fund accounts (post-1880 section headers):
  SYMES, KING CHARLES I, HOW, MICHELL, GIFFORD = specific endowment trust funds held by the college.
  Entries under these headers are financial trust transactions (→ category: financial).

EARLY-ERA NOTE (pre-1760): Amounts were sometimes embedded directly in the description field
  rather than in separate amount columns.  E.g. "0.12.6 Ralph Marshall Cayons pro Quinquennio"
  means 12s 6d paid to Ralph Marshall for capons, for 5 years.  Extract person/place/period
  from the description even when the amount columns are null.

"Progress" in a description = the Rector/Bursar's estate inspection tour (administrative travel).
  → category: administrative, direction: expenditure.

Common Latin entry terms:
  Janitori       = to the janitor (wages)
  Coquo          = to the cook (wages)
  Lotrici        = to the laundry worker (wages)
  Tonsori        = to the barber (wages)
  Bibliothecario = to the librarian (wages)
  Concionatori   = to the preacher / college preacher (wages → category: salary_stipend or ecclesiastical)
  Pauperibus     = to the poor (alms)
  Procurationes  = procuration dues (paid to bishop on visitation)
  Synodalls      = synodal dues (paid to bishop at synod)
  Augmentatio / Augm = augmentation (top-up grant for underfunded church livings → income)
  Rect / Rectori / Rectory = rector / rectory (property or person)
  Arrearagia     = arrears (overdue payments)
  Eidem / Eisd   = "to the same person" — Latin ditto for the person reference; resolve from the row above
  Impensa        = expenses / costs
  Dô / Do. / do / D.o / " " = ditto (same as the entry above — resolve from context).
                  Late-era pages (post-1850) often use " " (repeated quotes) as the ditto mark.
  Recepta        = received / receipts
  Soluta         = paid / payments

Additional calendar terms:
  Pasch / Paschal = Easter; Pentecost = 7 weeks after Easter. Feast-day payments (e.g. a vicar
  paid twice a year at Lady Day and Michaelmas) are typically half-yearly each → payment_period: "half_year".

Inter-collegiate payments — entries referencing other Oxford colleges (All Souls, Balliol, Christ
  Church, etc.) are inter-college fee payments → category: administrative.

College-specific income terms:
  "Plate Money"    = fee from new fellows toward the silver plate fund → administrative, income
  "Benefaction"    = endowment/legacy gift → financial, income
  "Increm: Donum"  = increment / common fund contribution → financial, income

Trust funds — any named individual or family name appearing as an entry under a "Trust Funds"
  section header is a named endowment distribution → category: financial, direction: income.

NOTE — amounts embedded in descriptions: some rows include the monetary value inside the description
  text (e.g. "Vicario Mert. Fest Ann. 31.16.5"). Treat the structured amount_pounds/shillings/pence
  fields as authoritative; the embedded amount is contextual information only.

The `section_header` field = most recent page header above this entry. Use it to infer direction.
The `side` field (when present): "left" = income column, "right" = expenditure column.

Return ONLY a valid JSON object with a single key "rows" — no prose, no markdown fences.
"""

USER_PROMPT_TEMPLATE = """You are enriching rows from a historical Oxford college ledger page.

Return a JSON object: {{"rows": [ ... ]}}
Include EVERY input row in the output array (same order, same row_index).
For header and total rows: copy row_index and row_type, set ALL enrichment fields to null.
For entry rows: populate all enrichment fields as described below.

=== FIELD DEFINITIONS ===

--- direction ---
Which way does money flow FROM THE COLLEGE's perspective?
  "income"        – college RECEIVES money (rents received, dividends, fees collected, arrears paid to college)
  "expenditure"   – college PAYS money out (wages, repairs, dues paid, charitable gifts)
  "transfer"      – internal movement between college accounts (e.g. balance carried forward)
  "balance_sheet" – entry appears on a balance-sheet or reconciliation page (Dr/Cr format, stock
                    valuations, caution-money deposits, trust-fund balances). These are asset/liability
                    statements, not income/expenditure transactions.
  "unclear"       – genuinely cannot determine from context
Hint: section_header "Recepta" → income; "Soluta*" or "Stipendia" → expenditure;
      Dr/Cr columns or "Balance Sheet" page title → balance_sheet.

--- category ---
What is this entry about? Choose the SINGLE best fit:

  "land_rent"      – income or expenditure related to land, farms, closes, tenements, rectories,
                     mills, ferries held or leased by the college.
                     Examples: "Kidlington pro anno", "Clifton Ferry pro bien", "Chase Hill 4 yrs",
                     "Lands let at Rackrent", "Quit Rents"

  "ecclesiastical" – church dues, tithes, procurations, synodals, augmentations, curate/vicar
                     payments, bishop visitation fees, feast-day payments, Easter offerings,
                     consecration fees, chapel purchases (Bibles, prayer books).
                     Examples: "Procurationes Cudl. Southn. Merton", "Synodalls", "Dño Episcopo
                     in visitatione Sua trien.", "Tenths of Merton", "Augment. S. Newington",
                     "Easter offering", "Consecration Sermon", "pd Mr Fletcher for a Bible for ye Chappel"
                     NOTE: "Diocesan Board of Education" → ecclesiastical (not educational); it is a
                     church governance body, not a student education fee.

  "maintenance"    – physical upkeep of buildings or property: repairs, building works, plumbing,
                     painting, draining, thatching, timber, tiles, walls, pipes, paving, glazing.
                     Examples: "Hewitt Pipes", "Brick Wall", "Thatching", "Draining Labor",
                     "pd for mending a pump", "Holship for painting the Hall"

  "salary_stipend" – wages, salaries, or stipends paid to named individuals or roles: servants,
                     college officers, scholars, rector, fellows, porter, gardener, cook, barber,
                     laundress, librarian, janitor, clerk, scullion.
                     Examples: "Janitori", "Lotrici", "Tonsori", "Porter Salary", "Duo Rectori",
                     "Stipend ad fest.", "Gardeners Wages"

  "administrative" – legal, professional, and governance costs: legal fees, stamps, agency fees,
                     management charges, audit costs, printing, stationery, account books,
                     city/court payments, visitation administration.
                     Examples: "Stamps", "Farrer copy of Opinion", "Agency, Management",
                     "Curia Hoystings", "pd for a new Acct Book", "pd Benj: Burroughs for
                     transcribing the Statutes"

  "educational"    – scholarships, exhibitions, prizes, tuition, battels (student board charges),
                     room rents from students, library books, subscriptions to academic works.
                     Examples: "Exhibitions", "Tuition", "Battels", "Room Rents", "Sr Richards
                     Divinity Prize", "Deans Fees", "pd Cockman a subscription to ... Council of Trent"

  "financial"      – investment income or management: funds, stocks, consols, dividends, interest,
                     annuities, loans, mortgages, bonds. Also building funds or endowment funds.
                     Examples: "Fund", "Dividends", "2000 Consols", "New Building Fund",
                     "1 yr Interest on £247", "Annuity", "Repayment from Choir Fund"

  "domestic"       – day-to-day running of the college household: food, drink, fuel, coal, wine,
                     candles, kitchen supplies, venison, corn prices, domestic provisions.
                     Examples: "Venison for the 30th of June", "pd Minshall two bills for Wine",
                     "pd Alex Glifford upon acct of Fuel", "Corne prices", "Mayor & capons"

  "charitable"     – payments to the poor, relief funds, alms, and subscriptions to external
                     charitable institutions (infirmaries, schools, clergy relief funds).
                     Examples: "Pauperibus", "Bedle of Beggars", "Infirmary", "Grey Coat School",
                     "Widows & Orphans of Clergy"

  "other"          – use ONLY when the entry genuinely does not fit any category above.

NOTE: "Poor Rate" / "Land Tax" / "Rate" entries can appear as income (collected from tenants and
  remitted) or expenditure (paid out by the college). Use section_header and side to determine
  direction; category is "administrative" in both cases.

--- language ---
What language is the description written in?
  "latin"   – entirely or predominantly Latin (including standard Latin abbreviations)
  "english" – entirely or predominantly English (including archaic English spelling)
  "mixed"   – meaningful mix of both (e.g. Latin phrase embedded in English sentence, or
               Latin abbreviations mixed with English proper nouns)

--- english_description ---
A concise plain English gloss of WHAT this entry is about — do NOT mention any monetary amounts.
- Translate Latin; expand abbreviations; identify the role/place/person where possible.
- For ditto rows (Dô / Do. / do / D.o / " " at start or as full description): resolve what it
  refers to from surrounding rows; write the resolved meaning, not "ditto".
- For "Eidem" rows: resolve the person from the row above; write the full meaning.
- For braced group rows (entry with `{{` brace or no amounts, heading a group): describe the group.
- If the description appears to be an OCR or transcription artifact (garbled, single letter,
  obviously corrupt text) write: "Unclear — possible transcription error."
- Do NOT copy the raw description verbatim — actually interpret it.
- Do NOT include any £/s/d figures.
Example: "Annual rent received from the Kidlington estate."

--- english_description_with_amount ---
Same as english_description but also incorporates the monetary value in natural language.
Use the structured amount fields (amount_pounds, amount_shillings, amount_pence_whole,
amount_pence_fraction) to state the sum. Write it as natural prose, e.g. "£4 10s 6d".
If amount fields are all null or empty, fall back to any amount embedded in the description text.
Example: "Annual rent received from the Kidlington estate, amounting to £12 0s 0d."

--- place_name ---
If the entry references a specific property, parish, farm, rectory, ferry, mill, or
geographic location, give the normalised full name (string). Otherwise null.
Normalise variants: "Cudl." / "Cuddlington" / "Cuddington" → "Cuddington"
Multiple places: use semicolons, e.g. "Cuddington; South Newington; Merton"

--- person_name ---
If the entry references a specific named individual (tenant, payee, contractor, officer),
give their normalised name (string). Otherwise null.
Use the role if no personal name is given: e.g. "Janitor", "Cook", "Laundress".
Do NOT use generic role names like "contractor" or "unknown" — use null if truly unclear.

--- payment_period ---
How long does this payment cover?
  "half_year"     – pro dimidio / dim / dimid / ½ yr / half year
  "annual"        – pro anno / pro an / 1 yr / per year / yearly
  "sesquiannual"  – pro sesquianno / 1½ years
  "biennial"      – pro biennio / pro bien / 2 yrs
  "triennial"     – pro triennio / pro trien / 3 yrs
  "quadrennial"   – pro quadriennio / 4 yrs
  "quinquennial"  – pro quinquennio / 5 yrs
  "multi_year"    – pro N annis where N > 5, or explicitly many years / arrears
  "one_off"       – a single payment with no recurring period (purchase, repair bill, fine)
  "unclear"       – cannot determine

--- is_signature ---
true ONLY if this row is clearly an auditor/witness signature from the page approval section
(e.g. "per nos", a bare personal name with no amount at the very end of a ledger page).
false for all other entries, including named payees with amounts.

--- is_arrears ---
true if this entry represents an overdue / late payment being collected after it was originally due.
Indicators:
  - section_header contains "Arrearagia" or "Arrears"
  - description contains "in arrear", "arrearagia", "arrears of"
  - a payment explicitly labelled as being for a prior year (e.g. "rent due 1843 pd 1845")
false for all other entries (current-year payments, one-off purchases, etc.).

=== RULES ===
1. Every input row must appear in the output with its row_index preserved.
2. header and total rows: all enrichment fields = null.
3. Ditto rows: resolve from context; do not output "ditto" as english_description.
4. When direction is clear from section_header or side field, always use it.
5. Use "unclear" (not null) for direction/category/payment_period when the entry exists
   but you cannot classify it — null is reserved for non-entry rows only.
6. is_arrears and is_signature default to false (not null) for entry rows.

Return format: {{"rows": [ {{...}}, {{...}} ]}}

Input rows:
{rows_json}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY ")
    if not key:
        sys.exit("OPENAI_API_KEY not found in .env")
    return key.strip()


def get_supervisor_files(page_filter: list[str] | None = None) -> list[Path]:
    pattern = "*_supervisor_gemini-flash.json"
    files = sorted(CACHE_DIR.glob(pattern))
    if page_filter:
        # e.g. "1700_1" matches "1700_1_image_supervisor_gemini-flash.json" exactly
        files = [f for f in files if any(f.name.startswith(p + "_image") for p in page_filter)]
    return files


def propagate_section_headers(rows: list[dict]) -> list[dict]:
    """Add section_header field to each row by carrying forward the last header."""
    current_header = ""
    result = []
    for row in rows:
        if row.get("row_type") == "header":
            desc = (row.get("description") or "").strip()
            if desc:
                current_header = desc
        row = dict(row)
        row["section_header"] = current_header
        result.append(row)
    return result


def strip_for_llm(row: dict) -> dict:
    """Keep only fields the LLM needs; drop notes (large, irrelevant)."""
    keep = {"row_index", "row_type", "description", "amount_pounds",
            "amount_shillings", "amount_pence_whole", "amount_pence_fraction",
            "side", "section_header"}
    return {k: v for k, v in row.items() if k in keep}


def call_openai(client: OpenAI, rows_json: str, dry_run: bool = False) -> list[dict]:
    prompt = USER_PROMPT_TEMPLATE.format(rows_json=rows_json)
    if dry_run:
        print("\n--- DRY RUN: prompt ---")
        print(SYSTEM_PROMPT[:300] + "…")
        print(prompt[:800] + "…")
        print("--- END DRY RUN ---\n")
        return []

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        timeout=180,
    )
    text = response.choices[0].message.content.strip()

    # The model returns {"rows": [...]} or just [...] — normalise
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        # find the first list value
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    return parsed


ENRICHMENT_FIELDS = [
    "direction", "category", "language", "english_description",
    "english_description_with_amount", "place_name", "person_name",
    "payment_period", "is_signature", "is_arrears",
]

VALID_VALUES: dict[str, set] = {
    "direction":      {"income", "expenditure", "transfer", "balance_sheet", "unclear"},
    "category":       {"land_rent", "ecclesiastical", "maintenance", "salary_stipend",
                       "administrative", "educational", "financial", "domestic",
                       "charitable", "other"},
    "language":       {"latin", "english", "mixed"},
    "payment_period": {"half_year", "annual", "sesquiannual", "biennial", "triennial",
                       "quadrennial", "quinquennial", "multi_year", "one_off", "unclear"},
}
FALLBACK: dict[str, Any] = {
    "direction": "unclear",
    "category":  "other",
    "language":  "mixed",
    "payment_period": "unclear",
}


def validate_row(row: dict, page_id: str) -> dict:
    """
    Coerce each enrichment field to a valid value.
    - Enum fields: replace out-of-vocab values with the fallback.
    - is_signature: cast to bool.
    - String fields (english_description, place_name, person_name): ensure str or None.
    - For header/total rows all enrichment fields should be None — enforce that.
    """
    row = dict(row)
    is_entry = row.get("row_type") == "entry"

    for field in ENRICHMENT_FIELDS:
        val = row.get(field)

        # Non-entry rows: force null
        if not is_entry:
            row[field] = None
            continue

        # Enum validation
        if field in VALID_VALUES:
            if val not in VALID_VALUES[field]:
                if val is not None:
                    print(f"  [coerce] {page_id} row {row.get('row_index')}: "
                          f"{field}={val!r} → {FALLBACK[field]!r}")
                row[field] = FALLBACK[field]

        # Boolean fields
        elif field in {"is_signature", "is_arrears"}:
            if not isinstance(val, bool):
                row[field] = bool(val) if val is not None else False

        # Free-text strings — accept str or None
        elif field in {"english_description", "english_description_with_amount", "place_name", "person_name"}:
            if val is not None and not isinstance(val, str):
                row[field] = str(val)
            elif val == "":
                row[field] = None

    return row


def merge_enrichment(original_rows: list[dict], enriched: list[dict], page_id: str = "") -> list[dict]:
    """Merge enrichment fields from LLM output back onto original rows by row_index."""
    enriched_by_idx = {r["row_index"]: r for r in enriched if "row_index" in r}
    result = []
    for row in original_rows:
        row = dict(row)
        idx = row.get("row_index")
        extra = enriched_by_idx.get(idx, {})
        for field in ENRICHMENT_FIELDS:
            row[field] = extra.get(field, None)
        row = validate_row(row, page_id)
        result.append(row)
    return result


def process_file(
    sup_file: Path,
    client: OpenAI,
    output_dir: Path,
    batch_size: int = 30,
    dry_run: bool = False,
) -> None:
    page_id = sup_file.name.replace("_supervisor_gemini-flash.json", "")
    out_path = output_dir / f"{page_id}_enriched.json"

    if out_path.exists():
        print(f"  [skip] {page_id} — already enriched")
        return

    with open(sup_file) as f:
        data = json.load(f)

    rows: list[dict] = data.get("rows", [])
    meta: dict       = data.get("_meta", {})

    # Pre-process: propagate section headers, drop notes
    rows = propagate_section_headers(rows)
    llm_input_rows = [strip_for_llm(r) for r in rows]

    # Batch
    all_enriched: list[dict] = []
    for i in range(0, len(llm_input_rows), batch_size):
        batch = llm_input_rows[i : i + batch_size]
        rows_json = json.dumps(batch, indent=2, ensure_ascii=False)

        try:
            enriched_batch = call_openai(client, rows_json, dry_run=dry_run)
        except Exception as e:
            print(f"  [error] {page_id} batch {i//batch_size}: {e}")
            # fill with nulls so we don't lose the page
            enriched_batch = [{"row_index": r["row_index"]} for r in batch]

        all_enriched.extend(enriched_batch)

        if dry_run:
            break  # only show first batch

        if i + batch_size < len(llm_input_rows):
            time.sleep(0.3)  # gentle rate limiting

    if dry_run:
        return

    merged = merge_enrichment(rows, all_enriched, page_id=page_id)

    output = {
        "page_id": page_id,
        "rows": merged,
        "_meta": {
            **meta,
            "enrichment_model": "gpt-5-mini",
            "enrichment_fields": ENRICHMENT_FIELDS,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  [done] {page_id} → {out_path.name}  ({len(merged)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Enrich supervisor rows with LLM semantic columns")
    parser.add_argument("--pages", nargs="+", metavar="PAGE_ID",
                        help="Only process these page IDs, e.g. 1700_1 1735_3")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Rows per LLM call (default: 30)")
    parser.add_argument("--page-sleep", type=float, default=1.0,
                        help="Seconds to sleep between pages to respect rate limits (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first batch prompt without calling API")
    args = parser.parse_args()

    api_key = load_env()
    client  = OpenAI(api_key=api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sup_files = get_supervisor_files(args.pages)
    if not sup_files:
        sys.exit(f"No supervisor files found matching {args.pages}")

    print(f"Found {len(sup_files)} supervisor file(s). Output → {OUTPUT_DIR}")

    for i, sup_file in enumerate(sup_files):
        print(f"Processing {sup_file.name} …")
        process_file(
            sup_file,
            client,
            OUTPUT_DIR,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        if not args.dry_run and i < len(sup_files) - 1:
            time.sleep(args.page_sleep)

    print("\nDone.")


if __name__ == "__main__":
    main()
