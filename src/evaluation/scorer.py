"""
src/evaluation/scorer.py
Two-axis evaluation framework.

AXIS 1 — Structural Accuracy
  • Row count match
  • Per-type count match (header / entry / total)
  • Header text match (fuzzy, Levenshtein >= 0.8)

AXIS 2 — Numerical Accuracy  (only on entry + total rows)
  • Description-aware greedy matching: when multiple pred rows share the
    same £/s/d triplet + type, the one whose description best fuzzy-matches
    the GT row is consumed first.  This prevents greedy cascade failures on
    pages with duplicate amounts.
  • Relaxed type matching: if no same-type match exists, an entry<->total
    cross-match is allowed at half credit.  This handles balanced balance
    sheets where the same amount is structurally both an entry and a total.
  • Partial-credit amount similarity: converts each row to total pence and
    scores similarity as 1 - (abs_diff / max(pred_pence, gt_pence)).  A row
    that is off by 1 penny on a L50 amount scores ~0.99 instead of 0.
  • Fraction match (exact, on rows that have fractions in GT).

Both axes return a score in [0, 1].  combined_score = (axis1 + axis2) / 2.
"""
import json


# ---------------------------------------------------------------------------
# Pure-Python Levenshtein ratio  (no external dependency needed)
# ---------------------------------------------------------------------------
def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions    = prev_row[j + 1] + 1
            deletions     = curr_row[j]     + 1
            substitutions = prev_row[j]     + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def lev_ratio(s1: str, s2: str) -> float:
    """Normalised similarity in [0, 1].  Equivalent to python-Levenshtein ratio()."""
    total_len = len(s1) + len(s2)
    if total_len == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(s1, s2) / total_len) * 2


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------
def load_ground_truth(gt_path: str) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_val(v) -> str:
    """
    Normalise an amount value to a comparable string.

    Key rule: 0 and "" are treated as equivalent.  LLMs consistently leave
    zero-valued £/s/d columns blank rather than writing 0, while ground truth
    sometimes stores explicit zeros.  Collapsing both to "" prevents false
    negatives on every row where only one or two of the three columns have
    real values.
    """
    if v == "" or v is None:
        return ""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s == "0":
        return ""
    return s


def _clean_desc(d) -> str:
    """Lowercase, strip, remove surrounding quotes for fuzzy comparison."""
    s = str(d).strip().lower()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s.strip()


def _safe_int(v) -> int:
    """Parse a value to int, returning 0 on failure."""
    if v == "" or v is None:
        return 0
    try:
        return int(float(str(v).strip()))
    except (ValueError, TypeError):
        return 0


def _to_pence(row: dict) -> int:
    """Convert a row's £/s/d to total pence.  Used for partial-credit similarity."""
    p = _safe_int(row.get("amount_pounds", ""))
    s = _safe_int(row.get("amount_shillings", ""))
    d = _safe_int(row.get("amount_pence_whole", ""))
    return (p * 240) + (s * 12) + d


def _amounts_match(pred_row: dict, gt_row: dict) -> bool:
    """Exact match on the £/s/d triplet."""
    keys = ("amount_pounds", "amount_shillings", "amount_pence_whole")
    return all(_normalise_val(pred_row.get(k, "")) == _normalise_val(gt_row.get(k, "")) for k in keys)


def _amount_similarity(pred_row: dict, gt_row: dict) -> float:
    """
    Partial-credit similarity between two rows based on total pence.
    Returns 1.0 for exact match, decays smoothly for small differences.
    A 1-penny error on a L50 row (12000 pence) scores 0.9999.
    A 1-penny error on a 5-pence row scores 0.8.
    """
    pred_pence = _to_pence(pred_row)
    gt_pence   = _to_pence(gt_row)
    if pred_pence == 0 and gt_pence == 0:
        return 1.0
    denom = max(pred_pence, gt_pence)
    return max(0.0, 1.0 - abs(pred_pence - gt_pence) / denom)


def _fraction_match(pred_row: dict, gt_row: dict) -> bool:
    """Exact match on the fraction field."""
    return _normalise_val(pred_row.get("amount_pence_fraction", "")) == \
           _normalise_val(gt_row.get("amount_pence_fraction", ""))


def _has_any_amount(row: dict) -> bool:
    """True if at least one of the three amount fields is non-empty."""
    return any(_normalise_val(row.get(k, "")) != "" for k in
               ("amount_pounds", "amount_shillings", "amount_pence_whole"))


def _desc_similarity(pred_row: dict, gt_row: dict) -> float:
    """Fuzzy similarity of the description fields.  Used as tiebreaker during matching."""
    return lev_ratio(_clean_desc(pred_row.get("description", "")),
                     _clean_desc(gt_row.get("description", "")))


# ---------------------------------------------------------------------------
# AXIS 1 — Structural Accuracy
# ---------------------------------------------------------------------------
def score_structure(pred: dict, gt: dict) -> dict:
    """
    Returns:
        {
            "row_count_score":      float,
            "type_count_score":     float,
            "header_text_score":    float,
            "axis1_score":          float,   # average of the above three
        }
    """
    pred_rows = pred.get("rows", [])
    gt_rows   = gt.get("rows", [])

    # --- Row count ---
    pred_count = len(pred_rows)
    gt_count   = len(gt_rows)
    if gt_count == 0:
        row_count_score = 1.0 if pred_count == 0 else 0.0
    else:
        row_count_score = max(0.0, 1.0 - abs(pred_count - gt_count) / gt_count)

    # --- Per-type count ---
    def type_counts(rows):
        counts = {"header": 0, "entry": 0, "total": 0}
        for r in rows:
            t = str(r.get("row_type", "")).lower().strip()
            if t in counts:
                counts[t] += 1
        return counts

    pred_tc = type_counts(pred_rows)
    gt_tc   = type_counts(gt_rows)

    type_scores = []
    for t in ("header", "entry", "total"):
        g = gt_tc[t]
        p = pred_tc[t]
        if g == 0 and p == 0:
            type_scores.append(1.0)
        elif g == 0:
            type_scores.append(0.0)
        else:
            type_scores.append(max(0.0, 1.0 - abs(p - g) / g))

    type_count_score = sum(type_scores) / len(type_scores)

    # --- Header text match (fuzzy) ---
    gt_headers   = [_clean_desc(r.get("description", "")) for r in gt_rows  if str(r.get("row_type", "")).lower() == "header"]
    pred_headers = [_clean_desc(r.get("description", "")) for r in pred_rows if str(r.get("row_type", "")).lower() == "header"]

    if not gt_headers:
        header_text_score = 1.0
    else:
        matched = 0
        available = list(pred_headers)
        for gh in gt_headers:
            best_sim = 0.0
            best_idx = -1
            for i, ph in enumerate(available):
                sim = lev_ratio(gh, ph)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_sim >= 0.8:
                matched += 1
                available.pop(best_idx)
        header_text_score = matched / len(gt_headers)

    axis1 = (row_count_score + type_count_score + header_text_score) / 3.0

    return {
        "row_count_score":   round(row_count_score, 4),
        "type_count_score":  round(type_count_score, 4),
        "header_text_score": round(header_text_score, 4),
        "axis1_score":       round(axis1, 4),
    }


# ---------------------------------------------------------------------------
# AXIS 2 — Numerical Accuracy  (entry + total rows only)
# ---------------------------------------------------------------------------
def score_numbers(pred: dict, gt: dict) -> dict:
    """
    Description-aware matching with relaxed type and partial credit.

    Matching priority for each GT row (tried in order, first hit wins):
      1. Same type + exact amounts + best description match among candidates
      2. Cross type (entry<->total) + exact amounts + best description match
         -> awarded 0.5 credit instead of 1.0
      3. No exact match found -> partial credit via amount_similarity

    Returns:
        {
            "amount_match_score":      float,   # exact-match fraction (with 0.5 for cross-type)
            "amount_similarity_score": float,   # partial-credit avg across all GT rows
            "fraction_match_score":    float,   # exact fraction match on rows that have fractions
            "axis2_score":             float,   # composite: 0.5*match + 0.3*similarity + 0.2*fraction
        }
    """
    pred_rows = pred.get("rows", [])
    gt_rows   = gt.get("rows", [])

    scoreable_types = {"entry", "total"}
    # GT: strict — only entry/total rows that actually have amounts.
    # Pred: any row that has at least one amount value.  A row the LLM
    # mislabelled "section_header" or "title" but filled with correct
    # numbers should still be eligible to match; the type-match logic
    # below already penalises the label mismatch appropriately.
    gt_scoreable   = [r for r in gt_rows  if str(r.get("row_type", "")).lower() in scoreable_types and _has_any_amount(r)]
    pred_scoreable = [r for r in pred_rows if _has_any_amount(r)]

    if not gt_scoreable:
        return {
            "amount_match_score": 1.0,
            "amount_similarity_score": 1.0,
            "fraction_match_score": 1.0,
            "axis2_score": 1.0,
        }

    # ---------------------------------------------------------------
    # MATCHING PASS
    # ---------------------------------------------------------------
    available = set(range(len(pred_scoreable)))
    match_credits = []       # 1.0 same-type exact, 0.5 cross-type exact, 0.0 miss
    similarity_scores = []   # partial credit per GT row

    CROSS_TYPE_PAIRS = {("entry", "total"), ("total", "entry")}

    for gt_row in gt_scoreable:
        gt_type = str(gt_row.get("row_type", "")).lower()

        # Collect all available pred rows with exact amount match
        exact_candidates = []   # (pred_idx, is_same_type, desc_sim)
        for p_idx in available:
            p_row = pred_scoreable[p_idx]
            if not _amounts_match(p_row, gt_row):
                continue
            p_type = str(p_row.get("row_type", "")).lower()
            is_same_type  = (p_type == gt_type)
            is_cross_type = (p_type, gt_type) in CROSS_TYPE_PAIRS
            if is_same_type or is_cross_type:
                desc_sim = _desc_similarity(p_row, gt_row)
                exact_candidates.append((p_idx, is_same_type, desc_sim))

        if exact_candidates:
            # Sort: same_type first (True > False), then best description similarity
            exact_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            best_idx, best_is_same, _ = exact_candidates[0]
            available.remove(best_idx)
            match_credits.append(1.0 if best_is_same else 0.5)
            similarity_scores.append(1.0)
        else:
            # No exact match — compute partial credit against all available pred rows
            match_credits.append(0.0)
            best_sim = 0.0
            for p_idx in available:
                sim = _amount_similarity(pred_scoreable[p_idx], gt_row)
                if sim > best_sim:
                    best_sim = sim
            similarity_scores.append(best_sim)

    amount_match_score      = sum(match_credits) / len(gt_scoreable)
    amount_similarity_score = sum(similarity_scores) / len(gt_scoreable)

    # ---------------------------------------------------------------
    # FRACTION MATCH  (exact, only on GT rows that have a fraction)
    # ---------------------------------------------------------------
    gt_with_fraction = [r for r in gt_scoreable if _normalise_val(r.get("amount_pence_fraction", "")) != ""]

    if not gt_with_fraction:
        fraction_match_score = 1.0
    else:
        available_f = set(range(len(pred_scoreable)))
        fraction_matches = 0

        for gt_row in gt_with_fraction:
            gt_type = str(gt_row.get("row_type", "")).lower()

            candidates = []
            for p_idx in available_f:
                p_row = pred_scoreable[p_idx]
                if not _amounts_match(p_row, gt_row):
                    continue
                if not _fraction_match(p_row, gt_row):
                    continue
                p_type = str(p_row.get("row_type", "")).lower()
                is_same  = (p_type == gt_type)
                is_cross = (p_type, gt_type) in CROSS_TYPE_PAIRS
                # Accept any pred row that passes amount + fraction checks.
                # Same-type matches are preferred (sorted first below) but a
                # mislabelled row should not be silently excluded.
                candidates.append((p_idx, is_same or is_cross, _desc_similarity(p_row, gt_row)))

            if candidates:
                candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                best_idx = candidates[0][0]
                available_f.remove(best_idx)
                fraction_matches += 1

        fraction_match_score = fraction_matches / len(gt_with_fraction)

    # ---------------------------------------------------------------
    # COMPOSITE axis2
    # ---------------------------------------------------------------
    axis2 = (0.5 * amount_match_score +
             0.3 * amount_similarity_score +
             0.2 * fraction_match_score)

    return {
        "amount_match_score":      round(amount_match_score, 4),
        "amount_similarity_score": round(amount_similarity_score, 4),
        "fraction_match_score":    round(fraction_match_score, 4),
        "axis2_score":             round(axis2, 4),
    }


# ---------------------------------------------------------------------------
# Combined scorer
# ---------------------------------------------------------------------------
def score_page(pred: dict, gt: dict) -> dict:
    """
    Full two-axis score for one page.
    Returns a flat dict with all sub-scores plus the final combined score.
    """
    s1 = score_structure(pred, gt)
    s2 = score_numbers(pred, gt)

    combined = (s1["axis1_score"] + s2["axis2_score"]) / 2.0

    return {
        **s1,
        **s2,
        "combined_score": round(combined, 4),
    }
