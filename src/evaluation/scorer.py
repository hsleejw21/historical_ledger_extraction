"""
src/evaluation/scorer.py
Two-axis evaluation framework.

AXIS 1 — Structural Accuracy
  • Row count match
  • Per-type count match (header / entry / total)
  • Header text match (fuzzy, Levenshtein ≥ 0.8)

AXIS 2 — Numerical Accuracy  (only on entry + total rows)
  • Per-row exact £/s/d triplet match  (bag-of-rows, greedy)
  • Per-row fraction match
  • Weighted composite score

Both axes return a score in [0, 1].  The final combined score is the average
of the two axes (can be adjusted later if one axis needs more weight).
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
    """Reads a single-sheet ground truth JSON file (or the per-sheet export)."""
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_val(v) -> str:
    """Normalise an amount value to a comparable string."""
    if v == "" or v is None:
        return ""
    s = str(v).strip()
    # Remove trailing .0 so that 15.0 == 15
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _clean_desc(d) -> str:
    """Lowercase, strip, remove surrounding quotes for fuzzy comparison."""
    s = str(d).strip().lower()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s.strip()


def _amounts_match(pred_row: dict, gt_row: dict) -> bool:
    """Exact match on the £/s/d triplet."""
    keys = ("amount_pounds", "amount_shillings", "amount_pence_whole")
    return all(_normalise_val(pred_row.get(k, "")) == _normalise_val(gt_row.get(k, "")) for k in keys)


def _fraction_match(pred_row: dict, gt_row: dict) -> bool:
    """Exact match on the fraction field."""
    return _normalise_val(pred_row.get("amount_pence_fraction", "")) == \
           _normalise_val(gt_row.get("amount_pence_fraction", ""))


def _has_any_amount(row: dict) -> bool:
    """True if at least one of the three amount fields is non-empty."""
    return any(_normalise_val(row.get(k, "")) != "" for k in
               ("amount_pounds", "amount_shillings", "amount_pence_whole"))


# ---------------------------------------------------------------------------
# AXIS 1 — Structural Accuracy
# ---------------------------------------------------------------------------
def score_structure(pred: dict, gt: dict) -> dict:
    """
    Returns:
        {
            "row_count_score":      float,   # 1.0 if counts match, else ratio
            "type_count_score":     float,   # avg match across header/entry/total counts
            "header_text_score":    float,   # fraction of GT headers matched by fuzzy text
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
        # Score = 1 - (abs difference / gt_count), clamped to [0, 1]
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
            type_scores.append(0.0)   # predicted types that don't exist in GT
        else:
            type_scores.append(max(0.0, 1.0 - abs(p - g) / g))

    type_count_score = sum(type_scores) / len(type_scores)

    # --- Header text match (fuzzy) ---
    gt_headers  = [_clean_desc(r.get("description", "")) for r in gt_rows if str(r.get("row_type", "")).lower() == "header"]
    pred_headers = [_clean_desc(r.get("description", "")) for r in pred_rows if str(r.get("row_type", "")).lower() == "header"]

    if not gt_headers:
        header_text_score = 1.0   # nothing to match
    else:
        matched = 0
        available = list(pred_headers)  # greedy matching, consume each pred header once
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
    Uses a "bag of rows" greedy approach:
      For each GT row (entry or total) that has at least one amount,
      find the best matching prediction row (amounts + type must match).

    Returns:
        {
            "amount_match_score":   float,   # fraction of GT amount-rows matched exactly
            "fraction_match_score": float,   # fraction of GT fraction-rows matched exactly
            "axis2_score":          float,   # weighted average (amounts 0.8, fractions 0.2)
        }
    """
    pred_rows = pred.get("rows", [])
    gt_rows   = gt.get("rows", [])

    # Filter to entry + total rows that have at least one amount
    scoreable_types = {"entry", "total"}
    gt_scoreable  = [r for r in gt_rows  if str(r.get("row_type","")).lower() in scoreable_types and _has_any_amount(r)]
    pred_scoreable = [r for r in pred_rows if str(r.get("row_type","")).lower() in scoreable_types]

    if not gt_scoreable:
        return {"amount_match_score": 1.0, "fraction_match_score": 1.0, "axis2_score": 1.0}

    # --- Amount match (bag-of-rows, greedy) ---
    available_indices = set(range(len(pred_scoreable)))
    amount_matches = 0

    for gt_row in gt_scoreable:
        gt_type = str(gt_row.get("row_type", "")).lower()
        best_idx = -1

        for p_idx in available_indices:
            p_row = pred_scoreable[p_idx]
            p_type = str(p_row.get("row_type", "")).lower()
            if p_type == gt_type and _amounts_match(p_row, gt_row):
                best_idx = p_idx
                break   # first exact match wins (greedy)

        if best_idx != -1:
            amount_matches += 1
            available_indices.remove(best_idx)

    amount_match_score = amount_matches / len(gt_scoreable)

    # --- Fraction match ---
    # Only score rows in GT that actually have a fraction value
    gt_with_fraction = [r for r in gt_scoreable if _normalise_val(r.get("amount_pence_fraction", "")) != ""]

    if not gt_with_fraction:
        fraction_match_score = 1.0
    else:
        available_indices_f = set(range(len(pred_scoreable)))
        fraction_matches = 0

        for gt_row in gt_with_fraction:
            gt_type = str(gt_row.get("row_type", "")).lower()
            best_idx = -1

            for p_idx in available_indices_f:
                p_row = pred_scoreable[p_idx]
                p_type = str(p_row.get("row_type", "")).lower()
                # Match on type + amounts + fraction
                if p_type == gt_type and _amounts_match(p_row, gt_row) and _fraction_match(p_row, gt_row):
                    best_idx = p_idx
                    break

            if best_idx != -1:
                fraction_matches += 1
                available_indices_f.remove(best_idx)

        fraction_match_score = fraction_matches / len(gt_with_fraction)

    # Weighted composite: amounts matter more than fractions
    axis2 = 0.8 * amount_match_score + 0.2 * fraction_match_score

    return {
        "amount_match_score":   round(amount_match_score, 4),
        "fraction_match_score": round(fraction_match_score, 4),
        "axis2_score":          round(axis2, 4),
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
