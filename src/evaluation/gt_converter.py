"""
src/evaluation/gt_converter.py
Converts the master ground_truth.xlsx (one sheet per page) into individual
JSON files that the scorer can load directly.

Run once after updating the ground truth:
    python -m src.evaluation.gt_converter
"""
import os
import json
import pandas as pd
from ..config import GT_DIR, BASE_DIR


# The master xlsx lives at data/ground_truth/ground_truth.xlsx
MASTER_XLSX = os.path.join(GT_DIR, "ground_truth.xlsx")


def _clean_val(val):
    """Normalise a cell: NaN / None / empty → empty string; floats that are ints → int."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return ""
    s = str(val).strip()
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return s


def _clean_text(val):
    """Strip and remove surrounding double-quotes from description text."""
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s


def convert_sheet(df: pd.DataFrame, sheet_name: str) -> dict:
    """Convert a single DataFrame (one sheet) to the standard JSON schema."""
    rows = []
    for _, row in df.iterrows():
        # Skip completely empty rows (the NaN rows in 1881_1 etc.)
        if pd.isna(row.get("row_index")) and pd.isna(row.get("row_type")):
            continue

        entry = {
            "row_index":             _clean_val(row.get("row_index", "")),
            "row_type":              str(row.get("row_type", "entry")).strip().lower(),
            "description":           _clean_text(row.get("description", "")),
            "amount_pounds":         _clean_val(row.get("amount_pounds", "")),
            "amount_shillings":      _clean_val(row.get("amount_shillings", "")),
            "amount_pence_whole":    _clean_val(row.get("amount_pence_whole", "")),
            "amount_pence_fraction": _clean_val(row.get("amount_pence_fraction", "")),
        }

        # Include "side" only if the column exists AND this row has a value
        if "side" in df.columns:
            side_val = _clean_text(row.get("side", ""))
            if side_val:
                entry["side"] = side_val

        rows.append(entry)

    return {"rows": rows}


def run_conversion():
    """Convert all sheets in the master xlsx to individual JSON files."""
    if not os.path.exists(MASTER_XLSX):
        print(f"[Error] Master xlsx not found at {MASTER_XLSX}")
        return

    xls = pd.ExcelFile(MASTER_XLSX)
    os.makedirs(GT_DIR, exist_ok=True)

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        gt_data = convert_sheet(df, sheet_name)

        # Output path: data/ground_truth/<sheet_name>.json
        out_path = os.path.join(GT_DIR, f"{sheet_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False)

        print(f"  [OK] {sheet_name}.json  ({len(gt_data['rows'])} rows)")

    print(f"\nConverted {len(xls.sheet_names)} sheets → {GT_DIR}")


if __name__ == "__main__":
    run_conversion()
