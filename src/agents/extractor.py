"""
src/agents/extractor.py
Agent 2 — Extractor.

Receives the skeleton from Agent 1 and fills in the monetary amounts.
The skeleton rows (index, type, description, side) are treated as an
immutable contract — the Extractor copies them verbatim and only adds
amount_pounds, amount_shillings, amount_pence_whole, amount_pence_fraction,
and confidence_score.
"""
import json
from ..clients import call_llm, parse_json_output
from ..config import MODELS
from ..prompts.extractor import EXTRACTOR_SYSTEM, EXTRACTOR_USER_TEMPLATE


class Extractor:
    def __init__(self, model_key: str):
        self.cfg = MODELS[model_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, image_path: str, skeleton: dict) -> dict:
        """
        Args:
            image_path : path to the ledger scan.
            skeleton   : the full skeleton dict from the Structurer
                         (must contain "rows" list).

        Returns:
            {
                "rows": [ { full row with amounts filled in } , ... ],
                "_meta": { "model": ... }
            }
        """
        skeleton_json = json.dumps(skeleton, indent=2, ensure_ascii=False)

        user_prompt = EXTRACTOR_USER_TEMPLATE.format(skeleton_json=skeleton_json)

        raw = call_llm(
            provider=self.cfg["provider"],
            model_name=self.cfg["model_name"],
            system_prompt=EXTRACTOR_SYSTEM,
            user_prompt=user_prompt,
            image_path=image_path,
        )

        extraction = parse_json_output(raw)

        # Safety: if parse failed, return skeleton rows with empty amounts
        if "rows" not in extraction or not extraction["rows"]:
            extraction = {"rows": self._fallback_rows(skeleton)}

        # Post-processing: enforce schema on every row
        extraction["rows"] = [self._sanitise_row(r) for r in extraction["rows"]]

        extraction["_meta"] = {"model": self.cfg["model_name"]}
        return extraction

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitise_row(row: dict) -> dict:
        """Ensure every row has all required keys with valid types."""
        AMOUNT_KEYS = ("amount_pounds", "amount_shillings", "amount_pence_whole")
        VALID_FRACTIONS = {0.25, 0.5, 0.75}

        sanitised = {
            "row_index":              row.get("row_index", 0),
            "row_type":               row.get("row_type", "entry"),
            "description":            row.get("description", ""),
            "amount_pounds":          "",
            "amount_shillings":       "",
            "amount_pence_whole":     "",
            "amount_pence_fraction":  "",
            "confidence_score":       row.get("confidence_score", 0.5),
        }

        # Copy side only if present
        if "side" in row:
            sanitised["side"] = row["side"]

        # Normalise amount fields: convert to int where possible, keep "" for blanks
        for key in AMOUNT_KEYS:
            val = row.get(key, "")
            if val == "" or val is None:
                sanitised[key] = ""
            else:
                try:
                    f = float(val)
                    sanitised[key] = int(f) if f == int(f) else f
                except (ValueError, TypeError):
                    sanitised[key] = ""

        # Normalise fraction
        frac = row.get("amount_pence_fraction", "")
        if frac == "" or frac is None:
            sanitised["amount_pence_fraction"] = ""
        else:
            try:
                frac_f = float(frac)
                sanitised["amount_pence_fraction"] = frac_f if frac_f in VALID_FRACTIONS else ""
            except (ValueError, TypeError):
                sanitised["amount_pence_fraction"] = ""

        # Clamp confidence to [0, 1]
        try:
            c = float(sanitised["confidence_score"])
            sanitised["confidence_score"] = max(0.0, min(1.0, c))
        except (ValueError, TypeError):
            sanitised["confidence_score"] = 0.5

        return sanitised

    @staticmethod
    def _fallback_rows(skeleton: dict) -> list:
        """If the LLM returned nothing useful, produce rows with empty amounts."""
        rows = []
        for r in skeleton.get("rows", []):
            row = {
                "row_index":              r.get("row_index", 0),
                "row_type":               r.get("row_type", "entry"),
                "description":            r.get("description", ""),
                "amount_pounds":          "",
                "amount_shillings":       "",
                "amount_pence_whole":     "",
                "amount_pence_fraction":  "",
                "confidence_score":       0.0,
            }
            if "side" in r:
                row["side"] = r["side"]
            rows.append(row)
        return rows
