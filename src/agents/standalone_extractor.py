"""
src/agents/standalone_extractor.py
V2 Standalone Extractor.

Unlike v1's Extractor (which fills a skeleton), this agent works directly
from the image.  It discovers the layout, classifies rows, and reads amounts
all in one pass.  Every row it outputs includes a confidence_score and a
detailed "notes" field explaining its reasoning — these are consumed by the
Supervisor.
"""
import json
from ..clients import call_llm, parse_json_output
from ..config import MODELS
from ..prompts.standalone_extractor import STANDALONE_EXTRACTOR_SYSTEM, STANDALONE_EXTRACTOR_USER


class StandaloneExtractor:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.cfg = MODELS[model_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, image_path: str) -> dict:
        """
        Returns:
            {
                "layout_type": "standard" | "complex",
                "rows": [ { full row with amounts + confidence + notes } ],
                "_meta": { "model_key": ..., "model_name": ... }
            }
        """
        raw = call_llm(
            provider=self.cfg["provider"],
            model_name=self.cfg["model_name"],
            system_prompt=STANDALONE_EXTRACTOR_SYSTEM,
            user_prompt=STANDALONE_EXTRACTOR_USER,
            image_path=image_path,
        )

        data = parse_json_output(raw)

        # Safety: if the model returned nothing useful
        if "rows" not in data or not data["rows"]:
            data = {"layout_type": "standard", "rows": []}

        # Post-process every row through sanitisation
        data["rows"] = [self._sanitise_row(r) for r in data["rows"]]

        data["_meta"] = {
            "model_key":  self.model_key,
            "model_name": self.cfg["model_name"],
        }
        return data

    # ------------------------------------------------------------------
    # Sanitisation
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitise_row(row: dict) -> dict:
        """Enforce schema, normalise types, guarantee notes field exists."""
        AMOUNT_KEYS    = ("amount_pounds", "amount_shillings", "amount_pence_whole")
        VALID_FRACS    = {0.25, 0.5, 0.75}

        sanitised = {
            "row_index":             row.get("row_index", 0),
            "row_type":              str(row.get("row_type", "entry")).strip().lower(),
            "description":           str(row.get("description", "")),
            "amount_pounds":         "",
            "amount_shillings":      "",
            "amount_pence_whole":    "",
            "amount_pence_fraction": "",
            "confidence_score":      row.get("confidence_score", 0.5),
            "notes":                 str(row.get("notes", "")).strip(),
        }

        # Preserve side if present
        if "side" in row and row["side"]:
            sanitised["side"] = str(row["side"]).strip().lower()

        # Normalise amount fields
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
                sanitised["amount_pence_fraction"] = frac_f if frac_f in VALID_FRACS else ""
            except (ValueError, TypeError):
                sanitised["amount_pence_fraction"] = ""

        # Clamp confidence
        try:
            c = float(sanitised["confidence_score"])
            sanitised["confidence_score"] = round(max(0.0, min(1.0, c)), 2)
        except (ValueError, TypeError):
            sanitised["confidence_score"] = 0.5

        # If notes is empty after sanitisation, fill a minimal default
        if not sanitised["notes"]:
            sanitised["notes"] = "No reasoning provided by extractor."

        # Enforce row_type is valid
        if sanitised["row_type"] not in ("header", "entry", "total"):
            sanitised["row_type"] = "entry"

        return sanitised
