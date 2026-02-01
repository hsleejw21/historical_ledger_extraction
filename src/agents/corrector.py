"""
src/agents/corrector.py
Agent 3 — Corrector.

Receives the Extractor's output and the original image.  Performs a
chain-of-thought forensic audit.  Returns both the audit log (thinking block)
and the corrected JSON so we can track what changed and why.
"""
import json
from ..clients import call_llm, parse_json_output
from ..config import MODELS
from ..prompts.corrector import CORRECTOR_SYSTEM, CORRECTOR_USER_TEMPLATE
from .extractor import Extractor  # reuse sanitise_row


class Corrector:
    def __init__(self, model_key: str):
        self.cfg = MODELS[model_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, image_path: str, extraction: dict) -> dict:
        """
        Args:
            image_path : path to the ledger scan.
            extraction : the full extraction dict from the Extractor
                         (must contain "rows" list).  _meta is stripped before
                         sending to the model.

        Returns:
            {
                "rows": [ corrected rows ],
                "_meta": {
                    "model": ...,
                    "thinking_block": <the audit log text>,
                    "changes_made": <bool>
                }
            }
        """
        # Strip internal meta before sending to the model
        extraction_for_prompt = {k: v for k, v in extraction.items() if k != "_meta"}
        extraction_json = json.dumps(extraction_for_prompt, indent=2, ensure_ascii=False)

        user_prompt = CORRECTOR_USER_TEMPLATE.format(extraction_json=extraction_json)

        raw = call_llm(
            provider=self.cfg["provider"],
            model_name=self.cfg["model_name"],
            system_prompt=CORRECTOR_SYSTEM,
            user_prompt=user_prompt,
            image_path=image_path,
        )

        thinking_block, corrected = self._parse_corrector_output(raw, extraction_for_prompt)

        # Sanitise every row through the same pipeline as the Extractor
        corrected["rows"] = [Extractor._sanitise_row(r) for r in corrected.get("rows", [])]

        # If the corrector returned nothing usable, fall back to the original extraction
        if not corrected["rows"]:
            corrected = extraction_for_prompt
            thinking_block = "Corrector returned empty output; falling back to Extractor result."

        corrected["_meta"] = {
            "model": self.cfg["model_name"],
            "thinking_block": thinking_block,
            "changes_made": thinking_block != "Corrector returned empty output; falling back to Extractor result.",
        }

        return corrected

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_corrector_output(raw: str, fallback_extraction: dict) -> tuple:
        """
        The Corrector is instructed to output:
            <thinking block text>
            ---CORRECTED_JSON---
            { ... json ... }

        Returns:
            (thinking_block: str, corrected_json: dict)
        """
        if not raw:
            return "No output from Corrector.", fallback_extraction

        DELIMITER = "---CORRECTED_JSON---"

        if DELIMITER in raw:
            parts = raw.split(DELIMITER, 1)
            thinking = parts[0].strip()
            json_part = parts[1].strip()
            corrected = parse_json_output(json_part)
            if corrected and "rows" in corrected:
                return thinking, corrected

        # Fallback: no delimiter found — try to extract JSON from anywhere in the output
        # and treat everything before it as the thinking block
        corrected = parse_json_output(raw)
        if corrected and "rows" in corrected:
            # Find where the JSON starts to split the thinking block
            json_start = raw.find("{")
            thinking = raw[:json_start].strip() if json_start > 0 else "No explicit thinking block found."
            return thinking, corrected

        # Nothing worked
        return "Failed to parse Corrector output.", fallback_extraction
