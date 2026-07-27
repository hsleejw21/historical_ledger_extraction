"""
src/agents/structurer.py
Agent 1 — Structurer.

Responsibilities:
  1. Classify the page layout (standard vs. complex).
  2. Produce a skeleton JSON: every row's index, type, and description.
     No monetary amounts — that's the Extractor's job.

The skeleton is the contract between Agent 1 and Agent 2.  If the Structurer
misses a row or misclassifies a type, the Extractor cannot recover it.
"""
import json
from ..clients import call_llm, parse_json_output
from ..config import MODELS
from ..prompts.structurer import (
    LAYOUT_CLASSIFIER_SYSTEM,
    LAYOUT_CLASSIFIER_USER,
    STRUCTURER_SYSTEM_STANDARD,
    STRUCTURER_SYSTEM_COMPLEX,
    STRUCTURER_USER,
)


class Structurer:
    def __init__(self, classifier_model_key: str, structurer_model_key: str):
        """
        classifier_model_key : cheap model used for the binary layout decision.
        structurer_model_key : the model benchmarked for skeleton quality.
        """
        self.classifier_cfg = MODELS[classifier_model_key]
        self.structurer_cfg = MODELS[structurer_model_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, image_path: str) -> dict:
        """
        Returns:
            {
                "layout_type": "standard" | "complex",
                "skeleton": { "layout_type": ..., "estimated_total_rows": ..., "rows": [...] },
                "_meta": { "classifier_model": ..., "structurer_model": ... }
            }
        """
        layout_type = self._classify(image_path)
        skeleton    = self._structure(image_path, layout_type)

        # Normalise: ensure layout_type in skeleton matches classifier decision
        skeleton["layout_type"] = layout_type

        return {
            "layout_type": layout_type,
            "skeleton": skeleton,
            "_meta": {
                "classifier_model": self.classifier_cfg["model_name"],
                "structurer_model": self.structurer_cfg["model_name"],
            }
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------
    def _classify(self, image_path: str) -> str:
        """Binary classification: standard or complex."""
        raw = call_llm(
            provider=self.classifier_cfg["provider"],
            model_name=self.classifier_cfg["model_name"],
            system_prompt=LAYOUT_CLASSIFIER_SYSTEM,
            user_prompt=LAYOUT_CLASSIFIER_USER,
            image_path=image_path,
        )
        result = parse_json_output(raw)
        layout = result.get("layout_type", "standard")
        # Sanitise: only accept known values
        return layout if layout in ("standard", "complex") else "standard"

    def _structure(self, image_path: str, layout_type: str) -> dict:
        """Produce the skeleton using the layout-appropriate prompt."""
        system = STRUCTURER_SYSTEM_COMPLEX if layout_type == "complex" else STRUCTURER_SYSTEM_STANDARD

        raw = call_llm(
            provider=self.structurer_cfg["provider"],
            model_name=self.structurer_cfg["model_name"],
            system_prompt=system,
            user_prompt=STRUCTURER_USER,
            image_path=image_path,
        )
        skeleton = parse_json_output(raw)

        # Safety: if the model returned nothing useful, return a minimal valid skeleton
        if "rows" not in skeleton or not skeleton["rows"]:
            skeleton = {
                "layout_type": layout_type,
                "estimated_total_rows": 0,
                "rows": []
            }

        return skeleton
