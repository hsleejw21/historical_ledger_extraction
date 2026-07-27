"""
src/agents/supervisor.py
V2 Supervisor Agent.

Receives the outputs of all standalone extractors for one page, assembles
them into a single prompt alongside the original image, and asks the
Supervisor LLM to produce a merged row-by-row selection.
"""
import json
from ..clients import call_llm, parse_json_output
from ..config import MODELS
from ..prompts.supervisor import SUPERVISOR_SYSTEM, SUPERVISOR_USER_TEMPLATE
from .standalone_extractor import StandaloneExtractor  # reuse sanitise_row


class Supervisor:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.cfg = MODELS[model_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, image_path: str, candidates: dict) -> dict:
        """
        Args:
            image_path : path to the ledger scan.
            candidates : dict mapping model_key → extractor output.
                         Each value must have "rows" list with notes.

        Returns:
            {
                "rows": [ merged rows ],
                "_meta": {
                    "supervisor_model": ...,
                    "candidate_models": [...],
                    "supervisor_meta": { ... stats from the LLM ... }
                }
            }
        """
        # Strip _meta from each candidate before sending to the LLM
        # (keeps the prompt focused on the actual data)
        candidates_clean = {}
        candidate_model_keys = []
        for model_key, output in candidates.items():
            candidate_model_keys.append(model_key)
            candidates_clean[model_key] = {
                k: v for k, v in output.items() if k != "_meta"
            }

        candidates_json = json.dumps(candidates_clean, indent=2, ensure_ascii=False)

        user_prompt = SUPERVISOR_USER_TEMPLATE.format(
            n_candidates=len(candidates_clean),
            candidates_json=candidates_json,
        )

        raw = call_llm(
            provider=self.cfg["provider"],
            model_name=self.cfg["model_name"],
            system_prompt=SUPERVISOR_SYSTEM,
            user_prompt=user_prompt,
            image_path=image_path,
        )

        merged = parse_json_output(raw)

        # Safety: if parse failed, fall back to the candidate with the most rows
        if "rows" not in merged or not merged["rows"]:
            merged = self._fallback(candidates)

        # Sanitise every row through the same logic the extractors use
        merged["rows"] = [StandaloneExtractor._sanitise_row(r) for r in merged["rows"]]

        # Extract supervisor_meta if the LLM included it
        supervisor_meta = merged.pop("_supervisor_meta", {})

        merged["_meta"] = {
            "supervisor_model":  self.model_key,
            "candidate_models":  candidate_model_keys,
            "supervisor_meta":   supervisor_meta,
        }

        return merged

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback(candidates: dict) -> dict:
        """
        If the Supervisor LLM returned nothing useful, pick the candidate
        with the most rows as a last resort.
        """
        best = {"rows": []}
        for output in candidates.values():
            if len(output.get("rows", [])) > len(best["rows"]):
                best = output
        # Return a copy with just rows (strip _meta if present)
        return {"rows": list(best.get("rows", []))}
