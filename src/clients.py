"""
src/clients.py
Unified LLM client.  One function (call_llm) routes to the correct provider.
All provider-specific boilerplate is contained here; the rest of the codebase
never imports an SDK directly.
"""
import base64
import json
import os
import time

from .config import OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY

# ---------------------------------------------------------------------------
# Lazy client initialisation  (only instantiate what's configured)
# ---------------------------------------------------------------------------
_client_openai = None
_client_google = None
_client_anthropic = None


def _get_openai():
    global _client_openai
    if _client_openai is None:
        from openai import OpenAI
        _client_openai = OpenAI(api_key=OPENAI_API_KEY)
    return _client_openai


def _get_google():
    global _client_google
    if _client_google is None:
        from google import genai
        _client_google = genai.Client(api_key=GOOGLE_API_KEY)
    return _client_google


def _get_anthropic():
    global _client_anthropic
    if _client_anthropic is None:
        import anthropic
        _client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client_anthropic


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _encode_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _media_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")


# ---------------------------------------------------------------------------
# Provider-specific call implementations
# ---------------------------------------------------------------------------
def _call_openai(model_name: str, system_prompt: str, user_prompt: str, image_path: str) -> str:
    client = _get_openai()
    b64 = _encode_base64(image_path)
    mime = _media_type(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                    "detail": "high"
                }}
            ]}
        ]
    )
    return response.choices[0].message.content or ""


def _call_google(model_name: str, system_prompt: str, user_prompt: str, image_path: str) -> str:
    client = _get_google()
    from PIL import Image
    img = Image.open(image_path)

    response = client.models.generate_content(
        model=model_name,
        contents=[system_prompt, user_prompt, img]
    )
    return response.text or ""


def _call_anthropic(model_name: str, system_prompt: str, user_prompt: str, image_path: str) -> str:
    client = _get_anthropic()
    b64 = _encode_base64(image_path)
    mime = _media_type(image_path)

    response = client.messages.create(
        model=model_name,
        max_tokens=8192,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": user_prompt}
            ]
        }]
    )
    return response.content[0].text or ""


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
_DISPATCH = {
    "openai":    _call_openai,
    "google":    _call_google,
    "anthropic": _call_anthropic,
}


def call_llm(provider: str, model_name: str, system_prompt: str, user_prompt: str, image_path: str) -> str:
    """
    Unified call.  Raises on unknown provider; returns empty string on API errors.
    """
    handler = _DISPATCH.get(provider)
    if handler is None:
        raise ValueError(f"Unknown provider: {provider}")

    try:
        return handler(model_name, system_prompt, user_prompt, image_path)
    except Exception as e:
        print(f"    [Error] {provider}/{model_name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# JSON extraction from raw LLM text
# ---------------------------------------------------------------------------
def parse_json_output(response_text: str) -> dict:
    """
    Extracts the first valid JSON object from LLM output.
    Handles: bare JSON, ```json … ```, text-before-JSON, thinking blocks, etc.
    """
    if not response_text:
        return {}

    text = response_text.strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif text.startswith("```"):
        text = text[3:]
        if "```" in text:
            text = text.split("```", 1)[0]

    text = text.strip()

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost { … }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {}
