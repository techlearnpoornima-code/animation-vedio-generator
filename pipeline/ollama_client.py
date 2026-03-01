"""
pipeline/ollama_client.py
Thin wrapper around the Ollama API used by every agent.
Handles retries, timeouts, and JSON extraction from responses.
"""

import json
import re
import time
import requests
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")



def _sanitize_json_string(raw: str) -> str:
    """
    Replace literal control characters inside JSON string values.
    These cause "Invalid control character at line X" errors from json.loads().
    Walks char-by-char tracking in/out of JSON strings.
    """
    result = []
    in_string = False
    escape_next = False

    CONTROL_ESCAPES = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
    }

    for ch in raw:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch in CONTROL_ESCAPES:
            result.append(CONTROL_ESCAPES[ch])
            continue
        if in_string and ord(ch) < 0x20:
            result.append(f"\\u{ord(ch):04x}")
            continue
        result.append(ch)

    return "".join(result)


class OllamaClient:
    def __init__(self, model: Optional[str] = None):
        self.base_url = OLLAMA_BASE_URL
        self.model = model or os.getenv("OLLAMA_MODEL_TEXT", "mistral")

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        retries: int = 3,
    ) -> str:
        """Send a prompt and return the text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
        }

        for attempt in range(retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"].strip()
            except Exception as e:
                if attempt < retries - 1:
                    print(f"  [ollama] Retry {attempt + 1}/{retries}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Ollama call failed after {retries} attempts: {e}")

    def chat_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.5,
        retries: int = 3,
    ) -> dict:
        """
        Like chat() but forces JSON output.
        Adds JSON instruction to system prompt, sanitizes the response,
        and retries up to `retries` times on parse failure.
        """
        json_instruction = (
            "\nYou MUST respond with valid JSON only. "
            "No explanation, no markdown fences, no extra text. "
            "All string values must be on a single line — do NOT include "
            "literal newlines or tab characters inside JSON string values. "
            "Use \\n for newlines and \\t for tabs inside strings. "
            "Output raw JSON that can be parsed with json.loads()."
        )
        system_with_json = (system or "") + json_instruction

        last_err: Exception = ValueError("No attempts made")

        for attempt in range(retries):
            raw = self.chat(prompt, system=system_with_json, temperature=temperature)

            try:
                return self._parse_json(raw)
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                print(f"  [ollama] JSON parse failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    # Ask the model to fix its own output
                    prompt = (
                        f"Your previous response was not valid JSON. Error: {e}\n"
                        f"Previous response (first 300 chars): {raw[:300]}\n\n"
                        f"Fix it and return ONLY valid JSON. No prose, no fences."
                    )
                    temperature = max(0.1, temperature - 0.15)  # reduce creativity on retry

        raise ValueError(f"chat_json failed after {retries} attempts. Last error: {last_err}")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        Aggressively clean and parse a JSON string from an LLM response.
        Handles: markdown fences, control characters, leading/trailing prose.
        """
        # 1. Strip markdown code fences
        raw = re.sub(r"```(?:json|JSON)?\s*", "", raw)
        raw = raw.replace("```", "").strip()

        # 2. Remove any prose before the first { or [
        first_brace = min(
            (raw.find(c) for c in ["{", "["] if raw.find(c) != -1),
            default=-1,
        )
        if first_brace > 0:
            raw = raw[first_brace:]

        # 3. Remove any prose after the last } or ]
        last_brace = max(raw.rfind("}"), raw.rfind("]"))
        if last_brace != -1:
            raw = raw[:last_brace + 1]

        # 4. Replace literal control characters inside JSON strings
        #    (the main cause of "Invalid control character" errors)
        #    We do this carefully: only replace control chars that appear
        #    INSIDE string values (between quotes), not structural chars.
        raw = _sanitize_json_string(raw)

        # 5. Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 6. Try with strict=False (allows some control chars)
        try:
            return json.loads(raw, strict=False)
        except json.JSONDecodeError:
            pass

        # 7. Last resort: extract first valid JSON structure
        for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
            match = re.search(pattern, raw)
            if match:
                candidate = _sanitize_json_string(match.group(1))
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        return json.loads(candidate, strict=False)
                    except json.JSONDecodeError:
                        pass

        raise ValueError(f"Could not parse JSON. First 400 chars:\n{raw[:400]}")

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        """Return list of locally available model names."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
