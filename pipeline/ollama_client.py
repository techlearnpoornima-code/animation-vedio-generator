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
    ) -> dict:
        """
        Like chat() but forces JSON output.
        Adds JSON instruction to system prompt and parses the response.
        """
        json_instruction = (
            "\nYou MUST respond with valid JSON only. "
            "No explanation, no markdown fences, no extra text. "
            "Output raw JSON that can be parsed with json.loads()."
        )
        system_with_json = (system or "") + json_instruction

        raw = self.chat(prompt, system=system_with_json, temperature=temperature)

        # Strip markdown code fences if model added them anyway
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract first JSON object/array from the text
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Could not parse JSON from model response:\n{raw[:500]}")

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
