"""
pipeline/quality_gate.py
Quality Gate — runs between Script Writer and Animation Agent.

Scores the script for coherence, entertainment value, and tone consistency.
Blocks low-quality output from entering the expensive animation stage.
"""

import os
from pipeline.state import EpisodeState
from pipeline.ollama_client import OllamaClient

MIN_SCRIPT_SCORE = float(os.getenv("MIN_SCRIPT_SCORE", "6.0"))
MIN_CHARACTER_SCORE = float(os.getenv("MIN_CHARACTER_SCORE", "5.5"))

SYSTEM_PROMPT = """You are a senior creative director reviewing animation scripts.
You give honest, critical evaluations. Your scores are strict — a 7 means genuinely good.
You catch: inconsistent tone, weak dialogue, unclear action descriptions, character voice issues."""


class QualityGate:

    def __init__(self):
        self.llm = OllamaClient(model=os.getenv("OLLAMA_MODEL_FAST", "llama3"))

    def _score_script(self, state: EpisodeState) -> dict:
        script = state.script

        prompt = f"""Review this cartoon episode script and score it.

Title: {script.title}
Genre: {state.trend.genre} | Tone: {state.trend.tone}
Scenes: {script.total_scenes}
Script excerpt (first 1000 chars):
{script.raw_text[:1000]}

Score each criterion 1-10 and return JSON:
{{
  "overall_score": 7.5,
  "coherence": 8,
  "entertainment": 7,
  "tone_consistency": 8,
  "dialogue_quality": 6,
  "visual_potential": 7,
  "verdict": "PASS or FAIL",
  "notes": "brief feedback (1-2 sentences)"
}}

Minimum passing overall_score: {MIN_SCRIPT_SCORE}"""

        return self.llm.chat_json(prompt, system=SYSTEM_PROMPT, temperature=0.3)

    def _score_characters(self, state: EpisodeState) -> dict:
        chars = state.characters
        char_summary = "\n".join(
            f"- {c['name']}: {c.get('description', '')} | visual: {c.get('visual_prompt', '')[:80]}"
            for c in chars.characters[:5]
        )

        prompt = f"""Review these cartoon character designs for animation quality.

Style: {chars.style_reference[:150]}
Characters:
{char_summary}

Score each criterion 1-10 and return JSON:
{{
  "overall_score": 7.0,
  "visual_distinctiveness": 7,
  "prompt_specificity": 7,
  "style_consistency": 8,
  "verdict": "PASS or FAIL",
  "notes": "brief feedback"
}}

Minimum passing overall_score: {MIN_CHARACTER_SCORE}"""

        return self.llm.chat_json(prompt, system=SYSTEM_PROMPT, temperature=0.3)

    def run(self, state: EpisodeState) -> tuple[EpisodeState, bool]:
        """
        Returns (updated_state, passed: bool).
        If False, the orchestrator should regenerate the failed stage.
        """
        print("\n[QA] Quality Gate running...")
        passed = True

        # --- Script scoring ---
        if state.script:
            try:
                result = self._score_script(state)
                score = float(result.get("overall_score", 0))
                state.script_score = score
                verdict = result.get("verdict", "FAIL")
                notes = result.get("notes", "")

                print(f"  Script score: {score}/10 | {verdict}")
                if notes:
                    print(f"  Notes: {notes}")

                if score < MIN_SCRIPT_SCORE or verdict == "FAIL":
                    state.add_error("quality_gate", f"Script score {score} below threshold {MIN_SCRIPT_SCORE}. {notes}")
                    passed = False

            except Exception as e:
                state.warnings.append(f"Script scoring failed: {e}")
                print(f"  Script scoring WARN: {e} — skipping gate")

        # --- Character scoring ---
        if state.characters:
            try:
                result = self._score_characters(state)
                score = float(result.get("overall_score", 0))
                state.character_score = score
                verdict = result.get("verdict", "FAIL")
                notes = result.get("notes", "")

                print(f"  Character score: {score}/10 | {verdict}")
                if notes:
                    print(f"  Notes: {notes}")

                if score < MIN_CHARACTER_SCORE or verdict == "FAIL":
                    state.warnings.append(f"Character score {score} below threshold — continuing anyway")

            except Exception as e:
                state.warnings.append(f"Character scoring failed: {e}")

        status = "PASSED" if passed else "FAILED"
        print(f"  [QA] {status}\n")
        return state, passed
