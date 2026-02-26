"""
agents/script_writer.py
Agent 2 — Script Writer

Takes the trend brief and writes a full cartoon episode script.
Uses Ollama/mistral with a genre-aware system prompt.
Outputs structured scene-by-scene JSON + raw readable script text.
"""

import os
import json
from pathlib import Path
from pipeline.state import EpisodeState, ScriptData
from pipeline.ollama_client import OllamaClient

SYSTEM_PROMPT = """You are an award-winning animation screenwriter.
You write scripts for all audiences — sharp adult satire, dark thriller, absurdist comedy, action drama.
Your scripts have:
- Tight, punchy dialogue that sounds natural when spoken aloud
- Strong visual gags and action beats (this is animation — show don't tell)
- A clear 3-act structure even in short episodes
- Memorable characters with distinct voices
- An unexpected twist or subversive element in every episode

Format matters: each scene must be clearly numbered with location, action, and dialogue separated."""


class ScriptWriter:
    def __init__(self):
        self.llm = OllamaClient(model=os.getenv("OLLAMA_MODEL_TEXT", "mistral"))

    # ------------------------------------------------------------------ #
    #  Scene structure generation                                           #
    # ------------------------------------------------------------------ #

    def _generate_scene_structure(self, state: EpisodeState) -> list[dict]:
        """First pass: generate scene-by-scene outline as JSON."""
        trend = state.trend

        prompt = f"""Write a cartoon episode outline for:
Topic: {trend.topic}
Genre: {trend.genre}
Tone: {trend.tone}
Audience: {trend.target_audience}
Style: {", ".join(trend.style_tags)}
Hooks to hit: {", ".join(trend.hooks)}

Return a JSON array of scenes (6-10 scenes for ~5 minute episode):
[
  {{
    "scene_num": 1,
    "location": "brief location description",
    "characters": ["CharacterName1", "CharacterName2"],
    "action": "2-3 sentences describing what happens visually",
    "dialogue": [
      {{"character": "NAME", "line": "dialogue text"}},
      {{"character": "NAME", "line": "dialogue text"}}
    ],
    "mood": "tense / funny / sad / exciting / eerie",
    "duration_secs": 30
  }}
]"""

        return self.llm.chat_json(prompt, system=SYSTEM_PROMPT, temperature=0.8)

    # ------------------------------------------------------------------ #
    #  Full script generation                                               #
    # ------------------------------------------------------------------ #

    def _generate_full_script(self, scenes: list[dict], state: EpisodeState) -> str:
        """Second pass: flesh out scenes into a proper readable script."""
        trend = state.trend
        scenes_summary = json.dumps(scenes, indent=2)

        prompt = f"""You have this scene outline for a {trend.genre} cartoon ({trend.tone} tone):
{scenes_summary}

Write the complete, production-ready script. For each scene include:
- Scene header: INT./EXT. LOCATION - TIME
- Action lines (present tense, visual, punchy)  
- Character dialogue with parentheticals where needed
- (BEAT), (PAUSE), (V.O.) markers where appropriate

Make dialogue sharp and character voices distinct. 
Total target: ~5 minutes of content."""

        return self.llm.chat(prompt, system=SYSTEM_PROMPT, temperature=0.75)

    # ------------------------------------------------------------------ #
    #  Title generation                                                     #
    # ------------------------------------------------------------------ #

    def _generate_title(self, state: EpisodeState, scenes: list[dict]) -> str:
        prompt = f"""Generate a catchy episode title for this {state.trend.genre} cartoon.
Topic: {state.trend.topic}
Tone: {state.trend.tone}
Scenes: {len(scenes)} scenes

Return ONLY the title, nothing else. No quotes."""
        return self.llm.chat(prompt, temperature=0.9).strip().strip('"').strip("'")

    # ------------------------------------------------------------------ #
    #  Save to disk                                                         #
    # ------------------------------------------------------------------ #

    def _save_script(self, state: EpisodeState) -> str:
        output_dir = Path(state.output_dir) / state.episode_id / "scripts"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save structured JSON
        json_path = output_dir / "script.json"
        with open(json_path, "w") as f:
            json.dump(state.script.model_dump(), f, indent=2)

        # Save human-readable script
        txt_path = output_dir / "script.txt"
        with open(txt_path, "w") as f:
            f.write(f"EPISODE: {state.script.title}\n")
            f.write(f"Genre: {state.trend.genre} | Tone: {state.trend.tone}\n")
            f.write(f"Audience: {state.trend.target_audience}\n")
            f.write("=" * 60 + "\n\n")
            f.write(state.script.raw_text)

        print(f"  Script saved → {txt_path}")
        return str(txt_path)

    # ------------------------------------------------------------------ #
    #  Public entrypoint                                                    #
    # ------------------------------------------------------------------ #

    def run(self, state: EpisodeState) -> EpisodeState:
        print("\n[2/6] Script Writer starting...")

        if not state.trend:
            state.add_error("script_writer", "No trend data found. Run TrendResearcher first.")
            return state

        try:
            print("  Generating scene structure...")
            scenes = self._generate_scene_structure(state)

            # Handle both list and dict responses from model
            if isinstance(scenes, dict):
                scenes = scenes.get("scenes", list(scenes.values())[0] if scenes else [])

            print(f"  Generated {len(scenes)} scenes. Writing full script...")
            raw_script = self._generate_full_script(scenes, state)

            title = self._generate_title(state, scenes)
            total_duration = sum(s.get("duration_secs", 30) for s in scenes) / 60

            # Determine episode number from output dir
            ep_num = len(list(Path(state.output_dir).glob("EP_*"))) + 1

            state.script = ScriptData(
                title=title,
                episode_number=ep_num,
                synopsis=state.trend.topic,
                scenes=scenes,
                total_scenes=len(scenes),
                estimated_duration_mins=round(total_duration, 1),
                tone=state.trend.tone,
                raw_text=raw_script,
            )

            self._save_script(state)
            state.mark_done("script_writing")

            print(f"  Title: \"{state.script.title}\"")
            print(f"  Scenes: {state.script.total_scenes} | Est. duration: {state.script.estimated_duration_mins} min")
            print("  [2/6] DONE\n")

        except Exception as e:
            state.add_error("script_writer", str(e))
            print(f"  [script_writer] ERROR: {e}")

        return state


# ------------------------------------------------------------------ #
#  Standalone test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from agents.trend_researcher import TrendResearcher

    state = EpisodeState(output_dir="./output")
    state = TrendResearcher().run(state, genre_hint="dark comedy")
    state = ScriptWriter().run(state)

    if state.script:
        print(f"\nTitle: {state.script.title}")
        print(f"Scenes: {state.script.total_scenes}")
        print("\n--- SCRIPT PREVIEW ---")
        print(state.script.raw_text[:800] + "...")
