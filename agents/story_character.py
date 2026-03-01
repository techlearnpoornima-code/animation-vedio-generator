"""
agents/story_character.py
Agent 3 — Story & Character Designer

Builds character reference sheets and scene-level SD prompts.
Uses Ollama/llava for visual reasoning + outputs Stable Diffusion prompts
with IP-Adapter references for character consistency across scenes.
"""

import os
import json
import base64
import requests
from pathlib import Path
from pipeline.state import EpisodeState, CharacterData
from pipeline.ollama_client import OllamaClient

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")

SYSTEM_PROMPT = """You are a visual development artist and character designer for animated films.
You create detailed, consistent character designs and translate scripts into precise image generation prompts.
Your prompts are optimized for Stable Diffusion — specific, visual, with style keywords first.
You ensure characters look identical across all scenes by using locked visual descriptors."""


class StoryCharacterAgent:
    def __init__(self):
        self.llm = OllamaClient(model=os.getenv("OLLAMA_MODEL_TEXT", "mistral"))

    # ------------------------------------------------------------------ #
    #  Character design                                                     #
    # ------------------------------------------------------------------ #

    def _design_characters(self, state: EpisodeState) -> list[dict]:
        """Generate visual character sheets for every named character."""
        # Collect unique character names from all scenes
        all_characters = set()
        for scene in state.script.scenes:
            for name in scene.get("characters", []):
                all_characters.add(name)

        style_str = ", ".join(state.trend.style_tags)
        tone = state.trend.tone
        genre = state.trend.genre

        prompt = f"""Design characters for a {genre} cartoon with {tone} tone and style: {style_str}.

Characters to design: {", ".join(all_characters)}
Episode context: {state.script.synopsis}

For EACH character return a JSON array:
[
  {{
    "name": "CHARACTER_NAME",
    "description": "personality and role in story (2 sentences)",
    "visual_prompt": "stable diffusion character sheet prompt — style keywords first, then physical description, colors, clothing. 40-60 words. NO negative prompts.",
    "voice_profile": {{
      "gender": "male/female/neutral",
      "age_range": "child/teen/adult/elder",
      "quality": "raspy/smooth/high-pitched/deep/nasal/warm",
      "energy": "nervous/confident/deadpan/enthusiastic/sinister"
    }},
    "color_palette": ["#hex1", "#hex2", "#hex3"]
  }}
]"""

        result = self.llm.chat_json(prompt, system=SYSTEM_PROMPT, temperature=0.7)

        # Normalize — model might wrap in dict
        if isinstance(result, dict):
            result = result.get("characters", list(result.values())[0] if result else [])
        return result

    # ------------------------------------------------------------------ #
    #  Style reference                                                      #
    # ------------------------------------------------------------------ #

    def _build_style_reference(self, state: EpisodeState) -> str:
        """Build a global SD style prefix applied to every scene prompt."""
        style_tags = ", ".join(state.trend.style_tags)
        tone = state.trend.tone
        genre = state.trend.genre

        style_map = {
            "dark":       "dark atmosphere, deep shadows, muted palette, noir lighting",
            "absurdist":  "surreal composition, exaggerated proportions, vivid flat colors",
            "heartfelt":  "warm soft lighting, pastel tones, gentle linework",
            "gritty":     "gritty texture, desaturated, heavy ink outlines, urban grit",
            "whimsical":  "whimsical, fairy-tale palette, bouncy character shapes",
            "surreal":    "dreamlike, impossible geometry, shifting colors, surrealist",
            "tense":      "high contrast, dramatic angles, cool blue shadows, tension",
        }

        genre_map = {
            "action":   "dynamic action pose, speed lines, cinematic composition",
            "comedy":   "comedic exaggeration, slapstick energy, bright primary colors",
            "horror":   "horror atmosphere, eerie lighting, unsettling composition",
            "satire":   "sharp editorial style, clean graphic lines, punchy composition",
            "sci-fi":   "sci-fi environment, tech details, cool neon accents",
            "fantasy":  "fantasy world, magical lighting, rich detailed backgrounds",
        }

        tone_desc = style_map.get(tone, "expressive animation style")
        genre_desc = genre_map.get(genre, "animation style")

        return (
            f"{style_tags}, {tone_desc}, {genre_desc}, "
            f"cartoon animation, high quality, detailed illustration, "
            f"professional animation production art"
        )

    # ------------------------------------------------------------------ #
    #  Scene prompts                                                        #
    # ------------------------------------------------------------------ #

    def _build_scene_prompts(self, state: EpisodeState, characters: list[dict], style_ref: str) -> list[str]:
        """Generate one SD image prompt per scene."""
        prompts = []
        char_lookup = {c["name"]: c["visual_prompt"] for c in characters}

        for scene in state.script.scenes:
            chars_in_scene = scene.get("characters", [])
            char_descs = " | ".join(
                char_lookup.get(c, c) for c in chars_in_scene
            )

            action = scene.get("action", "")
            location = scene.get("location", "")
            mood = scene.get("mood", "")

            prompt = (
                f"{style_ref}, "
                f"scene: {location}, {mood} mood, "
                f"{action[:120]}, "
                f"characters: {char_descs[:200]}"
            )
            prompts.append(prompt)

        return prompts

    # ------------------------------------------------------------------ #
    #  Optional: generate character ref images via SD                      #
    # ------------------------------------------------------------------ #

    def _generate_character_image(self, char: dict, style_ref: str, output_dir: Path) -> str | None:
        """Generate a character reference image via ComfyUI txt2img."""
        from pipeline.comfyui_client import ComfyUIClient
        from pipeline.comfyui_workflows import txt2img_workflow

        img_path = output_dir / f"{char['name'].lower().replace(' ', '_')}.png"
        char_prompt = (
            f"{style_ref}, character reference sheet, "
            f"{char.get('visual_prompt', char['name'])}, "
            f"multiple poses, white background"
        )

        try:
            comfy = ComfyUIClient(COMFYUI_URL)
            if not comfy.is_available():
                raise RuntimeError("ComfyUI not running")

            wf = txt2img_workflow(
                prompt=char_prompt,
                width=512, height=512,
                filename_prefix=f"char_{char['name'].replace(' ', '_')}",
                )
            history = comfy.run_workflow(wf, timeout=120)
            outputs = comfy.extract_outputs(history)
            if not outputs:
                raise RuntimeError("No outputs returned")

            ok = comfy.download_output(
                outputs[0]["filename"],
                outputs[0]["subfolder"],
                outputs[0]["type"],
                img_path,
            )
            if not ok:
                raise RuntimeError("Download failed")

            print(f"  Character image → {img_path.name}")
            return str(img_path)

        except Exception as e:
            print(f"  [story_character] ComfyUI unavailable, skipping image for {char['name']}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Save to disk                                                         #
    # ------------------------------------------------------------------ #

    def _save_characters(self, state: EpisodeState) -> None:
        output_dir = Path(state.output_dir).resolve() / state.episode_id / "characters"
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "characters.json"
        with open(path, "w") as f:
            json.dump(state.characters.model_dump(), f, indent=2)

        # Save scene prompts as a text file for easy review
        prompts_path = output_dir / "scene_prompts.txt"
        with open(prompts_path, "w") as f:
            for i, p in enumerate(state.characters.scene_prompts, 1):
                f.write(f"SCENE {i}:\n{p}\n\n")

        print(f"  Character data saved → {path}")

    # ------------------------------------------------------------------ #
    #  Public entrypoint                                                    #
    # ------------------------------------------------------------------ #

    def run(self, state: EpisodeState, generate_images: bool = True) -> EpisodeState:
        print("\n[3/6] Story & Character Agent starting...")

        if not state.script:
            state.add_error("story_character", "No script found. Run ScriptWriter first.")
            return state

        try:
            print("  Designing characters...")
            characters = self._design_characters(state)

            print("  Building style reference...")
            style_ref = self._build_style_reference(state)

            print("  Generating scene prompts...")
            scene_prompts = self._build_scene_prompts(state, characters, style_ref)

            # Collect all hex colors from all characters
            all_colors = []
            for c in characters:
                all_colors.extend(c.get("color_palette", []))

            state.characters = CharacterData(
                characters=characters,
                style_reference=style_ref,
                color_palette=list(set(all_colors)),
                scene_prompts=scene_prompts,
            )

            # Optionally generate character reference images via SD
            img_dir = Path(state.output_dir).resolve() / state.episode_id / "characters" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            print("  Generating character reference images via Stable Diffusion...")
            for char in characters:
                self._generate_character_image(char, style_ref, img_dir)

            self._save_characters(state)
            state.mark_done("story_character")

            print(f"  Characters: {len(characters)}")
            print(f"  Scene prompts: {len(scene_prompts)}")
            print("  [3/6] DONE\n")

        except Exception as e:
            state.add_error("story_character", str(e))
            print(f"  [story_character] ERROR: {e}")

        return state


# ------------------------------------------------------------------ #
#  Standalone test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from agents.trend_researcher import TrendResearcher
    from agents.script_writer import ScriptWriter

    state = EpisodeState(output_dir="./output")
    state = TrendResearcher().run(state)
    state = ScriptWriter().run(state)
    state = StoryCharacterAgent().run(state, generate_images=True)

    if state.characters:
        for c in state.characters.characters:
            print(f"\n{c['name']}: {c['description']}")
            print(f"  Voice: {c['voice_profile']}")
        print(f"\nStyle ref: {state.characters.style_reference[:100]}...")
