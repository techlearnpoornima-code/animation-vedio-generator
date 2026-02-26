"""
main.py — Run the Cartoon Production Pipeline

Usage:
  python main.py
  python main.py --genre "dark comedy"
  python main.py --genre "action" --images --upload
  python main.py --agent trend       # run only trend researcher
  python main.py --agent script      # run only script writer (needs trend output)
"""

import argparse
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_dependencies():
    """Warn about missing optional dependencies without crashing."""
    issues = []
    try:
        import ollama
    except ImportError:
        pass

    try:
        import requests
        from pipeline.ollama_client import OllamaClient
        client = OllamaClient()
        if not client.is_available():
            issues.append("Ollama server not running — start it with: ollama serve")
        else:
            models = client.list_models()
            if not models:
                issues.append("No Ollama models found — pull one: ollama pull mistral")
    except Exception:
        issues.append("Could not reach Ollama API")

    try:
        import subprocess
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if r.returncode != 0:
            issues.append("FFmpeg not found — install: sudo apt install ffmpeg")
    except FileNotFoundError:
        issues.append("FFmpeg not found — install: sudo apt install ffmpeg")

    if issues:
        print("\n⚠ Dependency warnings:")
        for i in issues:
            print(f"  - {i}")
        print()

    return issues


def run_single_agent(agent_name: str, genre: str, output_dir: str):
    """Run a single agent in isolation (for testing/debugging)."""
    from pipeline.state import EpisodeState

    # Try to load existing state from disk
    state_path = Path(output_dir) / "last_state.json"
    if state_path.exists() and agent_name != "trend":
        with open(state_path) as f:
            state = EpisodeState(**json.load(f))
        print(f"  Loaded existing state: {state.episode_id}")
    else:
        state = EpisodeState(output_dir=output_dir)

    if agent_name == "trend":
        from agents.trend_researcher import TrendResearcher
        state = TrendResearcher().run(state, genre_hint=genre)

    elif agent_name == "script":
        from agents.script_writer import ScriptWriter
        if not state.trend:
            print("Error: No trend data. Run --agent trend first.")
            sys.exit(1)
        state = ScriptWriter().run(state)

    elif agent_name == "character":
        from agents.story_character import StoryCharacterAgent
        if not state.script:
            print("Error: No script data. Run --agent script first.")
            sys.exit(1)
        state = StoryCharacterAgent().run(state)

    elif agent_name == "animation":
        from agents.animation_agent import AnimationAgent
        if not state.characters:
            print("Error: No character data. Run --agent character first.")
            sys.exit(1)
        state = AnimationAgent().run(state)

    elif agent_name == "voiceover":
        from agents.voiceover_agent import VoiceoverAgent
        if not state.script:
            print("Error: No script data.")
            sys.exit(1)
        state = VoiceoverAgent().run(state)

    elif agent_name == "editor":
        from agents.editor_upload import EditorUploadAgent
        state = EditorUploadAgent().run(state, upload=False)

    else:
        print(f"Unknown agent: {agent_name}")
        print("Valid agents: trend, script, character, animation, voiceover, editor")
        sys.exit(1)

    # Save state for next agent
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        f.write(state.model_dump_json(indent=2))
    print(f"\nState saved → {state_path}")


def main():
    parser = argparse.ArgumentParser(description="Cartoon Production Pipeline")
    parser.add_argument("--genre",  type=str, default="",
                        help='Genre/tone hint e.g. "dark comedy", "action thriller"')
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--images", action="store_true",
                        help="Generate character reference images via Stable Diffusion")
    parser.add_argument("--upload", action="store_true",
                        help="Upload final video to YouTube (requires OAuth setup)")
    parser.add_argument("--agent",  type=str, default=None,
                        help="Run a single agent: trend | script | character | animation | voiceover | editor")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip dependency checks")
    args = parser.parse_args()

    if not args.skip_checks:
        check_dependencies()

    if args.agent:
        # Single-agent mode
        print(f"\nRunning single agent: {args.agent}")
        run_single_agent(args.agent, args.genre, args.output)
    else:
        # Full pipeline
        from pipeline.orchestrator import run_pipeline
        run_pipeline(
            genre_hint=args.genre,
            generate_images=args.images,
            upload_to_youtube=args.upload,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
