#!/usr/bin/env python3
"""
tools/test_pipeline.py — Step-by-step pipeline debugger

Tests each agent individually, inspects state between steps,
diagnoses environment issues, and fixes the audio/animation fallbacks.

Usage:
    uv run python tools/test_pipeline.py --check          # environment check
    uv run python tools/test_pipeline.py --agent trend    # run + inspect one agent
    uv run python tools/test_pipeline.py --agent script
    uv run python tools/test_pipeline.py --agent character
    uv run python tools/test_pipeline.py --agent animation
    uv run python tools/test_pipeline.py --agent voiceover
    uv run python tools/test_pipeline.py --agent editor
    uv run python tools/test_pipeline.py --all            # run all agents, pause between each
    uv run python tools/test_pipeline.py --inspect        # inspect saved state from last run
    uv run python tools/test_pipeline.py --probe <file>   # probe an audio or video file
"""

import sys
import os
import json
import argparse
import subprocess
import time
from pathlib import Path

# ── Make sure project root is on path ────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # set cwd to project root so relative paths work

STATE_FILE = ROOT / "output" / "last_state.json"
OUTPUT_DIR  = str(ROOT / "output")

# ── Terminal colours ─────────────────────────────────────────────────────────
R  = "\033[0;31m"  # red
G  = "\033[0;32m"  # green
Y  = "\033[1;33m"  # yellow
C  = "\033[0;36m"  # cyan
B  = "\033[1m"     # bold
DIM= "\033[2m"     # dim
RST= "\033[0m"     # reset

def ok(msg):  print(f"  {G}✓{RST} {msg}")
def err(msg): print(f"  {R}✗{RST} {msg}")
def warn(msg):print(f"  {Y}!{RST} {msg}")
def info(msg):print(f"  {C}→{RST} {msg}")
def hdr(msg): print(f"\n{B}{msg}{RST}\n{'─'*len(msg)}")

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_environment():
    hdr("Environment Diagnostic")

    all_ok = True

    # ── Ollama ────────────────────────────────────────────────────────────
    print(f"\n{C}Ollama{RST}")
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        ok(f"Ollama running — {len(models)} models: {', '.join(models) or 'none'}")

        needed = ["mistral", "llama3", "llava"]
        for m in needed:
            found = any(m in x for x in models)
            (ok if found else warn)(f"  {m}: {'found' if found else 'MISSING — run: ollama pull ' + m}")
            if not found:
                all_ok = False
    except Exception as e:
        err(f"Ollama not reachable: {e}")
        info("Start with: ollama serve (and ComfyUI: python main.py --listen 0.0.0.0 --port 8188)")
        all_ok = False

    # ── FFmpeg ────────────────────────────────────────────────────────────
    print(f"\n{C}FFmpeg{RST}")
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ver = r.stdout.split("\n")[0]
        ok(ver)

        # Test colour card generation
        test_mp4 = Path("/tmp/test_ffmpeg_clip.mp4")
        r2 = subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=0x1a1a2e:size=256x256:rate=24",
            "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(test_mp4)
        ], capture_output=True)
        (ok if r2.returncode == 0 else err)(f"Video generation: {'OK' if r2.returncode == 0 else 'FAILED'}")

        # Test audio generation
        test_wav = Path("/tmp/test_ffmpeg_audio.wav")
        r3 = subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1",
            "-acodec", "pcm_s16le", "-ar", "44100",
            str(test_wav)
        ], capture_output=True)
        (ok if r3.returncode == 0 else err)(f"Audio generation: {'OK' if r3.returncode == 0 else 'FAILED'}")

    except FileNotFoundError:
        err("FFmpeg not found")
        info("Install: sudo apt install ffmpeg")
        all_ok = False

    # ── TTS engines ───────────────────────────────────────────────────────
    print(f"\n{C}TTS Engines{RST}")
    tts_found = False
    for binary in ["tts", "espeak-ng", "espeak", "say"]:
        r = subprocess.run(["which", binary], capture_output=True)
        if r.returncode == 0:
            ok(f"{binary} found at {r.stdout.strip().decode()}")
            tts_found = True
        else:
            print(f"  {DIM}  {binary}: not found{RST}")

    if not tts_found:
        warn("No TTS engine found — audio will be SILENT placeholders")
        info("Install Coqui: uv pip install TTS")
        info("Or espeak:     sudo apt install espeak-ng")

    # ── Stable Diffusion ──────────────────────────────────────────────────
    print(f"\n{C}ComfyUI{RST}")
    comfy_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
    try:
        import requests
        r = requests.get(f"{comfy_url}/system_stats", timeout=3)
        ok(f"ComfyUI running at {comfy_url}")

        # Check AnimateDiff
        r2 = requests.get(f"{comfy_url}/object_info/ADE_AnimateDiffLoaderWithContext", timeout=3)
        (ok if r2.status_code == 200 else warn)(
            f"AnimateDiff: {'available' if r2.status_code == 200 else 'NOT FOUND (animation will use FFmpeg placeholders)'}"
        )
    except Exception:
        warn(f"ComfyUI not running at at {comfy_url}")
        info("Animation will use FFmpeg colour-card placeholders")
        info("Install: https://github.com/AUTOMATIC1111/ComfyUI")

    # ── Python deps ───────────────────────────────────────────────────────
    print(f"\n{C}Python Dependencies{RST}")
    deps = {
        "pydantic":       "core",
        "langgraph":      "orchestrator",
        "langchain":      "agent framework",
        "requests":       "HTTP client",
        "feedparser":     "trend RSS",
        "fastapi":        "web UI",
        "uvicorn":        "web server",
        "dotenv":         "python-dotenv",
    }
    for pkg, label in deps.items():
        try:
            __import__(pkg)
            ok(f"{pkg} ({label})")
        except ImportError:
            err(f"{pkg} MISSING ({label}) — run: uv add {pkg}")
            all_ok = False

    # ── Output dir ────────────────────────────────────────────────────────
    print(f"\n{C}File System{RST}")
    out = ROOT / "output"
    out.mkdir(parents=True, exist_ok=True)
    ok(f"Output dir: {out}")
    info(f"Project root: {ROOT}")
    info(f"CWD: {os.getcwd()}")

    print()
    if all_ok:
        ok("All critical checks passed")
    else:
        warn("Some checks failed — see above")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_state():
    """Load saved state from disk, or create fresh."""
    from pipeline.state import EpisodeState
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        state = EpisodeState(**data)
        info(f"Loaded state: {state.episode_id}")
        return state
    info("No saved state found — creating fresh")
    return EpisodeState(output_dir=OUTPUT_DIR)


def save_state(state):
    """Save state to disk for next agent to pick up."""
    from pipeline.state import EpisodeState
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(state.model_dump_json(indent=2))
    ok(f"State saved → {STATE_FILE}")


def print_state(state, verbose=False):
    """Pretty-print the current pipeline state."""
    hdr(f"State: {state.episode_id}")

    steps = state.completed_steps
    all_steps = ["trend_research","script_writing","story_character","animation","voiceover","editor_upload"]
    print(f"\n  {'Step':<20} {'Status'}")
    print(f"  {'─'*20} {'─'*10}")
    for s in all_steps:
        done = s in steps
        sym  = f"{G}✓{RST}" if done else f"{DIM}○{RST}"
        print(f"  {sym} {s}")

    if state.trend:
        print(f"\n  {C}Trend{RST}")
        print(f"    Topic:    {state.trend.topic}")
        print(f"    Genre:    {state.trend.genre} | Tone: {state.trend.tone}")
        print(f"    Audience: {state.trend.target_audience}")
        print(f"    Style:    {', '.join(state.trend.style_tags)}")

    if state.script:
        print(f"\n  {C}Script{RST}")
        print(f"    Title:    {state.script.title}")
        print(f"    Scenes:   {state.script.total_scenes}")
        print(f"    Duration: {state.script.estimated_duration_mins} min")
        if verbose:
            for i, sc in enumerate(state.script.scenes[:3], 1):
                print(f"    Scene {i}: [{sc.get('location','')}] {sc.get('action','')[:60]}...")
                for d in sc.get('dialogue', [])[:2]:
                    print(f"      {d.get('character','?')}: {d.get('line','')[:50]}")

    if state.characters:
        print(f"\n  {C}Characters{RST}")
        for c in state.characters.characters:
            vp = c.get("voice_profile", {})
            print(f"    {c['name']}: {c.get('description','')[:50]}")
            print(f"      Voice: {vp.get('gender','')} · {vp.get('quality','')} · {vp.get('energy','')}")
        print(f"    Style: {state.characters.style_reference[:80]}...")

    if state.animation:
        print(f"\n  {C}Animation{RST}")
        print(f"    Clips: {state.animation.total_clips}")
        for p in state.animation.clip_paths:
            path = Path(p)
            exists = path.exists()
            size = f"{path.stat().st_size//1024}KB" if exists else "MISSING"
            sym = G if exists else R
            print(f"    {sym}{'✓' if exists else '✗'}{RST} {path.name} ({size})")
            if exists:
                _probe_file_brief(p)

    if state.audio:
        print(f"\n  {C}Audio{RST}")
        print(f"    Files: {len(state.audio.audio_paths)}")
        for p in state.audio.audio_paths:
            path = Path(p)
            exists = path.exists()
            size = f"{path.stat().st_size//1024}KB" if exists else "MISSING"
            sym = G if exists else R
            print(f"    {sym}{'✓' if exists else '✗'}{RST} {path.name} ({size})")
            if exists:
                _probe_file_brief(p)

    if state.final_video_path:
        path = Path(state.final_video_path)
        exists = path.exists()
        print(f"\n  {C}Final Video{RST}")
        print(f"    {'✓' if exists else '✗ MISSING'} {state.final_video_path}")
        if exists:
            _probe_file_brief(state.final_video_path)

    if state.errors:
        print(f"\n  {R}Errors{RST}")
        for e in state.errors:
            print(f"    ✗ {e}")

    if state.warnings:
        print(f"\n  {Y}Warnings{RST}")
        for w in state.warnings:
            print(f"    ! {w}")

    print()


def _probe_file_brief(path: str):
    """One-line ffprobe summary for a file."""
    try:
        r = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_name,codec_type,width,height,sample_rate,duration",
            "-show_entries", "format=duration,bit_rate",
            "-of", "json", path
        ], capture_output=True, text=True, timeout=5)
        data = json.loads(r.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        dur = float(fmt.get("duration", 0))
        br  = int(fmt.get("bit_rate", 0)) // 1000

        parts = []
        for s in streams:
            ct = s.get("codec_type", "")
            cn = s.get("codec_name", "")
            if ct == "video":
                parts.append(f"video:{cn} {s.get('width')}x{s.get('height')}")
            elif ct == "audio":
                sr = s.get("sample_rate", "?")
                parts.append(f"audio:{cn} {sr}Hz")
        print(f"      {DIM}{dur:.1f}s | {br}kbps | {' | '.join(parts)}{RST}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# FILE PROBE
# ─────────────────────────────────────────────────────────────────────────────

def probe_file(path: str):
    hdr(f"File Probe: {Path(path).name}")
    if not Path(path).exists():
        err(f"File not found: {path}")
        return

    size = Path(path).stat().st_size
    info(f"Size: {size:,} bytes ({size//1024} KB)")

    r = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", path
    ], capture_output=True, text=True)

    data = json.loads(r.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    print(f"\n  Format: {fmt.get('format_long_name', '?')}")
    print(f"  Duration: {float(fmt.get('duration', 0)):.2f}s")
    print(f"  Bitrate: {int(fmt.get('bit_rate', 0))//1000} kbps")
    print(f"  Size: {int(fmt.get('size', 0))//1024} KB")

    for i, s in enumerate(streams):
        ct = s.get("codec_type", "?")
        cn = s.get("codec_name", "?")
        print(f"\n  Stream {i} [{ct}]: {cn}")

        if ct == "video":
            print(f"    Resolution: {s.get('width')}x{s.get('height')}")
            print(f"    Frame rate: {s.get('r_frame_rate')}")
            print(f"    Pixel fmt:  {s.get('pix_fmt')}")

        elif ct == "audio":
            sr = int(s.get("sample_rate", 0))
            ch = s.get("channels", 0)
            bits = s.get("bits_per_raw_sample") or s.get("bits_per_sample", "?")
            print(f"    Sample rate: {sr} Hz")
            print(f"    Channels: {ch} ({'mono' if ch==1 else 'stereo' if ch==2 else ch})")
            print(f"    Bit depth: {bits} bit")

            # Check if audio is silent (silent = very small file relative to duration)
            dur = float(fmt.get("duration", 1))
            expected_bytes = sr * ch * 2 * dur  # PCM 16-bit
            actual_bytes = int(fmt.get("size", 0))
            ratio = actual_bytes / max(expected_bytes, 1)
            if cn == "pcm_s16le" and ratio < 0.01:
                warn("This audio appears to be SILENT (anullsrc placeholder)")
                info("No TTS engine was available when this was generated")
                info("Fix: sudo apt install espeak-ng  OR  uv pip install TTS")
            elif cn == "pcm_s16le":
                ok("Audio has content (not silent)")

    # For MP4, check if video is a solid colour (placeholder)
    suffix = Path(path).suffix.lower()
    if suffix == ".mp4":
        r2 = subprocess.run([
            "ffmpeg", "-y", "-i", path,
            "-vf", "select=eq(n\\,0),signalstats",
            "-f", "null", "-"
        ], capture_output=True, text=True)
        if "YDIF" in r2.stderr:
            ydif_line = [l for l in r2.stderr.split("\n") if "YDIF" in l]
            if ydif_line:
                print(f"\n  Video signal: {ydif_line[0].strip()}")
                if "0.000000" in ydif_line[0]:
                    warn("Video is a SOLID COLOUR (FFmpeg placeholder — no animation engine)")
                    info("Fix: Install AnimateDiff via SD WebUI, or Deforum")
                    info("     SD WebUI: https://github.com/AUTOMATIC1111/ComfyUI")
                    info("     AnimateDiff: install as SD extension, enable --api flag")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# AGENT RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(name: str, genre: str = "", fresh: bool = False, verbose: bool = False):
    hdr(f"Running Agent: {name}")

    from pipeline.state import EpisodeState

    if fresh or not STATE_FILE.exists():
        state = EpisodeState(output_dir=OUTPUT_DIR)
        info("Starting with fresh state")
    else:
        state = load_state()
        state.errors = []  # clear errors from previous run

    t0 = time.time()

    if name == "trend":
        from agents.trend_researcher import TrendResearcher
        state = TrendResearcher().run(state, genre_hint=genre)

    elif name == "script":
        if not state.trend:
            err("No trend data. Run 'trend' agent first.")
            return
        from agents.script_writer import ScriptWriter
        state = ScriptWriter().run(state)

    elif name == "character":
        if not state.script:
            err("No script. Run 'script' agent first.")
            return
        from agents.story_character import StoryCharacterAgent
        state = StoryCharacterAgent().run(state, generate_images=False)

    elif name == "animation":
        if not state.characters:
            err("No character data. Run 'character' agent first.")
            return
        from agents.animation_agent import AnimationAgent
        state = AnimationAgent().run(state)

    elif name == "voiceover":
        if not state.script:
            err("No script. Run 'script' agent first.")
            return
        from agents.voiceover_agent import VoiceoverAgent
        state = VoiceoverAgent().run(state)

    elif name == "editor":
        from agents.editor_upload import EditorUploadAgent
        state = EditorUploadAgent().run(state, upload=False)

    else:
        err(f"Unknown agent: {name}")
        print("  Valid: trend, script, character, animation, voiceover, editor")
        return

    elapsed = time.time() - t0
    save_state(state)
    print_state(state, verbose=verbose)
    info(f"Agent finished in {elapsed:.1f}s")

    if state.errors:
        err(f"{len(state.errors)} error(s):")
        for e in state.errors:
            print(f"    {R}{e}{RST}")
    else:
        ok("No errors")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL — pause between agents for inspection
# ─────────────────────────────────────────────────────────────────────────────

def run_all(genre: str = "", auto: bool = False, verbose: bool = False):
    hdr("Full Pipeline — Step by Step")

    agents = ["trend", "script", "character", "animation", "voiceover", "editor"]

    for i, agent in enumerate(agents, 1):
        print(f"\n{B}[{i}/{len(agents)}] Agent: {agent}{RST}")

        run_agent(agent, genre=genre, fresh=(i == 1), verbose=verbose)

        if not auto and i < len(agents):
            try:
                answer = input(f"\n  {Y}Continue to '{agents[i]}'? [Y/n/q]: {RST}").strip().lower()
                if answer == "q":
                    info("Stopped.")
                    break
                elif answer == "n":
                    info(f"Skipping {agents[i]}")
                    continue
            except (KeyboardInterrupt, EOFError):
                print()
                info("Stopped.")
                break

    hdr("Pipeline Complete")
    state = load_state()
    print_state(state, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK AUDIO/VIDEO GENERATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_generation():
    hdr("Generation Capability Test")

    out = ROOT / "output" / "test_generation"
    out.mkdir(parents=True, exist_ok=True)

    # ── Test 1: FFmpeg colour card (baseline animation placeholder) ───────
    print(f"\n{C}1. FFmpeg colour card (animation baseline){RST}")
    clip = out / "test_clip.mp4"
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=0x1a1a2e:size=512x512:rate=24",
        "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(clip)
    ], capture_output=True)
    (ok if r.returncode == 0 else err)(f"Colour card: {'OK' if r.returncode == 0 else 'FAILED'}")
    if clip.exists():
        _probe_file_brief(str(clip))

    # ── Test 2: Silent WAV (no-TTS fallback) ─────────────────────────────
    print(f"\n{C}2. Silent WAV (no-TTS fallback){RST}")
    silent = out / "test_silent.wav"
    r2 = subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anullsrc=r=22050:cl=mono",
        "-t", "3", "-acodec", "pcm_s16le",
        str(silent)
    ], capture_output=True)
    (ok if r2.returncode == 0 else err)(f"Silent WAV: {'OK — but will be inaudible' if r2.returncode == 0 else 'FAILED'}")
    warn("This is the fallback when no TTS is installed — install espeak-ng for real speech")

    # ── Test 3: espeak TTS ────────────────────────────────────────────────
    print(f"\n{C}3. espeak-ng TTS{RST}")
    speech = out / "test_speech.wav"
    for binary in ["espeak-ng", "espeak"]:
        r3 = subprocess.run(
            [binary, "-w", str(speech), "Hello, this is a test of the cartoon pipeline voice system."],
            capture_output=True
        )
        if r3.returncode == 0 and speech.exists():
            ok(f"{binary}: OK")
            _probe_file_brief(str(speech))
            break
        else:
            warn(f"{binary}: not found")
    else:
        warn("No espeak available")
        info("Install: sudo apt install espeak-ng")

    # ── Test 4: Coqui TTS ─────────────────────────────────────────────────
    print(f"\n{C}4. Coqui XTTS v2{RST}")
    r4 = subprocess.run(["which", "tts"], capture_output=True)
    if r4.returncode == 0:
        ok("Coqui TTS installed")
        info("Run 'tts --list_models' to verify XTTS v2 is available")
    else:
        warn("Coqui TTS not installed")
        info("Install: uv pip install TTS  (requires ~4GB download on first run)")

    # ── Test 5: AnimateDiff / SD WebUI ────────────────────────────────────
    print(f"\n{C}5. ComfyUI / AnimateDiff{RST}")
    comfy_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
    try:
        import requests
        r5 = requests.get(f"{comfy_url}/system_stats", timeout=3)
        ok(f"ComfyUI running at {comfy_url}")

        r6 = requests.get(f"{comfy_url}/object_info/ADE_AnimateDiffLoaderWithContext", timeout=3)
        if r6.status_code == 200:
            ok("AnimateDiff-Evolved: available — real animation will be generated")
        else:
            warn("AnimateDiff-Evolved: NOT installed")
            info("Animation will fall back to FFmpeg colour cards")
            info("Install: git clone ComfyUI-AnimateDiff-Evolved")
    except Exception:
        warn(f"ComfyUI not running at at {comfy_url}")
        info("Animation will use FFmpeg colour-card placeholders")

    # ── Test 6: Final video merge ─────────────────────────────────────────
    print(f"\n{C}6. Final video assembly (clip + audio merge){RST}")
    if clip.exists() and silent.exists():
        merged = out / "test_final.mp4"
        r7 = subprocess.run([
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(clip),
            "-i", str(silent),
            "-c:v", "libx264", "-c:a", "aac",
            "-t", "3", "-pix_fmt", "yuv420p", "-shortest",
            str(merged)
        ], capture_output=True)
        (ok if r7.returncode == 0 else err)(f"Video + audio merge: {'OK' if r7.returncode == 0 else 'FAILED'}")
        if merged.exists():
            _probe_file_brief(str(merged))
    else:
        warn("Skipped — clip or audio not available")

    print(f"\n  Test files written to: {out}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cartoon Pipeline — Step-by-step debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python tools/test_pipeline.py --check
  uv run python tools/test_pipeline.py --test-gen
  uv run python tools/test_pipeline.py --agent trend --genre "dark comedy"
  uv run python tools/test_pipeline.py --agent script --verbose
  uv run python tools/test_pipeline.py --agent animation
  uv run python tools/test_pipeline.py --all --genre "sci-fi" --auto
  uv run python tools/test_pipeline.py --inspect
  uv run python tools/test_pipeline.py --probe output/EP_xxx/audio/scene_01.wav
  uv run python tools/test_pipeline.py --fresh   (reset saved state)
        """
    )

    parser.add_argument("--check",    action="store_true", help="Run environment diagnostic")
    parser.add_argument("--test-gen", action="store_true", help="Test audio/video generation capabilities")
    parser.add_argument("--agent",    type=str, metavar="NAME",
                        help="Run one agent: trend|script|character|animation|voiceover|editor")
    parser.add_argument("--all",      action="store_true", help="Run all agents step by step")
    parser.add_argument("--auto",     action="store_true", help="Don't pause between agents (use with --all)")
    parser.add_argument("--inspect",  action="store_true", help="Print current saved state")
    parser.add_argument("--probe",    type=str, metavar="FILE", help="Probe an audio/video file")
    parser.add_argument("--fresh",    action="store_true", help="Delete saved state and start fresh")
    parser.add_argument("--genre",    type=str, default="", help="Genre hint for trend/script agents")
    parser.add_argument("--verbose",  action="store_true", help="Show more detail in state output")

    args = parser.parse_args()

    if args.fresh:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            ok("Saved state cleared")
        else:
            info("No saved state to clear")

    if args.check:
        check_environment()

    elif args.test_gen:
        test_generation()

    elif args.probe:
        probe_file(args.probe)

    elif args.inspect:
        from pipeline.state import EpisodeState
        if STATE_FILE.exists():
            state = EpisodeState(**json.loads(STATE_FILE.read_text()))
            print_state(state, verbose=args.verbose)
        else:
            warn("No saved state found. Run an agent first.")

    elif args.agent:
        run_agent(args.agent, genre=args.genre, verbose=args.verbose)

    elif args.all:
        run_all(genre=args.genre, auto=args.auto, verbose=args.verbose)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
