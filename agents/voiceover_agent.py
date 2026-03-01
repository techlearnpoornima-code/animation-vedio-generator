"""
agents/voiceover_agent.py
Agent 5 — Voiceover Agent

Synthesizes character dialogue using Coqui XTTS v2 (local, free).
Each character gets a unique voice derived from their voice_profile.

Fallback chain: Coqui XTTS v2 → espeak-ng/espeak → macOS say → silent WAV placeholder
All paths use resolved absolute paths to avoid cwd="/" issues in server context.
"""

import os
import json
import subprocess
from pathlib import Path
from pipeline.state import EpisodeState, AudioData

COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

SPEAKER_MAP = {
    ("male",    "confident"):    "Damien Black",
    ("male",    "nervous"):      "Viktor Eka",
    ("male",    "deadpan"):      "Abrahan Mack",
    ("male",    "sinister"):     "Baldur Svensson",
    ("male",    "enthusiastic"): "Gilberto Mathias",
    ("female",  "confident"):    "Ana Florence",
    ("female",  "nervous"):      "Annmarie Nele",
    ("female",  "deadpan"):      "Asya Anara",
    ("female",  "sinister"):     "Henriette Usha",
    ("female",  "enthusiastic"): "Claribel Dervla",
    ("neutral", "deadpan"):      "Tammie Ema",
    ("neutral", "enthusiastic"): "Eugenio Mataracı",
}
DEFAULT_SPEAKER = "Ana Florence"


def _pick_speaker(voice_profile: dict) -> str:
    gender = voice_profile.get("gender", "neutral")
    energy = voice_profile.get("energy", "confident")
    return SPEAKER_MAP.get((gender, energy), DEFAULT_SPEAKER)


class VoiceoverAgent:

    def __init__(self):
        self._tts_available = self._check_tts()

    def _check_tts(self) -> bool:
        try:
            result = subprocess.run(
                ["tts", "--list_models"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ------------------------------------------------------------------ #
    #  Synthesize a single line via Coqui                                   #
    # ------------------------------------------------------------------ #

    def _synthesize_line(self, text: str, speaker: str, output_path: Path, language: str = "en") -> bool:
        output_path = output_path.resolve()
        cmd = [
            "tts", "--model_name", COQUI_MODEL,
            "--text", text,
            "--speaker", speaker,
            "--language_idx", language,
            "--out_path", str(output_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0 and output_path.exists()
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  System TTS fallbacks                                                 #
    # ------------------------------------------------------------------ #

    def _system_tts_fallback(self, text: str, output_path: Path) -> bool:
        """Try espeak-ng, espeak, then macOS say."""
        output_path = output_path.resolve()

        for binary in ["espeak-ng", "espeak"]:
            try:
                subprocess.run(
                    [binary, "-w", str(output_path), text],
                    capture_output=True, check=True, timeout=15
                )
                if output_path.exists():
                    return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

        try:
            aiff_path = output_path.with_suffix(".aiff")
            subprocess.run(["say", "-o", str(aiff_path), text],
                           capture_output=True, check=True, timeout=15)
            subprocess.run(["ffmpeg", "-y", "-i", str(aiff_path), str(output_path)],
                           capture_output=True, check=True)
            aiff_path.unlink(missing_ok=True)
            if output_path.exists():
                return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        return False

    def _generate_silent_wav(self, duration_secs: float, output_path: Path) -> bool:
        """
        Generate an audible tone placeholder when no TTS engine is available.
        Uses a soft 440Hz sine wave with fade in/out so it's clearly a placeholder
        (audible) rather than true silence (impossible to distinguish from failure).
        Duration is estimated from dialogue text length (~0.08s per character).
        """
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.5, duration_secs)
        fade_out_start = max(0.0, duration - 0.1)
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"sine=frequency=440:duration={duration}",
                "-af", f"volume=0.08,afade=t=in:d=0.05,afade=t=out:st={fade_out_start:.2f}:d=0.1",
                "-acodec", "pcm_s16le",
                "-ar", "22050",
                str(output_path),
            ], check=True, capture_output=True, timeout=15)
            return output_path.exists()
        except Exception as e:
            print(f"  [voiceover] Tone placeholder generation failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Audio merging                                                        #
    # ------------------------------------------------------------------ #

    def _merge_audio_files(self, file_paths: list[str], output_path: Path) -> None:
        """
        Concatenate WAV files. Only includes files that actually exist on disk.
        Uses FFmpeg concat demuxer with absolute paths in the list file.
        """
        output_path = output_path.resolve()
        # Filter to files that genuinely exist
        existing = [f for f in file_paths if Path(f).exists()]

        if not existing:
            print(f"  [voiceover] No audio line files to merge for {output_path.name}")
            return

        if len(existing) == 1:
            import shutil
            shutil.copy(existing[0], output_path)
            return

        # Write concat list with absolute paths
        list_file = output_path.parent / f"concat_{output_path.stem}.txt"
        with open(list_file, "w") as f:
            for p in existing:
                # FFmpeg concat requires forward slashes and absolute paths
                abs_p = str(Path(p).resolve()).replace("\\", "/")
                f.write(f"file '{abs_p}'\n")

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output_path),
            ], check=True, capture_output=True, timeout=60)
        except subprocess.CalledProcessError as e:
            print(f"  [voiceover] Merge failed: {e.stderr[-200:].decode() if e.stderr else e}")
        finally:
            list_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    #  Scene audio synthesis                                                #
    # ------------------------------------------------------------------ #

    def _build_voice_map(self, state: EpisodeState) -> dict:
        voice_map = {}
        if not state.characters:
            return voice_map
        for char in state.characters.characters:
            name = char["name"]
            profile = char.get("voice_profile", {})
            voice_map[name] = _pick_speaker(profile)
        return voice_map

    def _synthesize_scene_audio(
        self,
        scene: dict,
        scene_num: int,
        voice_map: dict,
        output_dir: Path,
    ) -> str | None:
        dialogue = scene.get("dialogue", [])
        if not dialogue:
            return None

        output_dir = output_dir.resolve()
        tmp_dir = output_dir / f"scene_{scene_num:02d}_lines"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        line_files = []

        for j, entry in enumerate(dialogue):
            char = entry.get("character", "NARRATOR")
            line = entry.get("line", "").strip()
            if not line:
                continue

            speaker = voice_map.get(char, DEFAULT_SPEAKER)
            line_path = tmp_dir / f"line_{j:02d}.wav"
            success = False

            if self._tts_available:
                success = self._synthesize_line(line, speaker, line_path)
            if not success:
                success = self._system_tts_fallback(line, line_path)
            if not success:
                # Silent placeholder sized to estimated speech duration
                duration = max(0.8, len(line) * 0.08)
                success = self._generate_silent_wav(duration, line_path)

            if success and line_path.exists():
                line_files.append(str(line_path))

        if not line_files:
            return None

        scene_audio_path = output_dir / f"scene_{scene_num:02d}.wav"
        self._merge_audio_files(line_files, scene_audio_path)

        # Clean up individual line files
        for f in line_files:
            Path(f).unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        return str(scene_audio_path) if scene_audio_path.exists() else None

    # ------------------------------------------------------------------ #
    #  Public entrypoint                                                    #
    # ------------------------------------------------------------------ #

    def run(self, state: EpisodeState) -> EpisodeState:
        print("\n[5/6] Voiceover Agent starting...")

        if not state.script:
            state.add_error("voiceover", "No script found. Run ScriptWriter first.")
            return state

        if self._tts_available:
            print(f"  TTS engine: Coqui XTTS v2")
        else:
            print(f"  TTS engine: system fallback (espeak / say / silent placeholder)")

        # Always resolve output_dir to absolute path
        output_dir = Path(state.output_dir).resolve() / state.episode_id / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)

        voice_map = self._build_voice_map(state)
        print(f"  Voice assignments: {voice_map}")

        audio_paths = []
        total_duration = 0.0

        for i, scene in enumerate(state.script.scenes, 1):
            print(f"  Synthesizing scene {i}/{state.script.total_scenes}...", end=" ", flush=True)
            audio_path = self._synthesize_scene_audio(scene, i, voice_map, output_dir)

            if audio_path and Path(audio_path).exists():
                audio_paths.append(audio_path)
                size_kb = Path(audio_path).stat().st_size / 1024
                total_duration += size_kb / 88.0
                print(f"done → {Path(audio_path).name}")
            else:
                state.warnings.append(f"Scene {i}: no audio generated")
                print("skipped")

        state.audio = AudioData(
            audio_paths=audio_paths,
            total_duration_secs=round(total_duration, 1),
            voice_map=voice_map,
        )

        with open(output_dir / "audio_manifest.json", "w") as f:
            json.dump({
                "voice_map": voice_map,
                "audio_files": audio_paths,
                "total_duration_secs": total_duration,
            }, f, indent=2)

        state.mark_done("voiceover")
        print(f"  Audio files: {len(audio_paths)} | Est. duration: {total_duration:.1f}s")
        print("  [5/6] DONE\n")

        return state
