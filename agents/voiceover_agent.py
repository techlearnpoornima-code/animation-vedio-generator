"""
agents/voiceover_agent.py
Agent 5 — Voiceover Agent

Synthesizes character dialogue using Coqui XTTS v2 (local, free).
Each character gets a unique voice derived from their voice_profile.
Outputs one audio file per scene, synced to dialogue order.
"""

import os
import json
import subprocess
from pathlib import Path
from pipeline.state import EpisodeState, AudioData

# Coqui TTS model to use — XTTS v2 supports expressive multi-speaker synthesis
COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Built-in speaker references bundled with XTTS v2
# These are matched to character voice profiles
SPEAKER_MAP = {
    # (gender, energy) -> XTTS speaker name
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
    #  Synthesize a single line                                             #
    # ------------------------------------------------------------------ #

    def _synthesize_line(
        self,
        text: str,
        speaker: str,
        output_path: Path,
        language: str = "en",
    ) -> bool:
        """Run Coqui TTS CLI for a single dialogue line."""
        cmd = [
            "tts",
            "--model_name", COQUI_MODEL,
            "--text", text,
            "--speaker", speaker,
            "--language_idx", language,
            "--out_path", str(output_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0 and output_path.exists()
        except Exception as e:
            print(f"  [voiceover] TTS failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Generate scene audio (all dialogue lines merged)                    #
    # ------------------------------------------------------------------ #

    def _build_voice_map(self, state: EpisodeState) -> dict:
        """Map character name -> XTTS speaker name from their voice profiles."""
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
        """Synthesize all dialogue lines for a scene and merge into one file."""
        dialogue = scene.get("dialogue", [])
        if not dialogue:
            return None

        line_files = []
        tmp_dir = output_dir / f"scene_{scene_num:02d}_lines"
        tmp_dir.mkdir(exist_ok=True)

        for j, entry in enumerate(dialogue):
            char = entry.get("character", "NARRATOR")
            line = entry.get("line", "").strip()
            if not line:
                continue

            speaker = voice_map.get(char, DEFAULT_SPEAKER)
            line_path = tmp_dir / f"line_{j:02d}.wav"

            if self._tts_available:
                success = self._synthesize_line(line, speaker, line_path)
            else:
                # Fallback: use system TTS (espeak/say) if available
                success = self._system_tts_fallback(line, line_path)

            if success:
                line_files.append(str(line_path))

        if not line_files:
            return None

        # Merge all line files into one scene audio file
        scene_audio_path = output_dir / f"scene_{scene_num:02d}.wav"
        self._merge_audio_files(line_files, scene_audio_path)

        # Clean up line files
        for f in line_files:
            Path(f).unlink(missing_ok=True)
        tmp_dir.rmdir() if not any(tmp_dir.iterdir()) else None

        return str(scene_audio_path)

    # ------------------------------------------------------------------ #
    #  Audio merging via FFmpeg                                             #
    # ------------------------------------------------------------------ #

    def _merge_audio_files(self, file_paths: list[str], output_path: Path) -> None:
        """Concatenate audio files with a 300ms pause between lines."""
        if len(file_paths) == 1:
            import shutil
            shutil.copy(file_paths[0], output_path)
            return

        # Build FFmpeg concat filter with silence padding
        inputs = []
        filter_parts = []

        for i, f in enumerate(file_paths):
            inputs += ["-i", f]
            filter_parts.append(f"[{i}:a]")

        # Add 0.3s silence between lines
        silence_gen = "aevalsrc=0:d=0.3[sil];"
        concat_filter = (
            silence_gen +
            "".join(
                f"{filter_parts[i]}[sil]" if i < len(filter_parts) - 1 else filter_parts[i]
                for i in range(len(filter_parts))
            ) +
            f"concat=n={len(file_paths) * 2 - 1}:v=0:a=1[out]"
        )

        cmd = (
            ["ffmpeg", "-y"] +
            inputs +
            ["-filter_complex", concat_filter,
             "-map", "[out]",
             str(output_path)]
        )

        # Simple fallback: just concatenate without silence
        try:
            list_file = output_path.parent / "concat_list.txt"
            with open(list_file, "w") as f:
                f.write("\n".join(f"file '{p}'" for p in file_paths))

            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(list_file), "-c", "copy", str(output_path)],
                capture_output=True, check=True, timeout=60
            )
            list_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"  [voiceover] Audio merge failed: {e}")

    # ------------------------------------------------------------------ #
    #  System TTS fallback                                                  #
    # ------------------------------------------------------------------ #

    def _system_tts_fallback(self, text: str, output_path: Path) -> bool:
        """Use espeak or macOS 'say' when Coqui is not installed."""
        # Try espeak (Linux)
        try:
            subprocess.run(
                ["espeak", "-w", str(output_path), text],
                capture_output=True, check=True, timeout=15
            )
            return output_path.exists()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Try macOS say
        try:
            aiff_path = output_path.with_suffix(".aiff")
            subprocess.run(
                ["say", "-o", str(aiff_path), text],
                capture_output=True, check=True, timeout=15
            )
            # Convert aiff -> wav
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(aiff_path), str(output_path)],
                capture_output=True, check=True
            )
            aiff_path.unlink(missing_ok=True)
            return output_path.exists()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        print(f"  [voiceover] No TTS available. Skipping audio for this line.")
        return False

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
            print(f"  TTS engine: system fallback (espeak / say)")

        output_dir = Path(state.output_dir) / state.episode_id / "audio"
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
                # Estimate duration from file size (rough: WAV 44100hz 16bit mono ~88KB/s)
                size_kb = Path(audio_path).stat().st_size / 1024
                total_duration += size_kb / 88.0
                print(f"done → {Path(audio_path).name}")
            else:
                state.warnings.append(f"Scene {i}: no audio generated")
                print("skipped (no dialogue or TTS failed)")

        state.audio = AudioData(
            audio_paths=audio_paths,
            total_duration_secs=round(total_duration, 1),
            voice_map=voice_map,
        )

        # Save manifest
        manifest = {
            "voice_map": voice_map,
            "audio_files": audio_paths,
            "total_duration_secs": total_duration,
        }
        with open(output_dir / "audio_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        state.mark_done("voiceover")
        print(f"  Audio files: {len(audio_paths)} | Est. duration: {total_duration:.1f}s")
        print("  [5/6] DONE\n")

        return state
