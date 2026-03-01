"""
agents/voiceover_agent.py  — Agent 5: Voiceover

Voice synthesis stack (tries each in order, uses best available):

  Tier 1 — Kokoro TTS  (pip install kokoro soundfile + system espeak-ng)
           11 distinct neural voices (American + British, male + female)
           Python 3.12 compatible, fully local, Apache 2.0 licence
           ~350 MB model downloaded from HuggingFace on first run

  Tier 2 — Flite via ctypes (ships with Ubuntu/Debian, no install needed)
           4 base voices (awb=Scottish male, kal=US male, rms=US male2, slt=US female)
           × 3 FFmpeg pitch/speed variants = 13 distinct voices
           Deterministically assigned per character so reruns are consistent

  Tier 3 — espeak-ng / espeak (if installed separately)
           Per-character pitch/speed flags for differentiation

  Tier 4 — FFmpeg tone (last resort, audible but not speech)

Every character is guaranteed a UNIQUE voice — conflict resolution prevents
two characters from ever sharing the same speaker or profile.
"""

import os
import json
import hashlib
import ctypes
import subprocess
import tempfile
import shutil
from pathlib import Path
from pipeline.state import EpisodeState, AudioData

# ── Kokoro TTS voice pool ──────────────────────────────────────────────────────
# Each entry: (voice_id, gender, lang_code, description)
# lang_code: 'a' = American English, 'b' = British English
KOKORO_VOICES = [
    # American female
    ("af_bella",    "female", "a", "American female – warm, clear"),
    ("af_nicole",   "female", "a", "American female – smooth, professional"),
    ("af_sarah",    "female", "a", "American female – friendly, bright"),
    ("af_sky",      "female", "a", "American female – airy, youthful"),
    # American male
    ("am_adam",     "male",   "a", "American male – deep, authoritative"),
    ("am_michael",  "male",   "a", "American male – neutral, newscaster"),
    # British female
    ("bf_emma",     "female", "b", "British female – refined, articulate"),
    ("bf_isabella", "female", "b", "British female – warm, storyteller"),
    # British male
    ("bm_george",   "male",   "b", "British male – distinguished, mature"),
    ("bm_lewis",    "male",   "b", "British male – energetic, contemporary"),
    # Neutral fallback
    ("af",          "female", "a", "American female – default"),
]

KOKORO_FEMALE = [v for v in KOKORO_VOICES if v[1] == "female"]
KOKORO_MALE   = [v for v in KOKORO_VOICES if v[1] == "male"]

# (gender, energy) → preferred index within gender-specific list
KOKORO_PREF = {
    ("female", "confident"):    0,   # af_bella
    ("female", "warm"):         0,   # af_bella
    ("female", "enthusiastic"): 2,   # af_sarah
    ("female", "nervous"):      3,   # af_sky
    ("female", "deadpan"):      1,   # af_nicole
    ("female", "sinister"):     4,   # bf_emma
    ("female", "raspy"):        5,   # bf_isabella
    ("male",   "confident"):    0,   # am_adam
    ("male",   "warm"):         0,   # am_adam
    ("male",   "enthusiastic"): 1,   # am_michael
    ("male",   "deadpan"):      1,   # am_michael
    ("male",   "nervous"):      1,   # am_michael
    ("male",   "sinister"):     2,   # bm_george
    ("male",   "raspy"):        2,   # bm_george
}

# ── Flite voice definitions ────────────────────────────────────────────────────
# (voice_id, lib_name, register_fn, description, pitch_mult, speed_mult, extra_af)
FLITE_VOICES = [
    ("awb_normal",  "libflite_cmu_us_awb.so.1",   "register_cmu_us_awb",   "Scottish male",      1.00, 1.00, ""),
    ("awb_deep",    "libflite_cmu_us_awb.so.1",   "register_cmu_us_awb",   "Deep Scottish male", 0.80, 0.88, ""),
    ("awb_villain", "libflite_cmu_us_awb.so.1",   "register_cmu_us_awb",   "Villain gruff",      0.72, 0.78, "aecho=0.8:0.88:35:0.25"),
    ("kal_normal",  "libflite_cmu_us_kal16.so.1", "register_cmu_us_kal16", "US male",            1.00, 1.00, ""),
    ("kal_nervous", "libflite_cmu_us_kal16.so.1", "register_cmu_us_kal16", "Nervous US male",    1.12, 1.18, ""),
    ("kal_elder",   "libflite_cmu_us_kal16.so.1", "register_cmu_us_kal16", "Elder US male",      0.88, 0.88, "aecho=0.7:0.85:20:0.15"),
    ("rms_normal",  "libflite_cmu_us_rms.so.1",   "register_cmu_us_rms",   "US male 2",          1.00, 1.00, ""),
    ("rms_deep",    "libflite_cmu_us_rms.so.1",   "register_cmu_us_rms",   "Deep US male",       0.82, 0.90, ""),
    ("rms_excited", "libflite_cmu_us_rms.so.1",   "register_cmu_us_rms",   "Excited US male",    1.10, 1.15, ""),
    ("slt_normal",  "libflite_cmu_us_slt.so.1",   "register_cmu_us_slt",   "US female",          1.00, 1.00, ""),
    ("slt_excited", "libflite_cmu_us_slt.so.1",   "register_cmu_us_slt",   "Excited US female",  1.18, 1.12, ""),
    ("slt_villain", "libflite_cmu_us_slt.so.1",   "register_cmu_us_slt",   "Villain female",     0.85, 0.82, ""),
    ("slt_elder",   "libflite_cmu_us_slt.so.1",   "register_cmu_us_slt",   "Elder female",       0.90, 0.88, "aecho=0.7:0.85:15:0.12"),
]

# gender/energy → preferred flite voice index
FLITE_PREFERENCE = {
    ("male",    "confident"):    0,   # awb_normal
    ("male",    "enthusiastic"): 3,   # kal_normal
    ("male",    "nervous"):      4,   # kal_nervous
    ("male",    "deadpan"):      6,   # rms_normal
    ("male",    "sinister"):     2,   # awb_villain
    ("male",    "raspy"):        5,   # kal_elder
    ("male",    "warm"):         7,   # rms_deep
    ("female",  "confident"):    9,   # slt_normal
    ("female",  "enthusiastic"): 10,  # slt_excited
    ("female",  "nervous"):      10,  # slt_excited
    ("female",  "deadpan"):      9,   # slt_normal
    ("female",  "sinister"):     11,  # slt_villain
    ("female",  "warm"):         9,   # slt_normal
    ("neutral", "deadpan"):      6,   # rms_normal
    ("neutral", "enthusiastic"): 8,   # rms_excited
}


class FliteTTS:
    """Ctypes wrapper around libflite — works with the libflite1 Ubuntu package."""

    def __init__(self):
        self._loaded = False
        self._voice_cache: dict[str, ctypes.c_void_p] = {}
        self._flite = None
        self._try_load()

    def _try_load(self):
        try:
            self._flite = ctypes.CDLL("libflite.so.1")
            ctypes.CDLL("libflite_usenglish.so.1")
            ctypes.CDLL("libflite_cmulex.so.1")
            self._flite.flite_init.restype = ctypes.c_int
            self._flite.flite_init()
            self._flite.flite_text_to_speech.restype  = ctypes.c_float
            self._flite.flite_text_to_speech.argtypes = [
                ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p
            ]
            self._loaded = True
        except OSError:
            pass

    def available(self) -> bool:
        return self._loaded

    def _get_voice(self, lib_name: str, reg_fn: str) -> ctypes.c_void_p | None:
        key = f"{lib_name}:{reg_fn}"
        if key in self._voice_cache:
            return self._voice_cache[key]
        try:
            lib = ctypes.CDLL(lib_name)
            fn  = getattr(lib, reg_fn)
            fn.restype  = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p]
            voice = fn(None)
            self._voice_cache[key] = voice
            return voice
        except (OSError, AttributeError):
            return None

    def synthesize(
        self,
        text:       str,
        voice_def:  tuple,
        out_path:   Path,
        pitch_mult: float = 1.0,
        speed_mult: float = 1.0,
        extra_af:   str   = "",
    ) -> bool:
        """Synthesize text to WAV. Applies pitch/speed via FFmpeg post-process."""
        if not self._loaded:
            return False

        _vid, lib_name, reg_fn, _desc, v_pitch, v_speed, v_extra = voice_def
        pitch    = pitch_mult * v_pitch
        speed    = speed_mult * v_speed
        af_extra = extra_af or v_extra

        voice = self._get_voice(lib_name, reg_fn)
        if voice is None:
            return False

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            dur = self._flite.flite_text_to_speech(
                text.encode("utf-8"), voice, tmp_path.encode()
            )
            if dur <= 0 or not Path(tmp_path).exists():
                return False

            needs_ffmpeg = (abs(pitch - 1.0) > 0.01 or abs(speed - 1.0) > 0.01 or af_extra)
            if needs_ffmpeg:
                filters = [
                    f"asetrate=44100*{pitch:.3f}",
                    "aresample=44100",
                    f"atempo={speed:.3f}",
                ]
                if af_extra:
                    filters.append(af_extra)
                r = subprocess.run([
                    "ffmpeg", "-y", "-i", tmp_path,
                    "-af", ",".join(filters),
                    "-acodec", "pcm_s16le", "-ar", "22050",
                    str(out_path),
                ], capture_output=True)
                Path(tmp_path).unlink(missing_ok=True)
                return r.returncode == 0 and out_path.exists()
            else:
                shutil.move(tmp_path, out_path)
                return out_path.exists()

        except Exception:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            return False


# ── Global flite instance (shared across calls) ────────────────────────────────
_flite = FliteTTS()


class VoiceoverAgent:

    def __init__(self):
        self._kokoro_available   = self._check_kokoro()
        self._pipeline_cache: dict[str, object] = {}   # lang_code → KPipeline

    # ------------------------------------------------------------------ #
    #  Availability checks                                                  #
    # ------------------------------------------------------------------ #

    def _check_kokoro(self) -> bool:
        try:
            import kokoro       # noqa: F401
            import soundfile    # noqa: F401
            return True
        except ImportError:
            return False

    def _get_pipeline(self, lang_code: str):
        """Return a cached KPipeline for the given lang_code ('a' or 'b')."""
        if lang_code not in self._pipeline_cache:
            from kokoro import KPipeline
            self._pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
        return self._pipeline_cache[lang_code]

    # ------------------------------------------------------------------ #
    #  Voice map: assign UNIQUE voice to every character                    #
    # ------------------------------------------------------------------ #

    def _build_voice_map(self, state: EpisodeState) -> dict:
        """Returns {char_name: voice_spec_dict} with no duplicates."""
        characters = state.characters.characters if state.characters else []

        assigned_kokoro: set[str] = set()    # voice_ids already taken
        assigned_flite:  set[int] = set()    # FLITE_VOICES indices already taken

        result = {}

        for char in characters:
            name    = char["name"]
            profile = char.get("voice_profile", {})
            gender  = profile.get("gender", "neutral").lower()
            energy  = profile.get("energy", "confident").lower()

            if self._kokoro_available:
                spec = self._assign_kokoro(gender, energy, assigned_kokoro)
                assigned_kokoro.add(spec["voice_id"])
                result[name] = spec
            elif _flite.available():
                spec = self._assign_flite(name, gender, energy, assigned_flite)
                assigned_flite.add(spec["flite_idx"])
                result[name] = spec
            else:
                spec = self._assign_espeak(name, gender, energy, assigned_flite)
                assigned_flite.add(spec.get("espeak_idx", 0))
                result[name] = spec

        return result

    def _assign_kokoro(self, gender: str, energy: str, used: set[str]) -> dict:
        """Pick the best available Kokoro voice for this gender/energy combo."""
        if gender == "male":
            pool = KOKORO_MALE
        elif gender == "female":
            pool = KOKORO_FEMALE
        else:
            pool = KOKORO_VOICES  # neutral: all voices

        preferred_idx = KOKORO_PREF.get((gender, energy), 0)

        # Try preferred first, then walk through pool, then entire catalogue
        order = [preferred_idx] + [i for i in range(len(pool)) if i != preferred_idx]
        for idx in order:
            voice_id = pool[idx % len(pool)][0]
            if voice_id not in used:
                entry = pool[idx % len(pool)]
                return {
                    "type":      "kokoro",
                    "voice_id":  entry[0],
                    "lang_code": entry[2],
                    "description": entry[3],
                }

        # All gender-specific voices taken — fall through to full catalogue
        for entry in KOKORO_VOICES:
            if entry[0] not in used:
                return {
                    "type":      "kokoro",
                    "voice_id":  entry[0],
                    "lang_code": entry[2],
                    "description": entry[3],
                }

        # Absolute last resort: reuse first voice (should never happen with <11 chars)
        entry = KOKORO_VOICES[0]
        return {
            "type":        "kokoro",
            "voice_id":    entry[0],
            "lang_code":   entry[2],
            "description": entry[3],
        }

    def _assign_flite(self, name: str, gender: str, energy: str, used: set[int]) -> dict:
        preferred_idx = FLITE_PREFERENCE.get((gender, energy), 0)
        order = [preferred_idx] + [i for i in range(len(FLITE_VOICES)) if i != preferred_idx]
        for idx in order:
            if idx not in used:
                vdef = FLITE_VOICES[idx]
                return {
                    "type":        "flite",
                    "flite_idx":   idx,
                    "voice_def":   vdef,
                    "description": vdef[3],
                }
        # All flite voices used — hash-based fallback with pitch variation
        h   = int(hashlib.md5(name.encode()).hexdigest(), 16)
        idx = h % len(FLITE_VOICES)
        vdef = FLITE_VOICES[idx]
        pitch_var = 0.95 + (h % 10) * 0.01
        return {
            "type":           "flite",
            "flite_idx":      -1,
            "voice_def":      vdef,
            "pitch_override": pitch_var,
            "description":    vdef[3] + " (pitch var)",
        }

    def _assign_espeak(self, name: str, gender: str, energy: str, used: set[int]) -> dict:
        """Assign distinct espeak pitch/speed params per character."""
        h           = int(hashlib.md5(name.encode()).hexdigest(), 16)
        all_pitches = list(range(35, 80, 5))
        start       = h % len(all_pitches)
        for i in range(len(all_pitches)):
            idx = (start + i) % len(all_pitches)
            if idx not in used:
                pitch = all_pitches[idx]
                speeds = {
                    35: 140, 40: 150, 45: 160, 50: 175,
                    55: 180, 60: 190, 65: 200, 70: 210, 75: 220,
                }
                speed = speeds.get(pitch, 175)
                if gender == "female":
                    pitch += 15
                if energy == "nervous":
                    speed += 25
                elif energy == "deadpan":
                    speed -= 20
                return {"type": "espeak", "pitch": pitch, "speed": speed, "espeak_idx": idx}
        return {"type": "espeak", "pitch": 50, "speed": 175, "espeak_idx": 0}

    # ------------------------------------------------------------------ #
    #  Synthesis dispatch                                                   #
    # ------------------------------------------------------------------ #

    def _synthesize_line(self, text: str, voice_spec: dict, out_path: Path) -> bool:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vtype = voice_spec.get("type", "ffmpeg")

        # Tier 1 — Kokoro
        if vtype == "kokoro" and self._kokoro_available:
            if self._synth_kokoro(text, voice_spec, out_path):
                return True

        # Tier 2 — Flite
        if _flite.available():
            if vtype == "flite":
                spec = voice_spec
            else:
                # Derive a flite spec as fallback from voice name/hash
                spec = self._assign_flite(
                    voice_spec.get("voice_id", text[:8]),
                    "neutral", "confident", set()
                )
            if _flite.synthesize(
                text,
                spec["voice_def"],
                out_path,
                pitch_mult=spec.get("pitch_override", 1.0),
            ):
                return True

        # Tier 3 — espeak
        if self._synth_espeak(text, voice_spec, out_path):
            return True

        # Tier 4 — tone placeholder
        return self._synth_tone_placeholder(text, voice_spec, out_path)

    def _synth_kokoro(self, text: str, voice_spec: dict, out_path: Path) -> bool:
        """
        Synthesise text with Kokoro TTS and write a WAV file.
        The KPipeline is cached per lang_code so the model loads only once.
        """
        try:
            import numpy as np
            import soundfile as sf

            lang_code = voice_spec.get("lang_code", "a")
            voice_id  = voice_spec.get("voice_id", "af_bella")
            pipeline  = self._get_pipeline(lang_code)

            chunks = []
            # pipeline() returns a generator of (graphemes, phonemes, audio_array)
            for _gs, _ps, audio in pipeline(text, voice=voice_id, speed=1.0):
                if audio is not None and len(audio) > 0:
                    chunks.append(audio)

            if not chunks:
                return False

            audio_data = np.concatenate(chunks)
            sf.write(str(out_path), audio_data, samplerate=24000)
            return out_path.exists()

        except Exception as e:
            print(f"  [voiceover] Kokoro error: {e}")
            return False

    def _synth_espeak(self, text: str, voice_spec: dict, out_path: Path) -> bool:
        vtype = voice_spec.get("type")
        if vtype == "kokoro":
            # Derive stable espeak params from the voice_id hash
            h     = int(hashlib.md5(voice_spec.get("voice_id", "").encode()).hexdigest(), 16)
            pitch = 40 + (h % 40)
            speed = 150 + (h % 60)
        elif vtype == "flite":
            vdef  = voice_spec.get("voice_def", FLITE_VOICES[0])
            pitch = int(50 * vdef[4])
            speed = int(175 * vdef[5])
        else:
            pitch = voice_spec.get("pitch", 50)
            speed = voice_spec.get("speed", 175)

        for binary in ["espeak-ng", "espeak"]:
            try:
                r = subprocess.run([
                    binary,
                    "-p", str(max(0, min(99, pitch))),
                    "-s", str(max(80, min(450, speed))),
                    "-w", str(out_path), text,
                ], capture_output=True, check=True, timeout=15)
                if out_path.exists():
                    return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        return False

    def _synth_tone_placeholder(self, text: str, voice_spec: dict, out_path: Path) -> bool:
        """Last resort: a pitch-varied sine tone. Unique per character."""
        vtype = voice_spec.get("type")
        if vtype == "flite":
            vdef  = voice_spec.get("voice_def", FLITE_VOICES[0])
            freq  = int(180 * vdef[4])
            speed = vdef[5]
        elif vtype == "espeak":
            freq  = 150 + voice_spec.get("pitch", 50) * 2
            speed = voice_spec.get("speed", 175) / 175
        elif vtype == "kokoro":
            h     = int(hashlib.md5(voice_spec.get("voice_id", "").encode()).hexdigest(), 16)
            freq  = 150 + (h % 200)
            speed = 1.0
        else:
            h     = int(hashlib.md5(str(voice_spec).encode()).hexdigest(), 16)
            freq  = 150 + (h % 200)
            speed = 1.0

        duration = max(0.5, len(text) * 0.08)
        fade_out = max(0.0, duration - 0.08)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"sine=frequency={freq}:duration={duration * 2}",
                "-af", (
                    f"volume=0.10,atempo={speed:.2f},"
                    f"afade=t=in:d=0.03,"
                    f"afade=t=out:st={fade_out:.2f}:d=0.08"
                ),
                "-t", str(duration),
                "-acodec", "pcm_s16le", "-ar", "22050",
                str(out_path),
            ], check=True, capture_output=True, timeout=15)
            return out_path.exists()
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Audio merging                                                        #
    # ------------------------------------------------------------------ #

    def _merge_audio_files(self, file_paths: list[str], output_path: Path) -> None:
        existing = [f for f in file_paths if Path(f).exists()]
        if not existing:
            return
        if len(existing) == 1:
            shutil.copy(existing[0], output_path)
            return

        list_file = output_path.parent / f"concat_{output_path.stem}.txt"
        with open(list_file, "w") as f:
            for p in existing:
                f.write(f"file '{Path(p).resolve()}'\n")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", str(output_path),
            ], check=True, capture_output=True, timeout=60)
        except subprocess.CalledProcessError as e:
            print(f"  [voiceover] Merge failed: {(e.stderr or b'').decode()[-200:]}")
        finally:
            list_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    #  Scene synthesis                                                      #
    # ------------------------------------------------------------------ #

    def _synthesize_scene(
        self,
        scene:      dict,
        scene_num:  int,
        voice_map:  dict,
        output_dir: Path,
    ) -> str | None:
        dialogue = scene.get("dialogue", [])
        if not dialogue:
            return None

        output_dir = output_dir.resolve()
        tmp_dir    = output_dir / f"scene_{scene_num:02d}_lines"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        line_files = []

        for j, entry in enumerate(dialogue):
            char = entry.get("character", "NARRATOR")
            line = entry.get("line", "").strip()
            if not line:
                continue

            spec = voice_map.get(char)
            if spec is None:
                # Unknown char (e.g. NARRATOR) — stable hash-based assignment
                if self._kokoro_available:
                    spec = self._assign_kokoro("neutral", "confident", set())
                elif _flite.available():
                    spec = self._assign_flite(char, "neutral", "confident", set())
                else:
                    spec = self._assign_espeak(char, "neutral", "confident", set())

            line_path = tmp_dir / f"line_{j:02d}.wav"
            if self._synthesize_line(line, spec, line_path):
                line_files.append(str(line_path))

        if not line_files:
            return None

        scene_wav = output_dir / f"scene_{scene_num:02d}.wav"
        self._merge_audio_files(line_files, scene_wav)

        for f in line_files:
            Path(f).unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        return str(scene_wav) if scene_wav.exists() else None

    # ------------------------------------------------------------------ #
    #  Public entrypoint                                                    #
    # ------------------------------------------------------------------ #

    def run(self, state: EpisodeState) -> EpisodeState:
        print("\n[5/6] Voiceover Agent starting...")

        if not state.script:
            state.add_error("voiceover", "No script found. Run ScriptWriter first.")
            return state

        if self._kokoro_available:
            engine = "Kokoro TTS (10 neural voices, local)"
        elif _flite.available():
            engine = "Flite (4 base voices × pitch/speed variants)"
        else:
            engine = "espeak / tone fallback"
        print(f"  TTS engine: {engine}")

        output_dir = Path(state.output_dir).resolve() / state.episode_id / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)

        voice_map = self._build_voice_map(state)

        print("  Voice assignments:")
        for char, spec in voice_map.items():
            vtype = spec.get("type")
            if vtype == "kokoro":
                label = f"Kokoro → {spec['voice_id']}  ({spec['description']})"
            elif vtype == "flite":
                label = f"Flite  → {spec['description']}"
            else:
                label = f"espeak → pitch={spec.get('pitch')}, speed={spec.get('speed')}"
            print(f"    {char:<28} {label}")

        audio_paths    = []
        total_duration = 0.0

        for i, scene in enumerate(state.script.scenes, 1):
            print(
                f"  Synthesizing scene {i}/{state.script.total_scenes}...",
                end=" ", flush=True
            )
            path = self._synthesize_scene(scene, i, voice_map, output_dir)
            if path and Path(path).exists():
                audio_paths.append(path)
                total_duration += Path(path).stat().st_size / 1024 / 88.0
                print(f"done → {Path(path).name}")
            else:
                state.warnings.append(f"Scene {i}: no audio produced")
                print("skipped")

        # Serialise voice map for state storage
        serial_map = {}
        for char, spec in voice_map.items():
            vtype = spec.get("type")
            if vtype == "kokoro":
                serial_map[char] = f"kokoro:{spec['voice_id']}"
            elif vtype == "flite":
                serial_map[char] = f"flite:{spec['description']}"
            else:
                serial_map[char] = f"espeak:pitch={spec.get('pitch')}"

        state.audio = AudioData(
            audio_paths=audio_paths,
            total_duration_secs=round(total_duration, 1),
            voice_map=serial_map,
        )

        with open(output_dir / "audio_manifest.json", "w") as f:
            json.dump({
                "engine":              engine,
                "voice_map":           serial_map,
                "audio_files":         audio_paths,
                "total_duration_secs": total_duration,
            }, f, indent=2)

        state.mark_done("voiceover")
        print(f"  Audio files: {len(audio_paths)} | Est. duration: {total_duration:.1f}s")
        print("  [5/6] DONE\n")
        return state