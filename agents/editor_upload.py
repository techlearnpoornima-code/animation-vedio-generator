"""
agents/editor_upload.py
Agent 6 — Editor + SEO & Upload

Part A — Editor:
  - Stitches all scene clips together
  - Overlays audio per scene
  - Adds fade transitions between scenes
  - Renders final .mp4

Part B — SEO & Upload:
  - Uses Ollama to generate title, description, tags
  - Uploads to YouTube via Data API v3 (free tier)
"""

import os
import json
import subprocess
from pathlib import Path
from pipeline.state import EpisodeState
from pipeline.ollama_client import OllamaClient

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class EditorUploadAgent:

    def __init__(self):
        self.llm = OllamaClient(model=os.getenv("OLLAMA_MODEL_FAST", "llama3"))

    # ================================================================== #
    #  PART A — EDITOR                                                     #
    # ================================================================== #

    def _check_ffmpeg(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _get_audio_duration(self, audio_path: str) -> float:
        """Return audio file duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 4.0  # default clip length fallback

    def _sync_clip_to_audio(
        self,
        clip_path: str,
        audio_path: str | None,
        output_path: Path,
        default_duration: float = 4.0,
    ) -> str:
        """
        Combine a video clip with its audio track.
        If audio is longer than clip, loop/extend the video.
        If clip is longer, trim to audio duration.
        """
        duration = self._get_audio_duration(audio_path) if audio_path else default_duration

        if audio_path and Path(audio_path).exists():
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", clip_path,    # loop video
                "-i", audio_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-t", str(duration),
                "-pix_fmt", "yuv420p",
                "-shortest",
                str(output_path),
            ]
        else:
            # No audio — just set clip to default duration
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", clip_path,
                "-c:v", "libx264",
                "-t", str(duration),
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]

        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return str(output_path)

    def _add_transitions(self, synced_clips: list[str], output_dir: Path) -> list[str]:
        """Add 0.5s crossfade between consecutive clips using xfade filter."""
        if len(synced_clips) <= 1:
            return synced_clips

        transitioned = []
        current = synced_clips[0]

        for i, next_clip in enumerate(synced_clips[1:], 1):
            out = output_dir / f"transition_{i:02d}.mp4"
            try:
                # Get duration of current clip
                dur = self._get_audio_duration(current)
                offset = max(0, dur - 0.5)  # start xfade 0.5s before end

                cmd = [
                    "ffmpeg", "-y",
                    "-i", current, "-i", next_clip,
                    "-filter_complex",
                    f"[0:v][1:v]xfade=transition=fade:duration=0.5:offset={offset}[v];"
                    f"[0:a][1:a]acrossfade=d=0.5[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-c:a", "aac",
                    "-pix_fmt", "yuv420p",
                    str(out),
                ]
                subprocess.run(cmd, capture_output=True, check=True, timeout=120)
                current = str(out)
            except Exception as e:
                print(f"  [editor] Transition {i} failed, skipping: {e}")
                current = next_clip  # skip transition, use raw clip

            transitioned.append(current)

        return transitioned

    def _concat_final(self, clips: list[str], output_path: Path) -> str:
        """Concatenate all clips into the final episode video."""
        list_file = output_path.parent / "final_concat_list.txt"
        with open(list_file, "w") as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        list_file.unlink(missing_ok=True)
        return str(output_path)

    def _edit(self, state: EpisodeState, output_dir: Path) -> str | None:
        """Full editing pipeline: sync → transitions → concat."""
        if not self._check_ffmpeg():
            state.add_error("editor", "FFmpeg not found. Install it: sudo apt install ffmpeg")
            return None

        # Always use resolved absolute paths
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        clips = state.animation.clip_paths if state.animation else []
        audio_files = state.audio.audio_paths if state.audio else []

        # ── No clips at all ─────────────────────────────────────────────
        if not clips:
            # If we have audio, produce an audio-only mp4 with a black video track
            if audio_files:
                print("  No video clips — producing audio-only episode with black video...")
                return self._audio_only_video(audio_files, output_dir, state)
            state.add_error("editor", "No video clips and no audio to edit.")
            return None

        # ── Filter to clips that actually exist ─────────────────────────
        existing_clips = [c for c in clips if Path(c).exists()]
        if not existing_clips:
            print("  WARN: All clip paths missing from disk — producing audio-only fallback")
            if audio_files:
                return self._audio_only_video(audio_files, output_dir, state)
            state.add_error("editor", "No usable video clips found on disk.")
            return None

        sync_dir = output_dir / "synced"
        sync_dir.mkdir(exist_ok=True)

        synced_clips = []
        for i, clip in enumerate(existing_clips):
            audio = audio_files[i] if i < len(audio_files) else None
            # Skip audio paths that don't exist
            if audio and not Path(audio).exists():
                audio = None
            sync_out = sync_dir / f"synced_{i:02d}.mp4"
            print(f"  Syncing scene {i+1}/{len(existing_clips)}...", end=" ", flush=True)
            try:
                synced = self._sync_clip_to_audio(clip, audio, sync_out)
                synced_clips.append(synced)
                print("done")
            except Exception as e:
                print(f"WARN: {e}")
                synced_clips.append(clip)

        trans_dir = output_dir / "transitions"
        trans_dir.mkdir(exist_ok=True)
        print("  Adding transitions...")
        final_clips = self._add_transitions(synced_clips, trans_dir)

        final_path = output_dir / f"{state.episode_id}.mp4"
        print("  Concatenating final video...")
        return self._concat_final(final_clips if final_clips else synced_clips, final_path)

    def _audio_only_video(self, audio_files: list[str], output_dir: Path, state: EpisodeState) -> str | None:
        """
        Merge all audio files and pair with a black video track.
        Used when animation clips are missing but audio was generated.
        """
        output_dir = output_dir.resolve()
        existing_audio = [a for a in audio_files if Path(a).exists()]
        if not existing_audio:
            return None

        # Merge audio files into one
        merged_audio = output_dir / "merged_audio.wav"
        if len(existing_audio) == 1:
            import shutil
            shutil.copy(existing_audio[0], merged_audio)
        else:
            list_file = output_dir / "audio_concat_list.txt"
            with open(list_file, "w") as f:
                for a in existing_audio:
                    abs_a = str(Path(a).resolve()).replace("\\", "/")
                    f.write(f"file '{abs_a}'\n")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", str(merged_audio)
            ], check=True, capture_output=True)
            list_file.unlink(missing_ok=True)

        # Get audio duration
        duration = self._get_audio_duration(str(merged_audio))

        # Create black video + audio
        final_path = output_dir / f"{state.episode_id}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:size=512x512:rate=24",
            "-i", str(merged_audio),
            "-c:v", "libx264", "-c:a", "aac",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(final_path),
        ], check=True, capture_output=True)

        merged_audio.unlink(missing_ok=True)
        return str(final_path) if final_path.exists() else None

    # ================================================================== #
    #  PART B — SEO METADATA                                               #
    # ================================================================== #

    def _generate_seo_metadata(self, state: EpisodeState) -> dict:
        """Use Ollama to generate YouTube-optimized title, description, tags."""
        trend = state.trend
        script = state.script

        prompt = f"""Generate YouTube SEO metadata for a cartoon episode.

Episode title: {script.title}
Genre: {trend.genre}
Tone: {trend.tone}
Topic: {trend.topic}
Audience: {trend.target_audience}
Synopsis: {script.synopsis}

Return JSON:
{{
  "yt_title": "YouTube-optimized title under 70 chars",
  "description": "3-paragraph YouTube description (hook, episode summary, call-to-action). Under 500 chars total.",
  "tags": ["10 to 15 relevant tags as short strings"],
  "category_id": "1",
  "default_language": "en"
}}

Category IDs: Film & Animation=1, Comedy=23, Education=27"""

        return self.llm.chat_json(prompt, temperature=0.6)

    # ================================================================== #
    #  PART C — YOUTUBE UPLOAD                                             #
    # ================================================================== #

    def _get_youtube_client(self):
        """Authenticate and return YouTube API client (OAuth2)."""
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-api-python-client not installed.")

        creds = None
        token_file = Path("youtube_token.pickle")

        if token_file.exists():
            with open(token_file, "rb") as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "client_secrets.json", YOUTUBE_SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(token_file, "wb") as f:
                pickle.dump(creds, f)

        return build("youtube", "v3", credentials=creds)

    def _upload_to_youtube(self, video_path: str, metadata: dict, state: EpisodeState) -> str | None:
        """Upload the final video to YouTube."""
        try:
            youtube = self._get_youtube_client()
        except Exception as e:
            print(f"  [upload] YouTube auth failed: {e}")
            state.warnings.append(f"YouTube upload skipped: {e}")
            return None

        body = {
            "snippet": {
                "title": metadata.get("yt_title", state.script.title),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "categoryId": metadata.get("category_id", "1"),
                "defaultLanguage": metadata.get("default_language", "en"),
            },
            "status": {
                "privacyStatus": "private",  # start private, review before publishing
            },
        }

        media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"  Upload progress: {int(status.progress() * 100)}%")

        video_id = response.get("id")
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"  Uploaded (private): {url}")
        return url

    # ================================================================== #
    #  Public entrypoint                                                    #
    # ================================================================== #

    def run(self, state: EpisodeState, upload: bool = False) -> EpisodeState:
        print("\n[6/6] Editor & Upload Agent starting...")

        output_dir = Path(state.output_dir).resolve() / state.episode_id / "final"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Edit ---
        print("  Editing final video...")
        final_path = self._edit(state, output_dir)

        if final_path and Path(final_path).exists():
            state.final_video_path = final_path
            size_mb = Path(final_path).stat().st_size / (1024 * 1024)
            print(f"  Final video: {final_path} ({size_mb:.1f} MB)")
        else:
            print("  Editing failed or produced no output.")

        # --- SEO Metadata ---
        print("  Generating SEO metadata...")
        try:
            seo = self._generate_seo_metadata(state)

            meta_path = output_dir / "seo_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(seo, f, indent=2)
            print(f"  SEO metadata saved → {meta_path}")
            print(f"  YT Title: {seo.get('yt_title')}")

        except Exception as e:
            seo = {}
            state.warnings.append(f"SEO metadata generation failed: {e}")
            print(f"  SEO generation WARN: {e}")

        # --- YouTube Upload (optional) ---
        if upload and state.final_video_path:
            print("  Uploading to YouTube (private)...")
            try:
                url = self._upload_to_youtube(state.final_video_path, seo, state)
                if url:
                    state.youtube_url = url
            except Exception as e:
                state.warnings.append(f"YouTube upload failed: {e}")
                print(f"  Upload WARN: {e}")
        elif upload:
            print("  Upload skipped: no final video file.")

        state.mark_done("editor_upload")
        print("  [6/6] DONE\n")

        return state
