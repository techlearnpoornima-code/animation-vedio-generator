"""
agents/animation_agent.py
Agent 4 — Animation Agent

Converts scene prompts + character refs into video clips.
Primary: AnimateDiff via SD WebUI API (local, free)
Fallback: Deforum script via SD WebUI
Output: one .mp4 clip per scene saved to output/clips/
"""

import os
import json
import time
import base64
import requests
import subprocess
from pathlib import Path
from pipeline.state import EpisodeState, AnimationData

SD_API_URL = os.getenv("SD_API_URL", "http://localhost:7860")
FPS = 24
CLIP_DURATION_SECS = 4      # default clip length per scene
RESOLUTION = "512x512"


class AnimationAgent:

    def __init__(self):
        self.sd_url = SD_API_URL
        self.fps = FPS

    # ------------------------------------------------------------------ #
    #  SD availability check                                               #
    # ------------------------------------------------------------------ #

    def _sd_available(self) -> bool:
        try:
            resp = requests.get(f"{self.sd_url}/sdapi/v1/options", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _animatediff_available(self) -> bool:
        """Check if AnimateDiff extension is loaded in SD WebUI."""
        try:
            resp = requests.get(f"{self.sd_url}/animatediff/v1/status", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  AnimateDiff clip generation                                          #
    # ------------------------------------------------------------------ #

    def _generate_clip_animatediff(self, prompt: str, scene_num: int, output_dir: Path) -> str | None:
        """Generate animated clip using AnimateDiff SD extension."""
        num_frames = self.fps * CLIP_DURATION_SECS

        payload = {
            "prompt": prompt,
            "negative_prompt": (
                "nsfw, blurry, low quality, deformed, extra limbs, "
                "static, frozen, no motion, ugly, text, watermark"
            ),
            "steps": 20,
            "cfg_scale": 7.5,
            "width": 512,
            "height": 512,
            "animatediff_enabled": True,
            "animatediff_model": "mm_sd_v15_v2.ckpt",
            "animatediff_motion_scale": 1.0,
            "animatediff_video_length": num_frames,
            "animatediff_fps": self.fps,
            "animatediff_loop_number": 0,
            "animatediff_closed_loop": "R-P",
        }

        resp = requests.post(
            f"{self.sd_url}/animatediff/v1/txt2gif",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        # AnimateDiff returns base64 encoded gif/mp4
        video_b64 = data.get("video") or data.get("images", [None])[0]
        if not video_b64:
            return None

        out_path = output_dir / f"scene_{scene_num:02d}.mp4"
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(video_b64))

        return str(out_path)

    # ------------------------------------------------------------------ #
    #  Deforum fallback                                                     #
    # ------------------------------------------------------------------ #

    def _generate_clip_deforum(self, prompt: str, scene_num: int, output_dir: Path) -> str | None:
        """Fallback: use Deforum for animated clip generation."""
        num_frames = self.fps * CLIP_DURATION_SECS

        deforum_settings = {
            "animation_mode": "2D",
            "max_frames": num_frames,
            "fps": self.fps,
            "width": 512,
            "height": 512,
            "prompts": {
                "0": prompt,
                str(num_frames // 2): prompt + ", continuation",
            },
            "strength_schedule": "0:(0.65)",
            "zoom": "0:(1.01)",
            "translation_x": "0:(0)",
            "translation_y": "0:(0)",
            "outdir": str(output_dir),
            "batch_name": f"scene_{scene_num:02d}",
        }

        try:
            resp = requests.post(
                f"{self.sd_url}/deforum_api/batches",
                json={"deforum_settings": [deforum_settings]},
                timeout=300,
            )
            resp.raise_for_status()
            batch_id = resp.json().get("batch_id")

            # Poll for completion
            for _ in range(60):
                time.sleep(5)
                status_resp = requests.get(f"{self.sd_url}/deforum_api/batches/{batch_id}")
                status = status_resp.json().get("status")
                if status == "succeeded":
                    break
                elif status == "failed":
                    return None

            # Find the output file
            mp4_files = list(output_dir.glob(f"scene_{scene_num:02d}*.mp4"))
            return str(mp4_files[0]) if mp4_files else None

        except Exception as e:
            print(f"  [animation] Deforum failed for scene {scene_num}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Static frame fallback (when no SD available)                        #
    # ------------------------------------------------------------------ #

    def _generate_static_placeholder(self, prompt: str, scene_num: int, output_dir: Path) -> str:
        """
        Last resort: generate a 4-second static image clip using FFmpeg.
        This keeps the pipeline running end-to-end even without SD.
        """
        # First, try to get a static image from SD txt2img
        img_path = output_dir / f"scene_{scene_num:02d}_frame.png"

        try:
            payload = {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, deformed",
                "steps": 15,
                "width": 512,
                "height": 512,
            }
            resp = requests.post(f"{self.sd_url}/sdapi/v1/txt2img", json=payload, timeout=90)
            resp.raise_for_status()
            img_b64 = resp.json()["images"][0]
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            print(f"  Scene {scene_num}: static SD image generated")
        except Exception:
            # No SD at all — create blank frame with FFmpeg
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c=0x1a1a2e:size=512x512:rate={self.fps}",
                "-t", str(CLIP_DURATION_SECS),
                "-vf", f"drawtext=text='Scene {scene_num}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2",
                str(output_dir / f"scene_{scene_num:02d}.mp4"),
            ], check=True, capture_output=True)
            print(f"  Scene {scene_num}: placeholder clip created (no SD available)")
            return str(output_dir / f"scene_{scene_num:02d}.mp4")

        # Convert static image to short clip
        out_path = output_dir / f"scene_{scene_num:02d}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1",
            "-i", str(img_path),
            "-c:v", "libx264",
            "-t", str(CLIP_DURATION_SECS),
            "-pix_fmt", "yuv420p",
            "-vf", f"scale=512:512,fps={self.fps}",
            str(out_path),
        ], check=True, capture_output=True)

        img_path.unlink(missing_ok=True)  # clean up raw image
        return str(out_path)

    # ------------------------------------------------------------------ #
    #  Public entrypoint                                                    #
    # ------------------------------------------------------------------ #

    def run(self, state: EpisodeState) -> EpisodeState:
        print("\n[4/6] Animation Agent starting...")

        if not state.characters:
            state.add_error("animation", "No character/scene data. Run StoryCharacterAgent first.")
            return state

        output_dir = Path(state.output_dir) / state.episode_id / "clips"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Detect available backend
        sd_up = self._sd_available()
        animatediff_up = sd_up and self._animatediff_available()

        if animatediff_up:
            print("  Backend: AnimateDiff (SD WebUI)")
        elif sd_up:
            print("  Backend: Deforum / Static frames (SD WebUI, no AnimateDiff)")
        else:
            print("  Backend: FFmpeg placeholders (SD not available)")

        clip_paths = []
        scene_prompts = state.characters.scene_prompts

        for i, prompt in enumerate(scene_prompts, 1):
            print(f"  Rendering scene {i}/{len(scene_prompts)}...", end=" ", flush=True)
            clip_path = None

            try:
                if animatediff_up:
                    clip_path = self._generate_clip_animatediff(prompt, i, output_dir)
                elif sd_up:
                    clip_path = self._generate_clip_deforum(prompt, i, output_dir)

                if not clip_path:
                    clip_path = self._generate_static_placeholder(prompt, i, output_dir)

                clip_paths.append(clip_path)
                print(f"done → {Path(clip_path).name}")

            except Exception as e:
                state.warnings.append(f"Scene {i} clip failed: {e}")
                print(f"WARN: {e}")
                # Try placeholder as last resort
                try:
                    clip_path = self._generate_static_placeholder(prompt, i, output_dir)
                    clip_paths.append(clip_path)
                except Exception:
                    pass

        state.animation = AnimationData(
            clip_paths=clip_paths,
            total_clips=len(clip_paths),
            fps=self.fps,
            resolution=RESOLUTION,
        )

        # Save manifest
        manifest_path = output_dir / "clips_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({"clips": clip_paths}, f, indent=2)

        state.mark_done("animation")
        print(f"  Clips generated: {len(clip_paths)}/{len(scene_prompts)}")
        print("  [4/6] DONE\n")

        return state
