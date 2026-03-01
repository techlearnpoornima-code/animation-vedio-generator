"""
agents/animation_agent.py — Agent 4: Animation

Generates one video clip per scene using ComfyUI.

Backend priority:
  1. ComfyUI + AnimateDiff-Evolved + VHS  → real animated .mp4 per scene
  2. ComfyUI + txt2img only               → static image looped to .mp4
  3. FFmpeg colour-card placeholder        → solid-colour .mp4 (no ComfyUI)

ComfyUI setup instructions:
  1. Clone: git clone https://github.com/comfyanonymous/ComfyUI
  2. Install requirements: pip install -r requirements.txt
  3. Download a checkpoint into ComfyUI/models/checkpoints/
  4. Start:  python main.py --listen 0.0.0.0 --port 8188
  5. For AnimateDiff: install via ComfyUI Manager or manually:
       cd ComfyUI/custom_nodes
       git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
       git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
  6. Download motion module into ComfyUI/models/animatediff_models/
       e.g. mm_sd_v15_v2.ckpt from HuggingFace

All paths use .resolve() — safe when server cwd is "/".
"""

import os
import json
import subprocess
import time
from pathlib import Path

from pipeline.state import EpisodeState, AnimationData
from pipeline.comfyui_client import ComfyUIClient
from pipeline.comfyui_workflows import build_animatediff_workflow

COMFYUI_URL      = os.getenv("COMFYUI_URL", "http://localhost:8188")
FPS              = int(os.getenv("ANIMATION_FPS", "24"))
CLIP_DURATION    = int(os.getenv("CLIP_DURATION_SECS", "4"))
RESOLUTION_W     = int(os.getenv("ANIMATION_WIDTH",  "384"))
RESOLUTION_H     = int(os.getenv("ANIMATION_HEIGHT", "384"))
RESOLUTION       = f"{RESOLUTION_W}x{RESOLUTION_H}"

print("Resolution: ", RESOLUTION)
class AnimationAgent:

    def __init__(self):
        self.client = ComfyUIClient(base_url=COMFYUI_URL)
        self.fps    = FPS

    # ── ComfyUI availability ──────────────────────────────────────────────

    def _detect_backend(self) -> str:
        """
        Returns one of:
          'animatediff'  — ComfyUI + AnimateDiff-Evolved + VHS available
          'comfyui'      — ComfyUI available but no AnimateDiff/VHS
          'ffmpeg'       — ComfyUI not running, use placeholder clips
        """
        if not self.client.is_available():
            return "ffmpeg"
        if self.client.has_animatediff() and self.client.has_vhs():
            return "animatediff"
        return "comfyui"

    # ── AnimateDiff via ComfyUI ───────────────────────────────────────────

    def _render_animatediff(
        self,
        prompt: str,
        scene_num: int,
        output_dir: Path,
    ) -> str | None:
        """
        Queue an AnimateDiff workflow in ComfyUI, wait for output,
        download the resulting MP4 to output_dir.
        """
        try:
            
            workflow, out_type = build_animatediff_workflow(
                prompt=prompt,
                client=self.client,
                scene_num=scene_num,
                fps=self.fps,
                duration_secs=CLIP_DURATION,
                width=RESOLUTION_W,
                height=RESOLUTION_H,
                            )

            history = self.client.run_workflow(workflow, timeout=600)
            import pdb; pdb.set_trace()
            outputs = self.client.extract_outputs(history)

            if not outputs:
                print(f"  [animation] ComfyUI returned no outputs for scene {scene_num}")
                return None

            out_path = output_dir / f"scene_{scene_num:02d}.mp4"

            if out_type == "video":
                # Download MP4 directly
                for output in outputs:
                    if output["media_type"] in ("gifs", "videos") or \
                       output["filename"].endswith(".mp4"):
                        ok = self.client.download_output(
                            output["filename"],
                            output["subfolder"],
                            output["type"],
                            out_path,
                        )
                        if ok:
                            return str(out_path)

            elif out_type == "image":
                # Download PNG and loop it into an MP4
                for output in outputs:
                    if output["media_type"] == "images" or \
                       output["filename"].endswith(".png"):
                        img_path = output_dir / f"scene_{scene_num:02d}_frame.png"
                        ok = self.client.download_output(
                            output["filename"],
                            output["subfolder"],
                            output["type"],
                            img_path,
                        )
                        if ok:
                            return self._image_to_clip(img_path, out_path)

            return None

        except Exception as e:
            print(f"  [animation] ComfyUI render failed for scene {scene_num}: {e}")
            return None

    def _image_to_clip(self, img_path: Path, out_path: Path) -> str | None:
        """Loop a static image into a fixed-duration MP4."""
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(img_path),
                "-c:v", "libx264",
                "-t", str(CLIP_DURATION),
                "-pix_fmt", "yuv420p",
                "-vf", f"scale={RESOLUTION_W}:{RESOLUTION_H},fps={self.fps}",
                str(out_path),
            ], check=True, capture_output=True)
            img_path.unlink(missing_ok=True)
            return str(out_path) if out_path.exists() else None
        except subprocess.CalledProcessError:
            return None

    # ── FFmpeg placeholder fallback ────────────────────────────────────────

    def _render_placeholder(self, scene_num: int, output_dir: Path) -> str:
        """
        Guaranteed fallback: coloured card with scene number.
        Uses absolute paths and a 3-tier fallback so it never raises.
        """
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"scene_{scene_num:02d}.mp4"

        # Tier 1: coloured card (most compatible)
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x1a1a2e:size={RESOLUTION_W}x{RESOLUTION_H}:rate={self.fps}",
                "-t", str(CLIP_DURATION),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out),
            ], check=True, capture_output=True)
            return str(out)
        except subprocess.CalledProcessError:
            pass

        # Tier 2: testsrc (always available in any FFmpeg build)
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={RESOLUTION_W}x{RESOLUTION_H}:rate={self.fps}",
            "-t", str(CLIP_DURATION),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ], check=True, capture_output=True)
        return str(out)

    # ── Public entrypoint ─────────────────────────────────────────────────

    def run(self, state: EpisodeState) -> EpisodeState:
        print("\n[4/6] Animation Agent starting...")

        if not state.characters:
            state.add_error("animation", "No character/scene data. Run StoryCharacterAgent first.")
            return state

        output_dir = Path(state.output_dir).resolve() / state.episode_id / "clips"
        output_dir.mkdir(parents=True, exist_ok=True)

        backend = self._detect_backend()
        backend_labels = {
            "animatediff": f"ComfyUI + AnimateDiff-Evolved ({COMFYUI_URL})",
            "comfyui":     f"ComfyUI txt2img → looped clip ({COMFYUI_URL})",
            "ffmpeg":      "FFmpeg colour-card placeholder (ComfyUI not running)",
        }
        print(f"  Backend: {backend_labels[backend]}")

        if backend == "animatediff":
            checkpoints = self.client.list_checkpoints()
            modules     = self.client.list_motion_modules()
            print(f"  Checkpoints:    {checkpoints[:3]}")
            print(f"  Motion modules: {modules[:3]}")

        scene_prompts = state.characters.scene_prompts
        clip_paths    = []

        for i, prompt in enumerate(scene_prompts, 1):
            print(f"  Rendering scene {i}/{len(scene_prompts)}...", end=" ", flush=True)
            clip_path = None

            try:
                if backend in ("animatediff", "comfyui"):
                    clip_path = self._render_animatediff(prompt, i, output_dir)

                if not clip_path:
                    # Fallback: placeholder (also used when ComfyUI returns nothing)
                    clip_path = self._render_placeholder(i, output_dir)
                    if backend in ("animatediff", "comfyui"):
                        state.warnings.append(
                            f"Scene {i}: ComfyUI failed, used FFmpeg placeholder"
                        )

                clip_paths.append(clip_path)
                src = "ComfyUI" if backend != "ffmpeg" and clip_path else "placeholder"
                print(f"done → {Path(clip_path).name}")

            except Exception as e:
                print(f"WARN — {e}")
                state.warnings.append(f"Scene {i} render error: {e}")
                try:
                    fallback = self._render_placeholder(i, output_dir)
                    clip_paths.append(fallback)
                except Exception:
                    pass

        state.animation = AnimationData(
            clip_paths=clip_paths,
            total_clips=len(clip_paths),
            fps=self.fps,
            resolution=RESOLUTION,
        )

        manifest = output_dir / "clips_manifest.json"
        with open(manifest, "w") as f:
            json.dump({
                "backend":  backend,
                "clips":    clip_paths,
                "fps":      self.fps,
                "resolution": RESOLUTION,
            }, f, indent=2)

        state.mark_done("animation")
        print(f"  Clips generated: {len(clip_paths)}/{len(scene_prompts)}")
        print("  [4/6] DONE\n")
        return state
