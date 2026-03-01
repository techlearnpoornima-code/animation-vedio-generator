"""
pipeline/comfyui_workflows.py
ComfyUI workflow templates for the cartoon pipeline.

Each function returns a complete ComfyUI workflow graph (dict) ready to
POST to /prompt. Node IDs are strings; links are expressed as
["node_id", output_slot_index].

Workflows:
  animatediff_workflow()  — AnimateDiff-Evolved + VHS video output
  txt2img_workflow()      — Standard image generation (static frame fallback)

ComfyUI node numbering convention used here:
  "1"  CheckpointLoaderSimple
  "2"  CLIPTextEncode (positive)
  "3"  CLIPTextEncode (negative)
  "4"  KSampler  (or SamplerCustomAdvanced for ADE)
  "5"  VAEDecode
  "6"  SaveImage / VHS_VideoCombine
  "7"  ADE_AnimateDiffLoaderWithContext  (AnimateDiff only)
  "8"  ADE_UseEvolvedSampling            (AnimateDiff only)
  "9"  EmptyLatentImage
"""

NEGATIVE_PROMPT = (
    "nsfw, blurry, low quality, deformed, extra limbs, "
    "ugly, text, watermark, noise, grain, overexposed, "
    "bad anatomy, duplicate, mutation"
)


def animatediff_workflow(
    prompt:         str,
    checkpoint:     str = "v1-5-pruned-emaonly.ckpt",
    motion_module:  str = "mm_sd_v15_v2.ckpt",
    width:          int = 512,
    height:         int = 512,
    num_frames:     int = 96,        # 4s @ 24fps
    fps:            int = 24,
    steps:          int = 20,
    cfg:            float = 7.0,
    filename_prefix: str = "scene",
    seed:           int = -1,
) -> dict:
    """
    AnimateDiff-Evolved workflow for ComfyUI.
    Uses ADE_AnimateDiffLoaderWithContext + ADE_UseEvolvedSampling + VHS_VideoCombine.
    Falls back to AnimateDiffLoaderV2 node name if ADE_ nodes not found.
    """
    import random
    if seed == -1:
        seed = random.randint(0, 2**31)

    return {
        # ── 1. Load checkpoint ──────────────────────────────────────────
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint
            }
        },

        # ── 2. Positive prompt ──────────────────────────────────────────
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1]          # slot 1 = CLIP output of CheckpointLoader
            }
        },

        # ── 3. Negative prompt ──────────────────────────────────────────
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": NEGATIVE_PROMPT,
                "clip": ["1", 1]
            }
        },

        # ── 4. Empty latent ─────────────────────────────────────────────
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                # "batch_size": num_frames,
                # "width":  width,
                # "height": height,
                "batch_size": 16,
                "width": 384,
                "height": 384
            }
        },

        # ── 5. Load AnimateDiff motion module ───────────────────────────
        "5": {
            "class_type": "ADE_AnimateDiffLoaderWithContext",
            "inputs": {
                "model_name": motion_module,
                "beta_schedule": "autoselect",
                "context_options": None,
                "motion_lora":     None,
                "ad_settings":     None,
                "sample_settings": None,
                "motion_scale":    1.0,
                "apply_v2_models_properly": True,
                "model": ["1", 0]       # slot 0 = MODEL output of CheckpointLoader
            }
        },

        # ── 6. KSampler with ADE evolved sampling ───────────────────────
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed":        seed,
                "steps":       steps,
                "cfg":         cfg,
                "sampler_name": "euler_ancestral",
                "scheduler":   "karras",
                "denoise":     1.0,
                "model":       ["5", 0],   # AnimateDiff-patched model
                "positive":    ["2", 0],
                "negative":    ["3", 0],
                "latent_image": ["4", 0]
            }
        },

        # ── 7. VAE decode ────────────────────────────────────────────────
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae":     ["1", 2]       # slot 2 = VAE output of CheckpointLoader
            }
        },

        # ── 8. VHS VideoCombine — output as MP4 ─────────────────────────
        "8": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images":           ["7", 0],
                "frame_rate":       fps,
                "loop_count":       0,
                "filename_prefix":  filename_prefix,
                "format":           "video/h264-mp4",
                "pingpong":         False,
                "save_output":      True,
                # Optional: audio track (not used here)
                "audio":            None,
            }
        }
    }


def txt2img_workflow(
    prompt:          str,
    checkpoint:      str = "v1-5-pruned-emaonly.ckpt",
    width:           int = 512,
    height:          int = 512,
    steps:           int = 20,
    cfg:             float = 7.0,
    filename_prefix: str = "scene",
    seed:            int = -1,
) -> dict:
    """
    Standard txt2img workflow — generates a single PNG image.
    Used as static-frame fallback when AnimateDiff is not available.
    """
    import random
    if seed == -1:
        seed = random.randint(0, 2**31)

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["1", 1]}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "width": width, "height": height}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed":         seed,
                "steps":        steps,
                "cfg":          cfg,
                "sampler_name": "euler_ancestral",
                "scheduler":    "karras",
                "denoise":      1.0,
                "model":        ["1", 0],
                "positive":     ["2", 0],
                "negative":     ["3", 0],
                "latent_image": ["4", 0]
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images":          ["6", 0],
                "filename_prefix": filename_prefix
            }
        }
    }


def build_animatediff_workflow(
    prompt:         str,
    client,                         # ComfyUIClient instance for capability checks
    scene_num:      int,
    fps:            int = 24,
    duration_secs:  int = 4,
    width:          int = 512,
    height:         int = 512,
) -> tuple[dict, str]:
    """
    Build the best available AnimateDiff workflow given what's installed.
    Returns (workflow_dict, output_type) where output_type is 'video' or 'image'.
    Picks the first available checkpoint and motion module automatically.
    """
    checkpoints = client.list_checkpoints()
    checkpoint  = checkpoints[0] if checkpoints else "v1-5-pruned-emaonly.ckpt"

    motion_modules = client.list_motion_modules()
    motion_module  = motion_modules[0] if motion_modules else "mm_sd_v15_v2.ckpt"

    num_frames = fps * duration_secs
    prefix     = f"scene_{scene_num:02d}"

    if client.has_animatediff() and client.has_vhs():
        wf = animatediff_workflow(
            prompt=prompt, checkpoint=checkpoint,
            motion_module=motion_module,
            width=width, height=height,
            num_frames=num_frames, fps=fps,
            filename_prefix=prefix,
        )
        return wf, "video"
    else:
        # Fall back to single image (no AnimateDiff or no VHS)
        wf = txt2img_workflow(
            prompt=prompt, checkpoint=checkpoint,
            width=width, height=height,
            filename_prefix=prefix,
        )
        return wf, "image"
