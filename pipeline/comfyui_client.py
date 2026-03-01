"""
pipeline/comfyui_client.py
ComfyUI API client — handles workflow queueing, polling, and file retrieval.

ComfyUI's API is fundamentally different from SD WebUI:
  - Workflows are JSON graphs of nodes (not simple JSON payloads)
  - Jobs are queued via POST /prompt, tracked by prompt_id
  - Outputs are retrieved via GET /view?filename=...&subfolder=...&type=output
  - Real-time progress available via WebSocket /ws?clientId=<uuid>

Supports:
  - AnimateDiff-Evolved (ADE) for video generation
  - VHS (VideoHelperSuite) for video file output
  - Standard txt2img as static frame fallback
"""

import uuid
import json
import time
import requests
from pathlib import Path
from typing import Any

DEFAULT_URL     = "http://localhost:8188"
POLL_INTERVAL   = 2       # seconds between queue checks
POLL_TIMEOUT    = 600     # max seconds to wait for a job


class ComfyUIClient:

    def __init__(self, base_url: str = DEFAULT_URL):
        self.base_url  = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    # ── Connectivity ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def has_node(self, node_class: str) -> bool:
        """Check if a specific custom node class is installed."""
        try:
            r = requests.get(f"{self.base_url}/object_info/{node_class}", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def has_animatediff(self) -> bool:
        """Check for AnimateDiff-Evolved nodes."""
        return (
            self.has_node("ADE_AnimateDiffLoaderWithContext")
            or self.has_node("AnimateDiffLoaderV2")
        )

    def has_vhs(self) -> bool:
        """Check for VideoHelperSuite (needed for video output)."""
        return self.has_node("VHS_VideoCombine")

    def list_checkpoints(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/object_info/CheckpointLoaderSimple", timeout=5)
            data = r.json()
            return data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [{}])[0] or []
        except Exception:
            return []

    def list_motion_modules(self) -> list[str]:
        """List available AnimateDiff motion modules."""
        try:
            for node in ["ADE_AnimateDiffLoaderWithContext", "AnimateDiffLoaderV2"]:
                r = requests.get(f"{self.base_url}/object_info/{node}", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    node_data = data.get(node, {})
                    inputs = node_data.get("input", {}).get("required", {})
                    mm_field = inputs.get("model_name") or inputs.get("motion_model_name") or []
                    if isinstance(mm_field, list) and mm_field:
                        return mm_field[0] if isinstance(mm_field[0], list) else mm_field
            return []
        except Exception:
            return []

    # ── Job execution ─────────────────────────────────────────────────────

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow and return the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = requests.post(f"{self.base_url}/prompt", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"ComfyUI workflow error: {data['error']}")
        return data["prompt_id"]

    def wait_for_completion(self, prompt_id: str, timeout: int = POLL_TIMEOUT) -> dict:
        """
        Poll /history until the job is done.
        Returns the history entry for this prompt_id.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
            if r.status_code == 200:
                history = r.json()
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed"):
                        return entry
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI job failed: {status}")
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout}s")

    def download_output(self, filename: str, subfolder: str, file_type: str, dest: Path) -> bool:
        """Download a generated output file."""
        try:
            params = {"filename": filename, "subfolder": subfolder, "type": file_type}
            r = requests.get(f"{self.base_url}/view", params=params, timeout=120, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e:
            print(f"  [comfyui] Download failed: {e}")
            return False

    def run_workflow(self, workflow: dict, timeout: int = POLL_TIMEOUT) -> dict:
        """Queue a workflow, wait for completion, return history entry."""
        prompt_id = self.queue_prompt(workflow)
        return self.wait_for_completion(prompt_id, timeout=timeout)

    def extract_outputs(self, history_entry: dict) -> list[dict]:
        """
        Extract all output files from a history entry.
        Returns list of {filename, subfolder, type, node_id, media_type, fullpath}.
        """
        outputs = []
        
        if not history_entry or "outputs" not in history_entry:
            return outputs
        
        for node_id, node_output in history_entry["outputs"].items():
            # Handle different output types
            for media_type in ["images", "gifs", "videos", "audio"]:
                items = node_output.get(media_type, [])
                if not items:
                    continue
                    
                for item in items:
                    if not isinstance(item, dict):
                        continue
                        
                    filename = item.get("filename", "")
                    if not filename:
                        continue
                    
                    output = {
                        "filename": filename,
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                        "node_id": node_id,
                        "media_type": media_type.rstrip('s'),  # 'image', 'gif', 'video', 'audio'
                        "fullpath": item.get("fullpath", ""),  # Some versions include fullpath
                    }
                    
                    # Try to construct fullpath if not provided
                    if not output["fullpath"] and hasattr(self, 'output_dir'):
                        # You might need to set self.output_dir based on your ComfyUI path
                        base_dir = getattr(self, 'output_dir', '/Users/poornimabyregowda/playground/ai-tools/ComfyUI/output')
                        sub = output["subfolder"]
                        if sub:
                            output["fullpath"] = str(Path(base_dir) / sub / filename)
                        else:
                            output["fullpath"] = str(Path(base_dir) / filename)
                    
                    outputs.append(output)
    
        return outputs
