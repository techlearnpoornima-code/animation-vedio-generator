"""
webui/server.py
FastAPI backend for the Cartoon Pipeline Web UI.

Routes:
  GET  /                    → main UI
  GET  /api/status          → pipeline + ollama health
  POST /api/run             → start full pipeline run
  POST /api/run/agent       → start a single agent
  GET  /api/episodes        → list completed episode outputs
  GET  /api/episodes/{id}   → episode detail + file paths
  WS   /ws/logs             → real-time log streaming
"""

import asyncio
import json
import os
import sys
import queue
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="Cartoon Pipeline UI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ─── Global state ────────────────────────────────────────────────────────────

log_queue: queue.Queue = queue.Queue()
pipeline_running: bool = False
current_episode_id: Optional[str] = None
connected_clients: list[WebSocket] = []

OUTPUT_DIR = os.getenv("PIPELINE_OUTPUT_DIR", "./output")


# ─── Log interceptor ─────────────────────────────────────────────────────────

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put({"type": "log", "level": record.levelname, "msg": self.format(record), "ts": datetime.now().isoformat()})


def capture_print():
    """Redirect stdout so print() calls show up in the web UI."""
    import io

    class PrintCapture(io.TextIOWrapper):
        def __init__(self):
            self._buf = ""

        def write(self, text):
            sys.__stdout__.write(text)
            if text.strip():
                log_queue.put({"type": "log", "level": "INFO", "msg": text.rstrip(), "ts": datetime.now().isoformat()})
            return len(text)

        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = PrintCapture()


# ─── Request / response models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    genre: str = ""
    generate_images: bool = False
    upload_to_youtube: bool = False
    output_dir: str = OUTPUT_DIR


class AgentRequest(BaseModel):
    agent: str          # trend | script | character | animation | voiceover | editor
    genre: str = ""
    output_dir: str = OUTPUT_DIR


# ─── Background pipeline runner ───────────────────────────────────────────────

def _run_pipeline_thread(req: RunRequest):
    global pipeline_running, current_episode_id
    pipeline_running = True

    try:
        log_queue.put({"type": "status", "status": "running", "msg": "Pipeline started"})

        from pipeline.orchestrator import run_pipeline
        state = run_pipeline(
            genre_hint=req.genre,
            generate_images=req.generate_images,
            upload_to_youtube=req.upload_to_youtube,
            output_dir=req.output_dir,
        )

        current_episode_id = state.episode_id
        log_queue.put({
            "type": "complete",
            "status": "done",
            "episode_id": state.episode_id,
            "title": state.script.title if state.script else "",
            "video_path": state.final_video_path or "",
            "youtube_url": state.youtube_url or "",
            "errors": state.errors,
            "warnings": state.warnings,
        })

    except Exception as e:
        log_queue.put({"type": "error", "status": "failed", "msg": str(e)})
    finally:
        pipeline_running = False


def _run_agent_thread(req: AgentRequest):
    global pipeline_running
    pipeline_running = True
    try:
        log_queue.put({"type": "status", "status": "running", "msg": f"Agent [{req.agent}] started"})

        # Delegate to main.py's single-agent runner
        from main import run_single_agent
        run_single_agent(req.agent, req.genre, req.output_dir)

        log_queue.put({"type": "complete", "status": "done", "msg": f"Agent [{req.agent}] finished"})
    except Exception as e:
        log_queue.put({"type": "error", "status": "failed", "msg": str(e)})
    finally:
        pipeline_running = False


# ─── WebSocket log broadcaster ─────────────────────────────────────────────────

async def broadcast_logs():
    """Background task: drain queue and push to all WS clients."""
    while True:
        try:
            msg = log_queue.get_nowait()
            dead = []
            for ws in connected_clients:
                try:
                    await ws.send_text(json.dumps(msg))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                connected_clients.remove(ws)
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


@app.on_event("startup")
async def startup():
    capture_print()
    asyncio.create_task(broadcast_logs())


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/status")
async def status():
    ollama_ok = False
    ollama_models = []
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
        ollama_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    sd_ok = False
    try:
        r = requests.get(f"{os.getenv('SD_API_URL', 'http://localhost:7860')}/sdapi/v1/options", timeout=3)
        sd_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "pipeline_running": pipeline_running,
        "current_episode_id": current_episode_id,
        "ollama": {"available": ollama_ok, "models": ollama_models},
        "stable_diffusion": {"available": sd_ok},
        "output_dir": OUTPUT_DIR,
    }


@app.post("/api/run")
async def run_pipeline(req: RunRequest, background_tasks: BackgroundTasks):
    global pipeline_running
    if pipeline_running:
        return JSONResponse(status_code=409, content={"error": "Pipeline already running"})
    background_tasks.add_task(lambda: threading.Thread(target=_run_pipeline_thread, args=(req,), daemon=True).start())
    return {"started": True, "genre": req.genre}


@app.post("/api/run/agent")
async def run_agent(req: AgentRequest, background_tasks: BackgroundTasks):
    global pipeline_running
    if pipeline_running:
        return JSONResponse(status_code=409, content={"error": "Pipeline already running"})
    background_tasks.add_task(lambda: threading.Thread(target=_run_agent_thread, args=(req,), daemon=True).start())
    return {"started": True, "agent": req.agent}


@app.get("/api/episodes")
async def list_episodes():
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return {"episodes": []}

    episodes = []
    for ep_dir in sorted(output_path.glob("EP_*"), reverse=True):
        if not ep_dir.is_dir():
            continue
        meta = {"id": ep_dir.name, "created": ep_dir.stat().st_mtime}

        # Try to read script title
        script_json = ep_dir / "scripts" / "script.json"
        if script_json.exists():
            try:
                data = json.loads(script_json.read_text())
                meta["title"] = data.get("title", "")
                meta["genre"] = data.get("tone", "")
                meta["scenes"] = data.get("total_scenes", 0)
            except Exception:
                pass

        # Check final video
        final_dir = ep_dir / "final"
        mp4_files = list(final_dir.glob("*.mp4")) if final_dir.exists() else []
        meta["has_video"] = len(mp4_files) > 0
        meta["video_path"] = str(mp4_files[0]) if mp4_files else ""

        episodes.append(meta)

    return {"episodes": episodes}


@app.get("/api/episodes/{episode_id}")
async def episode_detail(episode_id: str):
    ep_dir = Path(OUTPUT_DIR) / episode_id
    if not ep_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Episode not found"})

    detail: dict = {"id": episode_id, "files": {}}

    for subdir in ["scripts", "characters", "clips", "audio", "final"]:
        sub = ep_dir / subdir
        if sub.exists():
            detail["files"][subdir] = [str(f.name) for f in sub.iterdir() if f.is_file()]

    script_json = ep_dir / "scripts" / "script.json"
    if script_json.exists():
        detail["script"] = json.loads(script_json.read_text())

    char_json = ep_dir / "characters" / "characters.json"
    if char_json.exists():
        detail["characters"] = json.loads(char_json.read_text())

    seo_json = ep_dir / "final" / "seo_metadata.json"
    if seo_json.exists():
        detail["seo"] = json.loads(seo_json.read_text())

    return detail


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        # Send current status immediately on connect
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": "running" if pipeline_running else "idle",
            "msg": "Connected to log stream",
        }))
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


# ─── Entry point ──────────────────────────────────────────────────────────────

def start():
    uvicorn.run("webui.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
