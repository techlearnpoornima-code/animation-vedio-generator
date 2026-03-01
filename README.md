# Cartoon Pipeline

Multi-agent AI system that produces cartoon episodes end-to-end.
Local-first — runs on Ollama + free-tier APIs only.

## Quick Start

```bash
git clone <your-repo> && cd cartoon-pipeline
bash setup.sh          # installs uv, deps, inits git
source .venv/bin/activate

# Web UI
uv run python -m webui.server
# → open http://localhost:8000

# CLI
uv run python main.py --genre "dark comedy"
```

## uv Package Management

```bash
uv add requests             # add dependency
uv add --dev pytest         # dev dependency
uv pip install ".[tts]"     # optional Coqui TTS
uv run python main.py       # run without activating venv
uv run ruff check .         # lint
```

## Stack

| Agent | Tool |
|---|---|
| Trend Researcher | Ollama llama3 + SerpAPI / RSS |
| Script Writer | Ollama mistral |
| Story & Character | Ollama llava + Stable Diffusion |
| Animation | ComfyUI |
| Voiceover | Coqui XTTS v2 → espeak fallback |
| Editor | FFmpeg |
| SEO & Upload | Ollama + YouTube Data API v3 |
| Orchestrator | LangGraph |
| Web UI | FastAPI + WebSocket |

See full docs in README for setup, YouTube OAuth, and troubleshooting.
