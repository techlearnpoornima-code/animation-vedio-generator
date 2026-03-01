#!/usr/bin/env bash
# setup.sh — Initialize the cartoon pipeline project
# Run once after cloning: bash setup.sh

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RESET="\033[0m"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Cartoon Pipeline — Project Setup       ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo ""

# ── 1. Check uv ────────────────────────────────────────────────────────
echo -e "${CYAN}[1/6] Checking uv package manager...${RESET}"
if ! command -v uv &>/dev/null; then
    echo "  uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo -e "  ${GREEN}✓ uv $(uv --version)${RESET}"

# ── 2. Create virtual environment & install deps ────────────────────────
echo ""
echo -e "${CYAN}[2/6] Creating virtual environment with uv...${RESET}"
uv venv .venv --python 3.11
echo -e "  ${GREEN}✓ .venv created (Python 3.11)${RESET}"

echo ""
echo -e "${CYAN}[3/6] Installing dependencies...${RESET}"
uv pip install -e ".[dev]"
echo -e "  ${GREEN}✓ Dependencies installed${RESET}"

# ── 3. Environment file ─────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[4/6] Setting up environment config...${RESET}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "  ${YELLOW}⚠  .env created from template — edit it to add your API keys${RESET}"
else
    echo -e "  ${GREEN}✓ .env already exists${RESET}"
fi

# ── 4. Git init ─────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[5/6] Initializing git repository...${RESET}"
if [ ! -d .git ]; then
    git init
    git add .
    git commit -m "feat: initial cartoon pipeline scaffold

- 6 AI agents: trend research, script writing, character design,
  animation, voiceover, editor/upload
- LangGraph orchestrator with quality gate + retry logic
- FastAPI web UI with real-time WebSocket log streaming
- Fallbacks: AnimateDiff → Deforum → FFmpeg placeholders
- Coqui XTTS v2 → espeak → macOS say fallback chain
- uv package management (pyproject.toml)"
    echo -e "  ${GREEN}✓ Git repository initialized with initial commit${RESET}"
else
    echo -e "  ${GREEN}✓ Git already initialized${RESET}"
fi

# ── 5. Create .gitignore ────────────────────────────────────────────────
if [ ! -f .gitignore ]; then
cat > .gitignore << 'EOF'
# Virtual environment
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Environment & secrets
.env
client_secrets.json
youtube_token.pickle

# Pipeline output (often large files)
output/
logs/

# Stable Diffusion / model weights
*.ckpt
*.safetensors
*.bin
*.pt

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp

# Build
dist/
*.egg-info/
.ruff_cache/
.pytest_cache/
EOF
    git add .gitignore
    git commit -m "chore: add .gitignore"
    echo -e "  ${GREEN}✓ .gitignore created${RESET}"
fi

# ── 6. Check Ollama ─────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[6/6] Checking Ollama...${RESET}"
if command -v ollama &>/dev/null; then
    echo -e "  ${GREEN}✓ Ollama found${RESET}"
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ' ')
        echo -e "  ${GREEN}✓ Ollama server running${RESET}"
        echo -e "  Models available: ${MODELS:-none}"
        if ! echo "$MODELS" | grep -q "mistral"; then
            echo -e "  ${YELLOW}⚠  Recommended: ollama pull mistral${RESET}"
        fi
        if ! echo "$MODELS" | grep -q "llama3"; then
            echo -e "  ${YELLOW}⚠  Recommended: ollama pull llama3${RESET}"
        fi
    else
        echo -e "  ${YELLOW}⚠  Ollama not running — start with: ollama serve${RESET}"
    fi
else
    echo -e "  ${YELLOW}⚠  Ollama not installed — get it at: https://ollama.com/download${RESET}"
fi

# ── Done ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Setup complete!                        ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Activate env:   ${CYAN}source .venv/bin/activate${RESET}"
echo -e "  Run Web UI:     ${CYAN}uv run python -m webui.server${RESET}"
echo -e "  Run CLI:        ${CYAN}uv run python main.py --genre 'dark comedy'${RESET}"
echo -e "  Run one agent:  ${CYAN}uv run python main.py --agent trend${RESET}"
echo -e "  Add deps:       ${CYAN}uv add <package>${RESET}"
echo -e "  Dev tools:      ${CYAN}uv run ruff check .${RESET}"
echo ""
