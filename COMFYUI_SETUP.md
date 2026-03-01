# 🎬 ComfyUI + AnimateDiff Setup Guide (Clean & Isolated)

This guide installs **ComfyUI** in a virtual environment and enables
full animation using AnimateDiff.

------------------------------------------------------------------------

# 🧱 Architecture Overview

  -----------------------------------------------------------------------
  Installed Components                                 Output
  ---------------------------------------------------- ------------------
  ComfyUI only                                         Static image
                                                       generation

  ComfyUI + AnimateDiff + VHS + motion module          🎬 Real animated
                                                       MP4 per scene

  ComfyUI offline                                      Placeholder video
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 1️⃣ Install ComfyUI (Inside Virtual Environment)

## Clone Repository

``` bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
```

## Create Virtual Environment (Recommended)

### Using uv

``` bash
uv venv
source .venv/bin/activate
```

### OR Using standard Python

``` bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 2️⃣ Install PyTorch Properly

## 🍏 Mac (Apple Silicon / MPS)

``` bash
pip install torch torchvision torchaudio
```

Run later with:

``` bash
python main.py --force-fp16
```

## 🖥 NVIDIA GPU (CUDA 12.1 example)

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🐢 CPU Only

``` bash
python main.py --cpu
```

------------------------------------------------------------------------

# 3️⃣ Download a Checkpoint (Model)

Place any SD 1.5 `.ckpt` or `.safetensors` inside:

    ComfyUI/models/checkpoints/

Popular free models: - v1-5-pruned-emaonly.ckpt -
dreamshaper_8.safetensors - toonyou_beta6.safetensors

------------------------------------------------------------------------

# 4️⃣ Install AnimateDiff + Video Support

``` bash
cd custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

pip install -r ComfyUI-AnimateDiff-Evolved/requirements.txt
pip install -r ComfyUI-VideoHelperSuite/requirements.txt
```

Restart required after installation.

------------------------------------------------------------------------

# 5️⃣ Download Motion Module

Create folder if missing:

    ComfyUI/models/animatediff_models/

Recommended: - mm_sd_v15_v2.ckpt

Faster option: - mm_sd_v14.ckpt

Place inside:

    models/animatediff_models/

------------------------------------------------------------------------

# 6️⃣ Start ComfyUI

``` bash
python main.py --listen 0.0.0.0 --port 8188
```

GPU optimized:

``` bash
python main.py --listen 0.0.0.0 --port 8188 --gpu-only
```

Low VRAM:

``` bash
python main.py --lowvram
```

Open in browser:

    http://localhost:8188

------------------------------------------------------------------------

# 7️⃣ Environment Configuration

Create `.env` file:

    COMFYUI_URL=http://localhost:8188
    CLIP_DURATION_SECS=4
    ANIMATION_FPS=24
    ANIMATION_WIDTH=512
    ANIMATION_HEIGHT=512

------------------------------------------------------------------------

# 8️⃣ Verify Setup

``` bash
uv run python tools/test_pipeline.py --check
```

Expected:

    ✓ ComfyUI running
    ✓ AnimateDiff installed
    ✓ VideoHelperSuite installed
    ✓ Checkpoints detected
    ✓ Motion modules detected
    ✓ Full animation pipeline ready

------------------------------------------------------------------------

# 🛠 Troubleshooting

  -----------------------------------------------------------------------
  Issue                                  Fix
  -------------------------------------- --------------------------------
  ComfyUI not running                    Start with
                                         `python main.py --port 8188`

  AnimateDiff not detected               Ensure inside `custom_nodes/`
                                         and restart

  No checkpoints found                   Add `.ckpt` or `.safetensors` to
                                         `models/checkpoints/`

  No motion modules                      Add file to
                                         `models/animatediff_models/`

  CUDA OOM                               Reduce width/height to 384

  Very slow                              Use GPU or reduce steps
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🧠 Recommended Project Structure

    ai-projects/
    │
    ├── comfyui/
    ├── animation-agent/
    ├── rag-system/
    ├── pinterest-bot/

Keep ComfyUI isolated from other ML projects to avoid dependency
conflicts.
