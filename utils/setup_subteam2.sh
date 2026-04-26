#!/bin/bash
# setup_subteam2.sh — Sub-team 2 Environment Setup
# Run this ONCE before training.
# Usage: bash setup_subteam2.sh

set -e  # exit on any error

echo "================================================"
echo " Sub-team 2 VideoLDM Environment Setup"
echo "================================================"

# Step 1: Upgrade diffusers from 0.11.1 to 0.21.4
echo ""
echo "[1/5] Upgrading diffusers to 0.21.4..."
pip install diffusers==0.21.4 --upgrade -q
python -c "import diffusers; print('diffusers version:', diffusers.__version__)"

# Step 2: Install video/image packages
echo ""
echo "[2/5] Installing imageio, opencv, einops..."
pip install imageio imageio-ffmpeg opencv-python-headless einops -q

# Step 3: Install modelscope for MindCine weights download
echo ""
echo "[3/5] Installing modelscope..."
pip install modelscope -q

# Step 4: Verify all critical imports
echo ""
echo "[4/5] Verifying imports..."
python -c "
import torch
import diffusers
import transformers
import accelerate
import imageio
print('torch:       ', torch.__version__)
print('diffusers:   ', diffusers.__version__)
print('transformers:', transformers.__version__)
print('accelerate:  ', accelerate.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"

# Step 5: Download ModelScope weights (MindCine base)
echo ""
echo "[5/5] Downloading ModelScope text-to-video weights..."
echo "This may take 5-10 minutes depending on connection speed."

mkdir -p ./modelscope_weights
python - <<'EOF'
import os
try:
    from modelscope.hub.snapshot_download import snapshot_download
    model_dir = snapshot_download(
        'damo/text-to-video-synthesis',
        cache_dir='./modelscope_weights'
    )
    print(f"Downloaded to: {model_dir}")
except Exception as e:
    print(f"ModelScope download failed: {e}")
    print("Falling back to Stable Diffusion v1.5 from HuggingFace...")
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype="auto",
    )
    os.makedirs("./modelscope_weights/unet", exist_ok=True)
    unet.save_pretrained("./modelscope_weights/unet")
    print("SD v1.5 UNet saved to ./modelscope_weights/unet")
EOF

echo ""
echo "================================================"
echo " Setup complete! Now run:"
echo "   python train_videoldm.py"
echo "================================================"
