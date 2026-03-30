"""
setup_verify.py
===============
Sub-team 2 – Generative Backbone | Phase 1 Environment Check
CS 671 EEG2Video Reproduction | Team 22

Run this FIRST to confirm all required packages are installed correctly
before touching any model code.

Usage:
    conda activate eeg2video_env
    python setup_verify.py
"""

import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
#  Environment Safeguard – MUST be eeg2video_env (server & local)
# ─────────────────────────────────────────────────────────────────────────────
_REQUIRED_ENV = "eeg2video_env"
_active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
if _active_env != _REQUIRED_ENV:
    print("\n" + "!"*65)
    print(f"  [BLOCKED] Wrong conda environment detected.")
    print(f"  Active  : '{_active_env or 'none'}'")
    print(f"  Required: '{_REQUIRED_ENV}'")
    print(f"\n  Fix: conda activate {_REQUIRED_ENV}")
    print("!"*65 + "\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Sub-team 2 required packages  (versions from actual server pip freeze)
#
#  Format: import_name -> (pip_install_name, expected_prefix, critical)
#    expected_prefix : checked with startswith(); None = any version OK
#    critical        : True  -> MISSING = [FAIL] + hard exit at end
#                      False -> MISSING = [WARN] only
#
#  WARNING – two packages are NOT in the server pip freeze:
#    einops    : used heavily in sd_backbone.py  -> CRITICAL, install first
#    accelerate: needed for Phase 2 training     -> non-critical for Phase 1
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED = {
    # import_name      pip_name         expected_prefix   critical
    "torch":        ("torch",           "2.5",            True),
    "torchvision":  ("torchvision",     "0.20",           True),
    "diffusers":    ("diffusers",       "0.25",           True),
    "transformers": ("transformers",    "4.",             True),
    "wandb":        ("wandb",           "0.23",           False),
    "numpy":        ("numpy",           "2.2",            True),
    "PIL":          ("Pillow",          "11.",            True),
    "tqdm":         ("tqdm",            "4.67",           False),
    "safetensors":  ("safetensors",     "0.7",            True),
    "einops":       ("einops",          None,             True),   # NOT on server yet
    "accelerate":   ("accelerate",      None,             False),  # NOT on server yet
}

PASS  = "[PASS]"
WARN  = "[WARN]"
FAIL  = "[FAIL]"
NW    = 16   # name column
VW    = 24   # version column
EW    = 12   # expected column

print("\n" + "="*70)
print("  Team 22 – Sub-team 2 | Environment Verification")
print("  Server ground truth: torch==2.5.1+cu121  |  CUDA 12.1")
print("="*70)
print(f"  {'Package':<{NW}} {'Installed':<{VW}} {'Expected':<{EW}} Status")
print("-"*70)

all_ok   = True
missing_critical = []
missing_warn     = []

for import_name, (pip_name, expected_prefix, critical) in REQUIRED.items():
    try:
        mod     = __import__(import_name)
        version = getattr(mod, "__version__", "n/a")

        if expected_prefix is None:
            tag          = PASS
            expected_str = "any"
        elif version.startswith(expected_prefix):
            tag          = PASS
            expected_str = f"{expected_prefix}*"
        else:
            # Installed but different version – warn, not a hard failure
            tag          = WARN
            expected_str = f"{expected_prefix}*"

        print(f"  {import_name:<{NW}} {version:<{VW}} {expected_str:<{EW}} {tag}")

    except ImportError:
        expected_str = (expected_prefix + "*") if expected_prefix else "any"
        print(f"  {import_name:<{NW}} {'NOT INSTALLED':<{VW}} {expected_str:<{EW}} {FAIL}  <-- install this")
        if critical:
            missing_critical.append(pip_name)
            all_ok = False
        else:
            missing_warn.append(pip_name)

# ── Install instructions for missing packages ─────────────────────────────────
if missing_critical or missing_warn:
    print("\n" + "-"*70)
    print("  ACTION REQUIRED:")
    if missing_critical:
        print(f"\n  [CRITICAL – needed NOW for sd_backbone.py]")
        for pkg in missing_critical:
            print(f"    pip install {pkg}")
    if missing_warn:
        print(f"\n  [Phase 2+ – not needed for Phase 1 check]")
        for pkg in missing_warn:
            print(f"    pip install {pkg}")
    print()

# ── PyTorch / CUDA info ───────────────────────────────────────────────────────
print("-"*70)
try:
    import torch
    print(f"  PyTorch version  : {torch.__version__}")
    print(f"  CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Total VRAM       : {vram:.1f} GB")
        print(f"  20% VRAM limit   : {vram * 0.2:.1f} GB  (protocol cap for testing)")
    else:
        print("  Device           : CPU")
except Exception as e:
    print(f"  [WARN] Could not query torch device: {e}")

# ── Interface contract shapes ─────────────────────────────────────────────────
print("\n" + "-"*70)
print("  Interface Contract Shapes (Phase 1 – agreed by all sub-teams)")
print("-"*70)
try:
    import torch
    BATCH = 2
    shapes = {
        "EEG Input      (B, 7, C, 100) ": torch.randn(BATCH, 7, 62, 100).shape,
        "Visual Latents (B, 6, 4, H, W)": torch.randn(BATCH, 6, 4, 32, 32).shape,
        "Text Embeddings(B, 77, 768)   ": torch.randn(BATCH, 77, 768).shape,
    }
    for name, shape in shapes.items():
        print(f"  {name} : {tuple(shape)}")
    print("\n  [OK] All interface contract tensors created successfully.")
except Exception as e:
    print(f"  [FAIL] Tensor creation error: {e}")
    all_ok = False

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
if all_ok:
    print("  RESULT: READY  - all critical packages present.")
    print("  Next step: python sd_backbone.py --mode phase1_check")
else:
    print("  RESULT: NOT READY - install the [CRITICAL] packages above first.")
    print("  [WARN] items are non-critical for Phase 1.")
print("="*70 + "\n")
