# EEG2Video CS671 - Team 22

# EEG2Video — Sub-team 2: Generative Backbone

**IIT Mandi · CS671 · B.Tech Project**

Sub-team 2 owns the **Latent Diffusion Backbone and Video Generation** stage of the EEG2Video pipeline. This repository implements Phase 4 of the project in dummy mode, ready for real-data replacement.

---

## Pipeline Overview

```
EEG Signals (B, 7, 62, 100)
        │
        ▼ [Sub-team 3]
Visual Latents (B, 6, 4, 16, 16) ──────────┐
                                             │
        ▼ [Sub-team 4]                       │
Text Embeddings (B, 77, 512) ────────────────┤
Motion Label    (B,)         ────────────────┤
                                             │
                                      ┌──────▼──────────────────────────────────┐
                                      │          Sub-team 2 (this repo)         │
                                      │                                          │
                                      │  ┌─────────┐   ┌─────────────────────┐ │
                                      │  │  DANA   │──▶│   TemporalUNet      │ │
                                      │  │ (dana)  │   │   (sd_backbone)     │ │
                                      │  └─────────┘   └─────────┬───────────┘ │
                                      │                           │ DDIM        │
                                      │                    ┌──────▼──────┐      │
                                      │                    │   Decoder   │      │
                                      │                    │  (decoder)  │      │
                                      │                    └──────┬──────┘      │
                                      └───────────────────────────┼─────────────┘
                                                                  │
                                                                  ▼
                                                    RGB Frames (B, 6, 3, 128, 128)
```

---

## Module Responsibilities

| File | Role |
|---|---|
| `dana.py` | Dynamic-Aware Noise Adding — mixes static/diverse noise per motion label |
| `sd_backbone.py` | TemporalUNet — Tune-A-Video style backbone with spatial + temporal attention |
| `decoder.py` | Latent-to-frame VAE-style decoder (16×16 → 128×128 RGB) |
| `inference.py` | End-to-end inference orchestration (Stages 1–8) |
| `train_backbone.py` | Phase 4 training loop (MSE noise prediction loss) |
| `phase4_metrics.py` | PSNR, SSIM, LPIPS, FID metric computation + epoch logger |
| `dummy_data.py` | Centralised dummy tensor factory (all sub-team interfaces) |
| `test_dana.py` | Unit tests for the DANA module |
| `verify_pipeline_shapes.py` | End-to-end shape contract verification |

---

## Shape Contracts

All modules use these tensor shapes:

| Tensor | Shape |
|---|---|
| EEG input | `(B, 7, 62, 100)` |
| Visual latents *(Sub-team 3)* | `(B, 6, 4, 16, 16)` |
| Text embeddings *(Sub-team 4)* | `(B, 77, 512)` |
| Motion label *(Sub-team 4)* | `(B,)` — binary 0/1 |
| DANA output | `(B, 6, 4, 16, 16)` |
| TemporalUNet output | `(B, 6, 4, 16, 16)` |
| Decoder output | `(B, 6, 3, 128, 128)` |

---

## Quick Start

```bash
pip install -r requirements.txt

# Verify all shape contracts
python verify_pipeline_shapes.py

# Run DANA unit tests
python test_dana.py

# Run dummy training (Phase 4)
python train_backbone.py --epochs 5 --batch_size 2

# Run inference
python inference.py --steps 20 --out_dir outputs/
```

---

## Checkpoint Files

After training, the following are saved to `checkpoints/`:

| File | Contents |
|---|---|
| `subteam2_temporalunet_dummy.pt` | Final backbone weights + optimizer state |
| `subteam2_temporalunet_best.pt` | Best validation loss checkpoint |
| `phase4_dummy_metrics.json` | Final PSNR / SSIM values |
| `phase4_dummy_config.json` | Training hyperparameters |
| `phase4_training_history.json` | Per-epoch loss and metric curves |

---

## Integration with Other Sub-teams

### Consuming Sub-team 3 output (visual latents)

```python
# In inference.py / train_backbone.py, replace:
visual_latents = get_dummy_visual_latents(B)

# With: load Sub-team 3's ViT model and run forward pass
from subteam3.visual_transformer import VisualTransformer
vit = VisualTransformer()
vit.load_state_dict(torch.load("vit_real_data.pth"))
visual_latents = vit(eeg_embeddings)   # (B, 6, 4, 16, 16)
```

### Consuming Sub-team 4 output (text + motion)

```python
# Replace dummy calls with:
from subteam4.predictors import SemanticPredictor, MotionClassifier
semantic = SemanticPredictor()
motion   = MotionClassifier()
text_embeddings = semantic(eeg_embedding)   # (B, 77, 512)
motion_label    = motion(eeg_embedding)     # (B,)
```

---

## Phase Status

| Component | Status |
|---|---|
| DANA module | ✅ Complete & tested |
| TemporalUNet backbone | ✅ Complete & shape-verified |
| Latent decoder | ✅ Complete |
| Inference pipeline (dummy) | ✅ Complete |
| Phase-4 training loop (dummy) | ✅ Complete |
| PSNR / SSIM metrics | ✅ Complete |
| Real visual latent integration | ⏳ Waiting on Sub-team 3 |
| Real text / motion integration | ⏳ Waiting on Sub-team 4 |
| Real data training + evaluation | ⏳ Phase 5 |

---

## References

- Liu et al., *EEG2Video: Towards Decoding Dynamic Visual Perception from EEG Signals*, NeurIPS 2024
- Wu et al., *Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation*, ICCV 2023
- Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021