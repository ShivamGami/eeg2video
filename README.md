<div align="center">

# 🧠 EEG2Video
### *EEG-to-Video Generation using Transformer and Latent Diffusion Models*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Diffusers-0.10%2B-FFD21E)](https://huggingface.co/docs/diffusers)
[![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?logo=openai)](https://github.com/openai/CLIP)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![IIT Mandi](https://img.shields.io/badge/IIT%20Mandi-CS671-blue)](https://www.iitmandi.ac.in/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracked-FFBE00?logo=weightsandbiases)](https://wandb.ai/)

<br/>

> **Reconstructing temporally coherent video clips directly from raw EEG brain signals**
> using an EEGNet-inspired Transformer encoder, CLIP semantic alignment, Dynamic-Aware Noise Addition (DANA), and Stable Diffusion / VideoLDM conditioning.

<br/>

---

**B.Tech Project · CS671 · Indian Institute of Technology, Mandi · 2025–26**

---

</div>

<br/>

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture-overview)
- [Key Concepts](#-key-concepts-glossary)
- [Dataset](#-dataset--seed-dv)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Training Pipeline](#-training-pipeline)
- [Inference Pipeline](#-inference-pipeline)
- [Checkpoint Management](#-checkpoint-management)
- [Configuration & Hyperparameters](#-configuration--hyperparameters)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Challenges & Mitigations](#-challenges--mitigations)
- [Team Contributions](#-team-contributions)
- [Future Work](#-future-work)
- [Troubleshooting](#-troubleshooting)
- [Acknowledgements](#-acknowledgements)
- [References](#-references)

---

## 🔭 Overview

**EEG2Video** is a research-grade generative pipeline that reconstructs temporally coherent video sequences directly from non-invasive EEG brain recordings. Given a 500 ms window of 62-channel EEG data recorded while a subject watches a video, the system generates a multi-frame video clip that semantically corresponds to the observed visual stimulus.

The pipeline is organized into **four specialized sub-teams**, each owning a distinct module of the end-to-end system:

| Sub-team | Module | Role |
|---|---|---|
| **Sub-team 1** | EEG Preprocessing | STFT spectrograms, bandpass filtering, epoch windowing |
| **Sub-team 2** | VideoLDM Fine-tuning | Stable Diffusion UNet, cross-attention conditioning, DDPM/DDIM |
| **Sub-team 3** | ViT Seq2Seq Latent Predictor | Visual latent estimation from EEG embeddings |
| **Sub-team 4** | Dynamics Predictor + Text MLP | CLIP embedding prediction, fast/slow motion classification |

The complete inference chain operates as follows:

```
Raw EEG (62ch × 500ms)
        │
        ▼
STFT Spectrogram (62 × 51 × 9)
        │
        ▼
EEG Adapter / Transformer Encoder ──── EEG Embedding (512-dim)
        │                                       │
        ▼                                       ▼
ViT Seq2Seq (Sub-team 3)              Text MLP → CLIP Embedding (77 × 768)
Visual Latents (6 × 4 × 32 × 32)     Dynamics Classifier → is_fast (0/1)
        │                                       │
        └──────────────┬────────────────────────┘
                       ▼
              DANA Noise Scheduler
        (β_fast=0.85 | β_slow=0.35)
                       │
                       ▼
         Noised Latent (6 × 4 × 32 × 32)
                       │
                       ▼
        VideoLDM UNet (SD v1.5 backbone)
        Conditioned via Cross-Attention
        on CLIP Embedding (77 × 768)
                       │
                       ▼
        Denoised Latent → VAE Decoder
                       │
                       ▼
         Generated Video (6 frames @ 3 FPS)
```

### What Makes This Pipeline Novel

- **DANA (Dynamic-Aware Noise Addition):** Instead of starting denoising from pure Gaussian noise, the system uses the ViT-predicted visual latent as a structural prior, adding controlled noise scaled by predicted motion dynamics (`β_fast = 0.85` for dynamic content, `β_slow = 0.35` for static content). This allows the diffusion model to refine structure rather than hallucinate from scratch.
- **Tri-Path EEG Conditioning:** Semantic (CLIP), structural (ViT), and dynamic (DANA) conditioning streams are fused, providing richer guidance than any single path alone.
- **Dual-Path EEG Adapter:** A flat spectral path (28,458 raw time-frequency values) is fused with a band-pooled path (310 frequency-band features across Delta/Theta/Alpha/Beta/Gamma) to produce a robust 512-dim EEG embedding.
- **Latent-Space Conditioning:** EEG embeddings are projected to 77 × 768 context tensors via learnable Transformer queries and injected into the Stable Diffusion UNet via cross-attention at every denoising step — identical to text conditioning in standard SD.

---

## 🏗 Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                     EEG2Video Full Pipeline                           │
├─────────────────┬─────────────────┬────────────────┬─────────────────┤
│   Sub-team 1    │   Sub-team 4    │  Sub-team 3    │  Sub-team 2     │
│ EEG Preprocess  │  EEG Adapter    │  ViT Seq2Seq   │  VideoLDM       │
│                 │  Text MLP       │  Latent Pred.  │  UNet + DANA    │
├─────────────────┼─────────────────┼────────────────┼─────────────────┤
│ • 62-ch EEG     │ • Flat Path     │ • ViT encoder  │ • SD v1.5 UNet  │
│ • STFT (SciPy)  │   (28458→256)   │ • Seq2Seq LSTM │ • Cross-attn    │
│ • Bandpass      │ • Band Path     │ • 6-frame pred │ • DDPM/DDIM     │
│ • Epoch window  │   (310→256)     │ • Shape:       │ • EMA ckpts     │
│ • Shape:        │ • Fusion        │   (6,4,32,32)  │ • fp16 training │
│  (62,51,9)      │   (512→512)     │                │                 │
│                 │ • Dynamics MLP  │                │ • VAE Decoder   │
│                 │   → is_fast     │                │ • 128×128 RGB   │
└─────────────────┴─────────────────┴────────────────┴─────────────────┘
```

---

## 📚 Key Concepts Glossary

| Concept | Description |
|---|---|
| **STFT Spectrogram** | Short-Time Fourier Transform applied per EEG channel, converting the 1D time-series into a 2D time-frequency map. Enables the Transformer to process EEG as image-like patches. |
| **EEG Embedding** | 512-dimensional dense vector produced by the Dual-Path EEG Adapter. Encodes both full spectral detail and frequency-band structure. |
| **Transformer Attention** | Multi-head self-attention (8 heads, d=256) applied across temporal patches of the EEG spectrogram, capturing long-range temporal dependencies. |
| **CLIP Alignment** | The EEG embedding is projected via an MLP into CLIP's 512-dim visual-language latent space. Cosine similarity loss enforces semantic alignment with ground-truth image embeddings. |
| **ViT Latents** | A Vision Transformer (ViT) encodes reference visual frames and a Seq2Seq network predicts 6 VAE-encoded latent frames `(6 × 4 × 32 × 32)` from the EEG embedding, providing structural guidance to the diffusion model. |
| **DANA Scheduler** | Dynamic-Aware Noise Addition: adds β-scaled Gaussian noise to ViT-predicted latents based on the predicted motion label. `β_fast=0.85`, `β_slow=0.35`. |
| **DDPM/DDIM Sampling** | DDPM for training; DDIM for accelerated inference. The UNet iteratively denoises the DANA-noised latent conditioned on EEG context. |
| **Classifier-Free Guidance** | EEG conditioning is randomly dropped during training (p=0.1), training the UNet to operate both conditioned and unconditionally. Guidance scale amplifies the conditioned direction at inference. |
| **EEGProjection** | A 4-layer Transformer with 77 learnable latent queries that converts the 512-dim EEG embedding into a `(77 × 768)` context tensor — matching CLIP text token shape for cross-attention injection. |
| **EMA Checkpoints** | Exponential Moving Average of model weights for more stable inference. Saved separately from the training checkpoint. |
| **VAE Encoder/Decoder** | The Stable Diffusion VAE encodes 128×128 RGB frames into `(4 × 32 × 32)` latent tensors and decodes generated latents back to pixel space. |

---

## 🗃 Dataset — SEED-DV

This project uses the **SEED-DV EEG-video dataset**.

### Dataset Statistics

| Property | Value |
|---|---|
| EEG Channels | 62 |
| Sampling Rate | ~200 Hz (after downsampling) |
| Window Size | 500 timesteps |
| Frames per Sample | 6 (@ 3 FPS) |
| Total Samples | ~291,000 processed samples |
| Train / Val / Test Split | 70% / 15% / 15% (random seed 42) |
| EEG Tensor Shape | `(62, 51, 9)` |
| Video Latent Shape | `(6, 4, 32, 32)` |
| Text Embedding Shape | `(77, 768)` |
| Visual Output Resolution | 128 × 128 RGB |

### Dataset Preparation

The processed dataset directory is expected at:

```
/home/teaching/TEAM_22_DATASET/processed/processed/
├── eeg_sample_000001.pt        # (62, 51, 9) EEG spectrogram tensor
├── video_sample_000001.pt      # (6, 4, 32, 32) VAE-encoded video latent
├── text_sample_000001.pt       # (77, 768) CLIP text embedding
├── dynamics_labels_fixed_BINARY.npy   # Binary fast/slow motion labels
├── train_split.txt
├── val_split.txt
└── test_split.txt
```

### EEG Preprocessing Details

Raw EEG signals are preprocessed using `MNE-Python` and `SciPy`:

1. **Bandpass Filtering:** Zero-phase Butterworth filter (0.5–50 Hz) — removes DC drift and high-frequency noise outside physiological EEG bands
2. **Artifact Removal:** ICA (Independent Component Analysis) to suppress ocular (EOG) and muscular (EMG) artifacts
3. **Epoch Segmentation:** 500 timestep epochs time-locked to video stimuli
4. **Standardization:** Per-channel z-score normalization (zero mean, unit variance)
5. **STFT Computation:** `scipy.signal.stft` applied per channel → `(51 freq bins × 9 time frames)`
6. **Stacking:** All 62 channels stacked → `(62, 51, 9)` tensor saved as `.pt`

**Expected tensor shapes at each pipeline stage:**

```python
eeg_spectrogram   : torch.Size([B, 62, 51, 9])     # input to EEG Adapter
eeg_embedding     : torch.Size([B, 512])             # EEG Adapter output
clip_context      : torch.Size([B, 77, 768])         # EEGProjection output → UNet
visual_latents    : torch.Size([B, 6, 4, 32, 32])   # ViT Seq2Seq output
is_fast           : torch.Size([B, 1])               # Dynamics classifier output
noised_latent     : torch.Size([B, 6, 4, 32, 32])   # DANA output → UNet input
generated_frames  : torch.Size([B, 6, 3, 128, 128]) # VAE-decoded RGB output
```

---

## 📂 Repository Structure

```
eeg2video-cs671/
│
├── 📁 data/
│   └── dataset.py                  # EEGVideoDataset — 70/15/15 splits
│
├── 📁 models/
│   ├── 📁 eeg_encoder/
│   │   ├── eeg_transformer.py      # EEGTransformer: 4-layer, 8-head Transformer encoder
│   │   ├── eeg_to_latent.py        # EEG → latent space projection utilities
│   │   └── vision_transformer.py   # ViT backbone for visual latent prediction
│   │
│   ├── 📁 decoder/
│   │   └── projection.py           # EEGProjection: 512 → (77 × 768) context tensor
│   │
│   ├── 📁 diffusion_backbone/
│   │   ├── dana.py                 # DANAModule: Dynamic-Aware Noise Addition
│   │   └── inference.py            # Full VideoLDM inference pipeline
│   │
│   └── 📁 dynamics_predictor/
│       └── subteam4_models.py      # EEGAdapter, TextProjectorMLP, DynamicsClassifier
│
├── 📁 training/
│   ├── base_train.py               # Base trainer class and common utilities
│   ├── train_videoldm.py           # Sub-team 2: VideoLDM UNet fine-tuning
│   ├── train_text_mlp.py           # Sub-team 4: Text MLP (CLIP embedding predictor)
│   ├── train_dynamics_mlp.py       # Sub-team 4: Dynamics classifier (fast/slow)
│   ├── train_vit_seq2seq.py        # Sub-team 3: ViT Seq2Seq latent predictor
│   └── sweep_train.py              # W&B hyperparameter sweep launcher
│
├── 📁 scripts/
│   ├── generate_tensors.py         # Export text_embeddings.pt + is_fast.pt
│   ├── generate_visual_latents.py  # Export visual_latents.pt from ViT Seq2Seq
│   ├── generate_real_latents.py    # Extract ground-truth VAE latents for comparison
│   └── generate_video.py           # Full inference: EEG → video frames
│
├── 📁 pipelines/
│   └── run_pipeline_test.py        # End-to-end pipeline integration test
│
├── 📁 evaluation/
│   ├── evaluate.py                 # SSIM, PSNR, LPIPS, FID, CLIP-sim evaluation
│   └── eval_results/
│       ├── metrics_latent.json     # Per-sample cosine similarity + MSE
│       └── metrics_video.json      # Aggregate video-level metrics
│
├── 📁 utils/
│   ├── subteam2_dataset.py         # VideoLDM-specific DataLoader
│   ├── myvideo.py                  # Video I/O utilities (frame assembly, GIF export)
│   ├── oracle_test.py              # Upper-bound evaluation with GT latents
│   ├── extract_truth.py            # Ground-truth frame extraction from dataset
│   ├── test_gt_reconstruction.py   # Sanity check: VAE encode → decode roundtrip
│   ├── test_model.py               # Quick model forward-pass test
│   ├── run_subteam4_full.py        # Sub-team 4 full inference runner
│   └── setup_subteam2.sh           # Environment setup script for Sub-team 2
│
├── 📁 checkpoints/
│   └── unet_best/
│       └── config.json             # SD v1.5 UNet2DConditionModel config
│
├── 📁 real_inputs/                 # Ground-truth EEG/video tensors for evaluation
├── 📁 outputs/                     # Generated video frames and GIFs
├── .gitignore
└── README.md
```

---

## ⚙ Installation & Environment Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ with a GPU ≥ 16 GB VRAM (A100 recommended for VideoLDM training)
- `conda` or `venv`

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-org>/eeg2video-cs671.git
cd eeg2video-cs671
```

### Step 2 — Create Environment

```bash
# Using conda (recommended)
conda create -n eeg2video python=3.10 -y
conda activate eeg2video

# Or using venv
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\Activate.ps1         # PowerShell (Windows)
```

### Step 3 — Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.10.2 transformers accelerate
pip install open-clip-torch mne scipy scikit-image scikit-video
pip install torchmetrics wandb tqdm numpy matplotlib
```

Or use the provided setup script:

```bash
bash utils/setup_subteam2.sh
```

### Step 4 — Download Pre-trained Backbone

```bash
python -c "
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet')
unet.save_pretrained('./modelscope_weights')
"
```

### Step 5 — Verify Installation

```bash
python utils/test_model.py
# Expected: EEGProjection forward pass: torch.Size([2, 77, 768]) ✓
```

---

## 🚀 Training Pipeline

### Stage 1 — Text MLP (CLIP Embedding Predictor)

```bash
python training/train_text_mlp.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --batch_size 128 \
    --lr 1e-3 \
    --epochs 30 \
    --output_dir ./checkpoints/text_mlp
```

**Objective:** Align EEG embeddings with CLIP's visual-language latent space.

```python
# Combined loss
loss_total = 0.5 * loss_mse + 0.5 * loss_cosine
# Cosine: loss = 1 - cosine_similarity(eeg_emb, clip_emb)
```

### Stage 2 — Dynamics Classifier (Fast/Slow Motion)

```bash
python training/train_dynamics_mlp.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --batch_size 256 \
    --lr 5e-4 \
    --epochs 20 \
    --output_dir ./checkpoints/dynamics
```

### Stage 3 — ViT Seq2Seq Latent Predictor

```bash
python training/train_vit_seq2seq.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --output_dir ./checkpoints/vit_seq2seq
```

### Stage 4 — VideoLDM Fine-Tuning

The `UNet2DConditionModel` is fine-tuned with **frozen backbone weights** and **trainable cross-attention layers only**:

```bash
python training/train_videoldm.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --weights_dir ./modelscope_weights \
    --save_dir ./checkpoints \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 1e-4 \
    --epochs 10 \
    --mixed_precision fp16 \
    --use_ema \
    --subset 20000
```

**Key training details:**
- Effective batch size = 8 (batch_size=2 × grad_accum=4)
- Only cross-attention (`attn2`) layers are trainable; all spatial conv/norm layers frozen
- fp16 mixed precision + gradient checkpointing required for GPUs < 24 GB VRAM

### W&B Hyperparameter Sweep

```bash
wandb sweep configs/sweep_config.yaml
wandb agent <sweep-id>
```

---

## 🎬 Inference Pipeline

### Step 1 — Export Conditioning Tensors

```bash
# Sub-team 4: Export text_embeddings.pt and is_fast.pt
python scripts/generate_tensors.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --text_mlp_path checkpoints/text_mlp_final.pth \
    --adapter_path checkpoints/eeg_adapter.pth \
    --dynamics_path checkpoints/dynamics_model.pth \
    --out_text text_embeddings.pt \
    --out_dynamics is_fast.pt

# Sub-team 3: Export visual_latents.pt
python scripts/generate_visual_latents.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --vit_checkpoint checkpoints/vit_seq2seq.pth \
    --output visual_latents.pt

# Ground-truth VAE latents for oracle evaluation
python scripts/generate_real_latents.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --output real_inputs/gt_latents.pt
```

### Step 2 — Generate Video

```bash
python scripts/generate_video.py \
    --eeg_data /path/to/test_eeg.pt \
    --text_embeddings text_embeddings.pt \
    --visual_latents visual_latents.pt \
    --is_fast is_fast.pt \
    --unet_ckpt checkpoints/unet_finetuned \
    --vae_path ./modelscope_weights/vae \
    --output_dir outputs/ \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --ddim
```

### End-to-End Pipeline Test

```bash
python pipelines/run_pipeline_test.py \
    --sample_idx 42 \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed
# Runs the full pipeline for a single sample and saves output GIF to outputs/
```

### Inference Flow (Code-Level)

```python
# 1. Load conditioning tensors
text_emb   = torch.load("text_embeddings.pt")    # (N, 77, 768)
vis_latent = torch.load("visual_latents.pt")      # (N, 6, 4, 32, 32)
is_fast    = torch.load("is_fast.pt")             # (N, 1)

# 2. DANA: add dynamics-aware noise to ViT latents
dana = DANAModule()
noised_latent, beta = dana(vis_latent[i:i+1], is_fast[i:i+1])

# 3. DDIM denoising loop with EEG cross-attention conditioning
scheduler = DDIMScheduler(...)
scheduler.set_timesteps(50)
latent = noised_latent.reshape(1, 6*4, H, W)

for t in scheduler.timesteps:
    noise_pred = unet(latent, t, encoder_hidden_states=text_emb[i:i+1]).sample
    latent = scheduler.step(noise_pred, t, latent).prev_sample

# 4. VAE decode each frame
frames = []
for frame_latent in latent.reshape(6, 4, H, W):
    frame = vae.decode(frame_latent.unsqueeze(0) / 0.18215).sample
    frames.append(frame)   # each: (1, 3, 128, 128)

# 5. Save output
save_video(frames, "outputs/generated_clip.gif", fps=3)
```

---

## 💾 Checkpoint Management

### Directory Layout

```
checkpoints/
├── unet_best/
│   ├── config.json                 # UNet2DConditionModel architecture config
│   ├── diffusion_pytorch_model.bin # Best validation checkpoint
│   └── scheduler_config.json
├── unet_ema/
│   └── ema_weights.pt             # EMA-averaged weights for inference
├── text_mlp_final.pth             # TextProjectorMLP (Sub-team 4)
├── eeg_adapter.pth                # EEGAdapter dual-path model
├── dynamics_model.pth             # DynamicsClassifier
└── vit_seq2seq.pth                # ViT Seq2Seq latent predictor
```

### Loading for Inference

```python
from diffusers import UNet2DConditionModel
from models.decoder.projection import EEGProjection
from models.dynamics_predictor.subteam4_models import TextProjectorMLP, EEGAdapter

unet = UNet2DConditionModel.from_pretrained("checkpoints/unet_best")
unet.eval().to(device)

eeg_adapter = EEGAdapter(output_dim=512)
eeg_adapter.load_state_dict(torch.load("checkpoints/eeg_adapter.pth"))

# Load EMA weights for inference (preferred over raw checkpoint)
ema_weights = torch.load("checkpoints/unet_ema/ema_weights.pt")
unet.load_state_dict(ema_weights)
```

---

## 🔧 Configuration & Hyperparameters

```yaml
# EEG Encoder
eeg_adapter:
  flat_dim: 28458        # 62 × 51 × 9
  band_dim: 310          # 62 channels × 5 frequency bands
  output_dim: 512
  dropout: 0.3

# EEG Projection (SD conditioning)
eeg_projection:
  input_dim: 512
  output_dim: 768
  seq_len: 77
  num_heads: 12
  num_layers: 4
  dropout: 0.05

# VideoLDM Training
training:
  batch_size: 2
  grad_accum: 4           # effective batch = 8
  learning_rate: 1.0e-4
  num_epochs: 10
  mixed_precision: fp16
  use_ema: true
  ema_decay: 0.9999
  subset: 20000

# DANA
dana:
  beta_fast: 0.85
  beta_slow: 0.35

# Diffusion
diffusion:
  num_train_timesteps: 1000
  num_inference_steps: 50
  guidance_scale: 7.5
  beta_schedule: linear
  scheduler: ddim         # ddpm for training, ddim for inference

# Data
data:
  n_channels: 62
  n_freq_bins: 51
  n_time_frames: 9
  n_video_frames: 6
  latent_height: 32
  latent_width: 32

# Loss weights (Text MLP)
loss:
  lambda_mse: 0.5
  lambda_cosine: 0.5
```

---

## 📏 Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **SSIM** | Structural Similarity Index — luminance, contrast, structure ↑ | ≥ 0.30 |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) ↑ | ≥ 20 dB |
| **LPIPS** | Learned Perceptual Image Patch Similarity ↓ | < 0.40 |
| **FID** | Fréchet Inception Distance ↓ | < 100 |
| **CLIP Cosine Sim** | Cosine similarity between CLIP embeddings of generated and GT images ↑ | ≥ 0.10 |
| **MSE (Latent)** | Per-sample MSE between predicted and GT VAE latents ↓ | < 0.20 |

### Running Evaluation

```bash
python evaluation/evaluate.py \
    --generated_dir outputs/ \
    --gt_dir real_inputs/ \
    --metrics ssim psnr lpips fid clip_sim \
    --output_json evaluation/eval_results/metrics_video.json \
    --device cuda

# Latent-space evaluation
python evaluation/evaluate.py \
    --mode latent \
    --pred_latents text_embeddings.pt \
    --gt_latents real_inputs/gt_latents.pt \
    --output_json evaluation/eval_results/metrics_latent.json
```

---

## 📈 Results

### Quantitative Metrics

| Metric | Full Model (DANA + ViT) | w/o DANA | w/o Vision Transformer |
|---|---|---|---|
| **SSIM ↑** | **0.6967** | 0.5459 | 0.4647 |
| **PSNR ↑** | **29.17 dB** | 20.14 dB | 16.53 dB |
| **LPIPS ↓** | **0.3526** | 0.4004 | 0.5400 |
| **FID ↓** | **24.50** | — | — |

**Latent-space evaluation** (test split, ~100 samples):

| Metric | Mean | Min | Max |
|---|---|---|---|
| Cosine Similarity ↑ | 0.0416 | −0.0910 | 0.1163 |
| MSE ↓ | 0.2247 | 0.1846 | 0.4152 |

> The SSIM/PSNR/LPIPS/FID metrics reflect VideoLDM Phase 4 evaluation. The TemporalUNet approaches the architectural target of 0.0005 MSE over 5 training epochs, validating tensor shape contracts and gradient flow. Low latent cosine similarity reflects an early-stage CLIP alignment; the Text MLP requires ~10–15 epochs to produce strongly aligned embeddings.

### SOTA Comparison (EEG2Video Paper)

| Metric | EEG2Video (SOTA) |
|---|---|
| 2-way Video Semantic Accuracy | 79.8% |
| 40-way Video Semantic Accuracy | 15.9% |
| SSIM | 0.256 (0.300 on 10-class subset) |

---

## ⚠ Challenges & Mitigations

| Challenge | Root Cause | Fix Applied |
|---|---|---|
| **DataLoader RAM crash** | `num_workers > 0` causes memory explosion with per-file `.pt` loading on 291k samples | Set `num_workers=0`; use `TRAIN_SUBSET=20000` |
| **GPU memory overflow** | Full 6-frame batch through UNet2DConditionModel exceeds 16 GB | fp16 mixed precision + `GRAD_ACCUM=4` + `BATCH_SIZE=2` |
| **Wrong frame grouping in VAE latents** | Reshaping a flat tensor groups unrelated frames | Switched to per-sample `video_sample_XXXXXX.pt` loading |
| **CLIP alignment instability** | EEG Adapter collapsed to near-zero embeddings early in training | Raw CLIP embedding injection as shortcut; train Adapter + MLP jointly |
| **Mode collapse in generation** | Diffusion model ignores EEG conditioning | Classifier-free guidance (dropout p=0.1); guidance scale=7.5 |
| **VAE scale mismatch** | Synthetic latents had std≈0.18 vs real VAE std≈0.9–5.0 | Correct normalization: `latent / 0.18215` per SD convention |
| **Wrong tensor alignment** | Sub-team 3 and Sub-team 4 outputs in different sample orders | Strict alphabetical sort enforced in `AlignedInferenceDataset` |

---

## 👥 Team Contributions

| Member(s) | Sub-team | Responsibilities |
|---|---|---|
| **Milan Jadav & Kamesh Singh** | Sub-team 1 — EEG Preprocessing | MNE-Python bandpass filtering, ICA artifact removal, STFT computation, epoch windowing, spectrogram validation, `data/dataset.py` |
| **Shivam Gami** | Sub-team 2 — VideoLDM | SD v1.5 UNet fine-tuning, EEGProjection, DANA integration, DDPM/DDIM training loop, mixed precision, EMA, checkpoint management (`training/train_videoldm.py`, `models/diffusion_backbone/`) |
| **Manan Sahni** | Sub-team 3 — ViT Seq2Seq | Vision Transformer encoder, Seq2Seq LSTM latent predictor, visual latent generation (`models/eeg_encoder/vision_transformer.py`, `scripts/generate_visual_latents.py`) |
| **Vishal Meena** | Sub-team 4 — Dynamics + Text MLP | Dual-path EEG Adapter, TextProjectorMLP, DynamicsClassifier, frequency-band pooling, W&B sweeps (`models/dynamics_predictor/`, `training/train_text_mlp.py`, `training/train_dynamics_mlp.py`) |
| **Rishab Bagul & Shivam Shingla** | Evaluation | SSIM/PSNR/LPIPS/FID/CLIP-sim scripts, latent-space evaluation, oracle test, ablation experiments (`evaluation/evaluate.py`, `utils/oracle_test.py`) |
| **Vipresh Gupta** | Project Lead / Integration | Module interface contracts, tensor alignment (`AlignedInferenceDataset`), pipeline integration test, final coordination (`pipelines/run_pipeline_test.py`) |
| **Sarthak Kardam & Raghav Bansal** | Technical Writing | Documentation, README, mid-project report, final report, presentation slides |

---

## 🔮 Future Work

- **Temporal Attention Modules:** Insert 3D attention layers between spatial UNet blocks to enforce inter-frame consistency (full VideoLDM extension)
- **Subject-Adaptive Conditioning:** Add learnable subject ID embedding to handle inter-subject EEG variability
- **LoRA Fine-tuning:** Memory-efficient full UNet fine-tuning using Low-Rank Adaptation
- **Higher Resolution Output:** Scale from 128×128 to 256×256 using SD v2.1 or SDXL backbone
- **Contrastive EEG Pre-training:** Pre-train EEG Adapter with CLIP contrastive loss on large-scale EEG datasets
- **Real-Time Inference:** Optimize DANA + DDIM pipeline via distillation / consistency models for < 1 second per clip

---

## 🛠 Troubleshooting

**`RuntimeError: CUDA out of memory` during training**

```bash
python training/train_videoldm.py --batch_size 1 --grad_accum 8 \
    --gradient_checkpointing --mixed_precision fp16
```

**DataLoader worker killed**

```python
# Set in train_videoldm.py:
DataLoader(..., num_workers=0, pin_memory=False)
```

**Generated videos are all identical (mode collapse)**

```python
guidance_scale = 10.0  # default 7.5; try 10–15 for more diversity
print(is_fast.mean())  # should be between 0.2 and 0.8 for healthy split
```

**`cosine_sim` in metrics_latent.json is near 0 or negative**

Expected at early training stages. Monitor `val_cosine_sim` in W&B — it should trend positive by epoch 5 with `lr=1e-3`.

**Tensor shape mismatch between sub-team outputs**

```bash
python -c "
import torch
vis = torch.load('visual_latents.pt')
txt = torch.load('text_embeddings.pt')
isf = torch.load('is_fast.pt')
print(vis.shape[0], txt.shape[0], isf.shape[0])  # Must all be equal
"
```

---

## 🙏 Acknowledgements

- **Prof. [Mentor Name]**, IIT Mandi — for project mentorship, dataset access, and technical guidance
- **IITM HPC Cluster** — A100 GPU compute resources for VideoLDM training
- **RunwayML** — Stable Diffusion v1.5 pre-trained weights
- **OpenAI** — CLIP pre-trained visual-language embedding model
- **HuggingFace 🤗** — `diffusers` library for SD pipeline and DDPM/DDIM schedulers
- **Weights & Biases** — Experiment tracking and hyperparameter sweeps

---

## 📖 References

```
[1] Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.

[2] Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.

[3] Chen, Y., et al. (2023). DreamDiffusion: Generating High-Quality Images from Brain EEG Signals. arXiv:2306.16934.

[4] Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. Journal of Neural Engineering, 15(5).

[5] Blattmann, A., et al. (2023). Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. CVPR 2023.

[6] Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.

[7] Song, J., et al. (2021). Denoising Diffusion Implicit Models. ICLR 2021.

[8] Wu, J., et al. (2023). Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation. ICCV 2023.

[9] Palazzo, S., et al. (2020). Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features. IEEE TPAMI (SEED-DV / EEGCVPR40 dataset).
```

---

<div align="center">

*B.Tech CS671 · Group 22 · Indian Institute of Technology, Mandi · 2025–26*

</div>
