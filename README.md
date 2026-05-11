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
*Group 2 — EEG Transformer + Latent Diffusion Model Pipeline*

---

</div>

<br/>

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture-overview)
- [Key Concepts](#-key-concepts-glossary)
- [Dataset](#-dataset--seed-dv--eeg-imagenet)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Phase 1 — Setup & Literature Review](#-phase-1--setup--literature-review)
- [Phase 2 — EEG Preprocessing & Spectrogram Generation](#-phase-2--eeg-preprocessing--spectrogram-generation)
- [Phase 3 — Transformer Encoder + CLIP Alignment](#-phase-3--transformer-encoder--clip-alignment)
- [Phase 4 — Latent Diffusion Fine-Tuning & Video Generation](#-phase-4--latent-diffusion-fine-tuning--video-generation)
- [Phase 5 — Evaluation & Final Report](#-phase-5--evaluation--final-report)
- [Training Pipeline](#-training-pipeline)
- [Inference Pipeline](#-inference-pipeline)
- [Checkpoint Management](#-checkpoint-management)
- [Configuration & Hyperparameters](#-configuration--hyperparameters)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Experiments & Ablations](#-experiments--ablations)
- [Challenges & Mitigations](#-challenges--mitigations)
- [Team Contributions](#-team-contributions)
- [Future Work](#-future-work)
- [Troubleshooting](#-troubleshooting)
- [Acknowledgements](#-acknowledgements)
- [References](#-references)
- [Citation](#-citation)

---

## 🔭 Overview

**EEG2Video** is a research-grade generative pipeline that reconstructs temporally coherent video sequences directly from non-invasive EEG brain recordings. Given a 500 ms window of 62-channel EEG data recorded while a subject watches a video, the system generates a 6-frame video clip at 3 FPS that semantically corresponds to the observed visual stimulus.

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

- **DANA (Dynamic-Aware Noise Addition):** Instead of starting denoising from pure Gaussian noise, the system uses the ViT-predicted visual latent as a structural prior, adding controlled noise scaled by the predicted motion dynamics (`β_fast = 0.85` for dynamic content, `β_slow = 0.35` for static content). This allows the diffusion model to refine structure rather than hallucinate from scratch.
- **Dual-Path EEG Adapter:** A flat spectral path (28,458 raw time-frequency values) is fused with a band-pooled path (310 frequency-band features across Delta/Theta/Alpha/Beta/Gamma) to produce a robust 512-dim EEG embedding.
- **Latent-Space Conditioning:** EEG embeddings are projected to 77 × 768 context tensors via learnable Transformer queries, injected into the Stable Diffusion UNet via cross-attention at every denoising step — identical to text conditioning in standard SD.

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

> 📌 **Architecture diagram placeholder** — replace with `assets/architecture.png`
>
> | Full Pipeline Diagram | DANA Scheduler Flow |
> |---|---|
> | `![Architecture](assets/architecture.png)` | `![DANA](assets/dana_flow.png)` |

---

## 📚 Key Concepts Glossary

| Concept | Description |
|---|---|
| **STFT Spectrogram** | Short-Time Fourier Transform applied per EEG channel, converting the 1D time-series into a 2D time-frequency map (freq bins × time frames). Enables the Transformer to process EEG as image-like patches. |
| **EEG Embedding** | 512-dimensional dense vector representation of the EEG spectrogram produced by the Dual-Path EEG Adapter. Encodes both full spectral detail and frequency-band structure. |
| **Transformer Attention** | Multi-head self-attention (8 heads, d=256) applied across temporal patches of the EEG spectrogram, capturing long-range temporal dependencies within and across EEG channels. |
| **CLIP Alignment** | The EEG embedding is projected via an MLP into CLIP's 512-dim visual-language latent space. Cosine similarity loss enforces that the EEG representation aligns semantically with the ground-truth image embedding. |
| **ViT Latents** | A Vision Transformer (ViT) encodes reference visual frames and a Seq2Seq network predicts 6 VAE-encoded latent frames `(6 × 4 × 32 × 32)` from the EEG embedding — providing structural guidance to the diffusion model. |
| **DANA Scheduler** | Dynamic-Aware Noise Addition: adds β-scaled Gaussian noise to ViT-predicted latents based on the predicted motion label (`is_fast`). `β_fast=0.85` (high noise for dynamic content), `β_slow=0.35` (low noise for static content). |
| **DDPM/DDIM Sampling** | Denoising Diffusion Probabilistic Models (DDPM) for training; DDIM for accelerated inference. The UNet iteratively denoises the DANA-noised latent conditioned on EEG context. |
| **Classifier-Free Guidance** | During training, EEG conditioning is randomly dropped (p=0.1), training the UNet to work both conditioned and unconditionally. At inference, guidance scale amplifies the conditioned direction. |
| **EEGProjection** | A 4-layer Transformer with 77 learnable latent queries that converts the 512-dim EEG embedding into a `(77 × 768)` context tensor — matching the token shape of CLIP text embeddings for cross-attention injection. |
| **EMA Checkpoints** | Exponential Moving Average of model weights used during training for more stable inference. Saved separately from the training checkpoint. |
| **TemporalUNet** | The VideoLDM extension — the SD v1.5 UNet2DConditionModel processes all 6 frames independently in the spatial dimension; temporal consistency is enforced via DANA's structural priors and latent-space regularization. |
| **VAE Encoder/Decoder** | The Stable Diffusion Variational Autoencoder encodes 128×128 RGB frames into `(4 × 16 × 16)` or `(4 × 32 × 32)` latent tensors and decodes generated latents back to pixel space. |
| **Cross-Attention Conditioning** | At every UNet denoising step, the `(77 × 768)` EEG context tensor is injected into the UNet's attention layers as key-value pairs, with spatial features as queries — identical to text-to-image SD conditioning. |

---

## 🗃 Dataset — SEED-DV / EEG-ImageNet

This project uses the **SEED-DV EEG-video dataset** (also referenced as EEG-ImageNet / EEGCVPR40 for preprocessing compatibility).

### Dataset Statistics

| Property | Value |
|---|---|
| EEG Channels | 62 |
| Sampling Rate | ~1000 Hz (downsampled during preprocessing) |
| Window Size | 500 ms sliding windows |
| Window Overlap | 50% |
| Frames per Sample | 6 (2-second video clips @ 3 FPS) |
| Total Samples | ~291,000 processed samples |
| Train / Val / Test Split | 70% / 15% / 15% (random seed 42) |
| EEG Tensor Shape | `(62, 51, 9)` — channels × freq bins × time frames |
| Video Latent Shape | `(6, 4, 16, 16)` or `(6, 4, 32, 32)` |
| Text Embedding Shape | `(77, 768)` — CLIP token space |
| Visual Output Resolution | 128 × 128 RGB |

### Dataset Preparation

The processed dataset directory is expected at:

```
/home/teaching/TEAM_22_DATASET/processed/processed/
├── eeg_sample_000001.pt        # (62, 51, 9) EEG spectrogram tensor
├── video_sample_000001.pt      # (6, 4, 16, 16) VAE-encoded video latent
├── text_sample_000001.pt       # (77, 768) CLIP text embedding
├── dynamics_labels_fixed_BINARY.npy   # Binary fast/slow motion labels
├── train_split.txt
├── val_split.txt
└── test_split.txt
```

### EEG Preprocessing Details

Raw EEG signals are preprocessed using `MNE-Python` and `SciPy`:

1. **Bandpass Filtering:** 0.5–45 Hz (removes DC drift and high-frequency noise)
2. **Epoch Windowing:** 500 ms non-overlapping segments (or 50% overlap for video generation)
3. **STFT Computation:** `scipy.signal.stft` applied per channel → `(51 freq bins × 9 time frames)`
4. **Normalization:** Per-channel z-score normalization across the spectrogram
5. **Stacking:** All 62 channels stacked → `(62, 51, 9)` tensor saved as `.pt`

**Expected tensor shapes at each pipeline stage:**

```python
eeg_spectrogram   : torch.Size([B, 62, 51, 9])     # input to EEG Adapter
eeg_embedding     : torch.Size([B, 512])             # EEG Adapter output
clip_context      : torch.Size([B, 77, 768])         # EEGProjection output → UNet
visual_latents    : torch.Size([B, 6, 4, 32, 32])   # ViT Seq2Seq output
text_embeddings   : torch.Size([B, 77, 768])         # Text MLP output (saved)
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
│   └── dataset.py                  # EEGVideoDataset class — 70/15/15 splits
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
│   ├── evaluate.py                 # SSIM, PSNR, FVD, FID, CLIP-sim evaluation
│   └── eval_results/
│       ├── metrics_latent.json     # Per-sample cosine similarity + MSE
│       └── metrics_video.json      # Aggregate video-level metrics
│
├── 📁 utils/
│   ├── subteam2_dataset.py         # VideoLDM-specific DataLoader
│   ├── myvideo.py                  # Video I/O utilities (frame assembly, GIF export)
│   ├── oracle_test.py              # Upper-bound evaluation with ground-truth latents
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

Or use the provided setup script (Sub-team 2 environment):

```bash
bash utils/setup_subteam2.sh
```

### Step 4 — Download Pre-trained Backbone

```bash
# Stable Diffusion v1.5 weights (required for VideoLDM backbone)
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

## 📖 Phase 1 — Setup & Literature Review

**Weeks 1–2 | Objectives:** Environment setup, repository initialization, literature survey, dataset download and verification.

### Objectives

- Initialize GitHub repository with consistent module interfaces
- Complete literature review of key papers: DreamDiffusion, CLIP, Latent Diffusion Models, EEGNet, VideoLDM
- Download and verify the SEED-DV / EEG-ImageNet dataset
- Implement and test the base `EEGVideoDataset` DataLoader

### Key Papers

| Paper | Relevance |
|---|---|
| Rombach et al. (2022) — *High-Resolution Image Synthesis with Latent Diffusion Models* | Core diffusion backbone (Stable Diffusion) |
| Radford et al. (2021) — *Learning Transferable Visual Models From Natural Language Supervision* | CLIP semantic alignment |
| Chen et al. (2023) — *DreamDiffusion: Generating High-Quality Images from Brain EEG Signals* | EEG-to-image generation with diffusion |
| Lawhern et al. (2018) — *EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs* | EEGNet-inspired Transformer encoder |
| Blattmann et al. (2023) — *Align your Latents: High-Resolution Video Synthesis with LDMs* | VideoLDM temporal extension |

### Deliverables

- [ ] Repository structure confirmed with agreed tensor I/O contracts
- [ ] Literature summary (3–5 papers, 1-page each)
- [ ] `EEGVideoDataset` unit-tested with correct 70/15/15 splits
- [ ] Dataset integrity check: all `eeg_sample_*.pt`, `video_sample_*.pt`, `text_sample_*.pt` files verified

### Dataset Verification

```bash
python -c "
from data.dataset import EEGVideoDataset
ds = EEGVideoDataset('/home/teaching/TEAM_22_DATASET/processed/processed', mode='train')
eeg, text, video, label = ds[0]
print('EEG shape:  ', eeg.shape)    # Expected: (62, 51, 9)
print('Text shape: ', text.shape)   # Expected: (77, 768)
print('Video shape:', video.shape)  # Expected: (6, 4, 16, 16)
print('Label:      ', label.item()) # Expected: 0.0 or 1.0
"
```

---

## 🔬 Phase 2 — EEG Preprocessing & Spectrogram Generation

**Weeks 3–4 | Objectives:** Implement the complete EEG preprocessing pipeline from raw signals to STFT spectrograms.

### Pipeline Steps

```
Raw EEG Signal
[62 channels × N_samples @ ~1000 Hz]
        │
        ├─ 1. Bandpass Filter (0.5–45 Hz) — MNE-Python
        │
        ├─ 2. Epoch Segmentation (500 ms windows, 50% overlap)
        │      → Each epoch: (62 × 500 samples)
        │
        ├─ 3. STFT Spectrogram (scipy.signal.stft per channel)
        │      → Each channel: (51 freq bins × 9 time frames)
        │
        ├─ 4. Per-channel Z-score Normalization
        │
        └─ 5. Stack channels → (62 × 51 × 9) tensor, saved as .pt
```

### STFT Parameters

```python
from scipy import signal
import torch

def compute_stft_spectrogram(eeg_epoch, fs=250, nperseg=32, noverlap=16):
    """
    Convert a single EEG epoch to STFT spectrogram.

    Args:
        eeg_epoch : np.ndarray of shape (n_channels, n_samples)
        fs        : sampling frequency after downsampling (Hz)
        nperseg   : STFT window length (samples)
        noverlap  : STFT overlap (samples)

    Returns:
        torch.Tensor of shape (n_channels, n_freq_bins, n_time_frames)
        → (62, 51, 9) for default params
    """
    specs = []
    for ch in range(eeg_epoch.shape[0]):
        f, t, Zxx = signal.stft(eeg_epoch[ch], fs=fs,
                                 nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)
        # Z-score normalization per channel
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-6)
        specs.append(magnitude)
    return torch.tensor(np.stack(specs), dtype=torch.float32)
    # Output shape: (62, 51, 9)
```

### Frequency Band Structure

The Dynamics Predictor additionally uses frequency-band pooling across 5 bands:

| Band | Frequency Range | Index in (51 bins) |
|---|---|---|
| Delta | 0–4 Hz | bins 0–3 |
| Theta | 4–8 Hz | bins 4–7 |
| Alpha | 8–14 Hz | bins 8–13 |
| Beta | 14–31 Hz | bins 14–30 |
| Gamma | 31–45 Hz | bins 31–50 |

### Deliverables

- [ ] Clean EEG epochs saved as `eeg_sample_XXXXXX.pt` (shape `62 × 51 × 9`)
- [ ] `EEGVideoDataset` unit-tested: DataLoader iterates without shape errors
- [ ] Spectrogram visualizations validated per channel (no flat/zero channels)
- [ ] `dynamics_labels_fixed_BINARY.npy` — binary fast/slow motion labels generated

---

## 🤖 Phase 3 — Transformer Encoder + CLIP Alignment

**Week 5 | Objectives:** Train the EEG Transformer encoder and align its output embeddings to CLIP's visual-language latent space.

### EEG Transformer Architecture

The `EEGTransformer` (Sub-team 1/2) takes the spectrogram tensor and produces sequential token embeddings:

```python
# models/eeg_encoder/eeg_transformer.py
class EEGTransformer(nn.Module):
    """
    Input : (B, T, C, S) = (B, 7, 62, 100) — temporal windows × channels × samples
    Output: (B, T*S, latent_dim)            — sequential token embeddings

    Architecture:
        Linear projection : C → 256
        TransformerEncoder: 4 layers, 8 heads, d_model=256
        Output projection : 256 → latent_dim
    """
```

The `EEGAdapter` (Sub-team 4) uses a **dual-path architecture**:

```
EEG Spectrogram (62 × 51 × 9)
         │
  ┌──────┴──────┐
  │             │
Flat Path    Band Path
(28458→512   (310→256)
  →256)          │
  │             │
  └──────┬──────┘
         │  concat(256+256)
         ▼
     Fusion MLP (512→512)
         │
    EEG Embedding (512-dim)
```

- **Flat path:** Preserves all 28,458 raw time-frequency values → highest information retention (achieved best val_loss = 0.675)
- **Band path:** Delta/Theta/Alpha/Beta/Gamma mean per channel → 310-dim spectral structure vector

### CLIP Alignment

The `EEGProjection` module converts the 512-dim EEG embedding into a `(77 × 768)` context tensor compatible with SD's cross-attention:

```python
# models/decoder/projection.py — EEGProjection
# Input : (B, 512)
# Output: (B, 77, 768)

# Step 1: Deep MLP — 512 → 1536 → 768
# Step 2: 77 Learnable Latent Queries — (1, 77, 768) expanded to (B, 77, 768)
# Step 3: Inject EEG context → queries + eeg_features.unsqueeze(1)
# Step 4: 4-layer Transformer Encoder (12 heads, d_model=768)
# Step 5: Output projection + LayerNorm
```

**Training objective (Text MLP / CLIP alignment):**

```python
# Cosine similarity alignment loss
loss = 1 - F.cosine_similarity(eeg_embedding, clip_image_embedding, dim=-1).mean()

# Combined loss for Text MLP
loss_total = λ1 * loss_mse + λ2 * loss_cosine
# λ1 = 0.5, λ2 = 0.5 (tuned via W&B sweep)
```

### Training the Text MLP (CLIP Embedding Predictor)

```bash
# Linux / macOS
python training/train_text_mlp.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --batch_size 128 \
    --lr 1e-3 \
    --epochs 30 \
    --output_dir ./checkpoints/text_mlp

# PowerShell (Windows)
python training/train_text_mlp.py `
    --data_dir "C:\datasets\eeg_processed" `
    --batch_size 128 --lr 1e-3 --epochs 30
```

### Training the Dynamics Classifier

```bash
python training/train_dynamics_mlp.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --batch_size 256 \
    --lr 5e-4 \
    --epochs 20 \
    --output_dir ./checkpoints/dynamics
```

### Deliverables

- [ ] Trained EEG Adapter checkpoint: `checkpoints/eeg_adapter.pth`
- [ ] Trained Text MLP checkpoint: `checkpoints/text_mlp_final.pth`
- [ ] Trained Dynamics Classifier checkpoint: `checkpoints/dynamics_model.pth`
- [ ] CLIP cosine similarity ≥ 0.55 on validation set
- [ ] W&B training curves logged

---

## 🎬 Phase 4 — Latent Diffusion Fine-Tuning & Video Generation

**Weeks 6–7 | Objectives:** Integrate EEG embeddings into Stable Diffusion's cross-attention, train the VideoLDM with DANA conditioning, and generate video frame sequences.

### DANA — Dynamic-Aware Noise Addition

The DANA module replaces the standard "start-from-pure-noise" diffusion initialization:

```python
# models/diffusion_backbone/dana.py

BETA_FAST = 0.85   # High noise for dynamic/fast content
BETA_SLOW = 0.35   # Low noise for static/slow content

class DANAModule(nn.Module):
    """
    Inputs:
        visual_latent : (B, 6, 4, 32, 32)  — from Sub-team 3 ViT Seq2Seq
        is_fast       : (B, 1)              — from Sub-team 4 Dynamics MLP
                        0.0 = slow motion, 1.0 = fast motion

    Output:
        noised_latent : (B, 6, 4, 32, 32)  — starting point for DDPM denoising
        beta_used     : (B,)               — β value applied per sample

    Noising formula:
        noised = sqrt(1 - β²) * latent + β * N(0, I)
        # β_slow → preserves ~65% of structural signal
        # β_fast → preserves ~15%, UNet fills in dynamic content
    """
```

**Why DANA matters:** Without DANA, the UNet must generate all visual content from pure Gaussian noise conditioned only on the weak EEG signal. With DANA, the ViT-predicted latent provides structural scaffolding — the UNet only needs to refine and condition, not hallucinate from scratch.

### VideoLDM Training

The `UNet2DConditionModel` from Stable Diffusion v1.5 is fine-tuned with **frozen backbone weights** and **trainable cross-attention layers only**:

```python
# training/train_videoldm.py — key hyperparameters
BATCH_SIZE    = 2       # per-GPU batch (GPU memory constraint)
GRAD_ACCUM    = 4       # effective batch = 8 samples
LR            = 1e-4    # learning rate
NUM_EPOCHS    = 10
NUM_FRAMES    = 6       # frames per video clip
MAX_GRAD_NORM = 1.0     # gradient clipping

# Frozen: all UNet spatial conv/norm layers
# Trainable: all CrossAttention key/value projection layers
#            EEGProjection module
```

**Training strategy:**

```python
# Freeze backbone, unfreeze cross-attention
for name, param in unet.named_parameters():
    if "attn2" in name:  # cross-attention layers
        param.requires_grad = True
    else:
        param.requires_grad = False

# EEGProjection is always trainable
for param in eeg_proj.parameters():
    param.requires_grad = True
```

### Launch VideoLDM Training

```bash
# Linux (single GPU)
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

# PowerShell
python training/train_videoldm.py `
    --data_dir "C:\datasets\eeg_processed" `
    --batch_size 2 --grad_accum 4 --lr 1e-4 `
    --epochs 10 --mixed_precision fp16 --use_ema
```

**Mixed precision + gradient checkpointing** (required for GPUs < 24 GB VRAM):

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast(dtype=torch.float16):
    noise_pred = unet(noised_latent_flat, timesteps, context).sample
    loss = F.mse_loss(noise_pred, noise_target)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
scaler.step(optimizer)
scaler.update()
```

### Training the ViT Seq2Seq Latent Predictor

```bash
python training/train_vit_seq2seq.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --output_dir ./checkpoints/vit_seq2seq
```

### Deliverables

- [ ] Fine-tuned VideoLDM checkpoint: `checkpoints/unet_finetuned/`
- [ ] ViT Seq2Seq checkpoint: `checkpoints/vit_seq2seq.pth`
- [ ] EMA checkpoint: `checkpoints/unet_ema/`
- [ ] Sample video sequences generated from test EEG windows
- [ ] W&B training loss curves (diffusion MSE loss per epoch)

---

## 📊 Phase 5 — Evaluation & Final Report

**Week 8 | Objectives:** Full evaluation on the official test split, ablation studies, and final report preparation.

### Generating Conditioning Tensors for Inference

Before running the full inference pipeline, export all conditioning tensors from the trained sub-team models:

```bash
# Step 1: Export text_embeddings.pt and is_fast.pt (Sub-team 4)
python scripts/generate_tensors.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --text_mlp_path checkpoints/text_mlp_final.pth \
    --adapter_path checkpoints/eeg_adapter.pth \
    --dynamics_path checkpoints/dynamics_model.pth \
    --out_text text_embeddings.pt \
    --out_dynamics is_fast.pt

# Step 2: Export visual_latents.pt (Sub-team 3 ViT Seq2Seq)
python scripts/generate_visual_latents.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --vit_checkpoint checkpoints/vit_seq2seq.pth \
    --output visual_latents.pt
# Output shape: (N_test_samples, 6, 4, 32, 32)

# Step 3: Generate ground-truth VAE latents for oracle evaluation
python scripts/generate_real_latents.py \
    --data_dir /home/teaching/TEAM_22_DATASET/processed/processed \
    --output real_inputs/gt_latents.pt
```

---

## 🚀 Inference Pipeline

### End-to-End Video Generation

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

### Pipeline Integration Test

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
latent = noised_latent.reshape(1, 6*4, H, W)   # flatten frames

for t in scheduler.timesteps:
    noise_pred = unet(latent, t, encoder_hidden_states=text_emb[i:i+1]).sample
    latent = scheduler.step(noise_pred, t, latent).prev_sample

# 4. VAE decode each frame
frames = []
latent_frames = latent.reshape(6, 4, H, W)
for frame_latent in latent_frames:
    frame = vae.decode(frame_latent.unsqueeze(0) / 0.18215).sample
    frames.append(frame)   # each: (1, 3, 128, 128)

# 5. Save as GIF / MP4
save_video(frames, "outputs/generated_clip.gif", fps=3)
```

---

## 💾 Checkpoint Management

### Checkpoint Directory Layout

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

### Loading Checkpoints for Inference

```python
from diffusers import UNet2DConditionModel
from models.decoder.projection import EEGProjection
from models.dynamics_predictor.subteam4_models import TextProjectorMLP, EEGAdapter

# Load UNet
unet = UNet2DConditionModel.from_pretrained("checkpoints/unet_best")
unet.eval().to(device)

# Load EEG modules
eeg_adapter = EEGAdapter(output_dim=512)
eeg_adapter.load_state_dict(torch.load("checkpoints/eeg_adapter.pth"))

eeg_proj = EEGProjection(input_dim=512, output_dim=768, seq_len=77)
# (EEGProjection weights are stored inside the UNet checkpoint after fine-tuning)

# Load EMA weights for inference (preferred over raw checkpoint)
ema_weights = torch.load("checkpoints/unet_ema/ema_weights.pt")
unet.load_state_dict(ema_weights)
```

---

## 🔧 Configuration & Hyperparameters

### YAML Config (`configs/train_config.yaml`)

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
  save_every: 5
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
  latent_height: 16       # or 32 depending on processed/ version
  latent_width: 16

# Loss weights (Text MLP)
loss:
  lambda_mse: 0.5
  lambda_cosine: 0.5
```

### W&B Sweep Example

```yaml
# configs/sweep_config.yaml
program: training/sweep_train.py
method: bayes
metric:
  name: val_cosine_sim
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  dropout:
    values: [0.1, 0.2, 0.3, 0.4]
  lambda_cosine:
    values: [0.3, 0.5, 0.7, 1.0]
  batch_size:
    values: [64, 128, 256]
```

```bash
wandb sweep configs/sweep_config.yaml
wandb agent <sweep-id>
```

---

## 📏 Evaluation Metrics

All metrics are computed on the official test split (15% of ~291k samples).

| Metric | Category | Description | Target |
|---|---|---|---|
| **SSIM** | Visual Quality | Structural Similarity Index — luminance, contrast, structure. Range 0–1 ↑ | ≥ 0.30 |
| **PSNR** | Visual Quality | Peak Signal-to-Noise Ratio in dB ↑ | ≥ 20 dB |
| **FID** | Distribution | Fréchet Inception Distance — Inception-v3 feature distribution ↓ | < 100 |
| **FVD** | Temporal | Fréchet Video Distance — I3D features across frames ↓ | < 1000 |
| **CLIP Cosine Sim** | Semantic | Cosine similarity between CLIP embeddings of generated and GT images ↑ | ≥ 0.10 |
| **Top-5 Accuracy** | Semantic | Inception-v3 top-5 classification accuracy on generated frames ↑ | ≥ 10% |
| **MSE (Latent)** | Reconstruction | Per-sample MSE between predicted and GT VAE latents ↓ | < 0.20 |

### Running Evaluation

```bash
python evaluation/evaluate.py \
    --generated_dir outputs/ \
    --gt_dir real_inputs/ \
    --metrics ssim psnr fid fvd clip_sim top5 \
    --output_json evaluation/eval_results/metrics_video.json \
    --device cuda

# Latent-space evaluation (faster, no VAE decode needed)
python evaluation/evaluate.py \
    --mode latent \
    --pred_latents text_embeddings.pt \
    --gt_latents real_inputs/gt_latents.pt \
    --output_json evaluation/eval_results/metrics_latent.json
```

---

## 📈 Results

### Quantitative Results

> 📌 **Placeholder** — fill in after final evaluation run

| Model Variant | SSIM ↑ | PSNR ↑ | FID ↓ | FVD ↓ | CLIP Sim ↑ | Top-5 Acc ↑ |
|---|---|---|---|---|---|---|
| Baseline (no conditioning) | — | — | — | — | — | — |
| EEG Adapter + Text MLP only | — | — | — | — | 0.06 | — |
| + ViT Seq2Seq (no DANA) | — | — | — | — | — | — |
| **Full Pipeline (+ DANA)** | — | — | — | — | — | — |
| Oracle (GT latents) | — | — | — | — | — | — |

*Current latent-space evaluation (metrics_latent.json): mean cosine_sim ≈ 0.04–0.12, mean MSE ≈ 0.19–0.24*

### Qualitative Results

> 📌 **Placeholder** — replace with actual GIF previews after video generation

| Ground Truth | Generated Output |
|---|---|
| ![GT Sample 1](assets/gt_001.gif) | ![Gen Sample 1](assets/gen_001.gif) |
| ![GT Sample 2](assets/gt_002.gif) | ![Gen Sample 2](assets/gen_002.gif) |
| ![GT Sample 3](assets/gt_003.gif) | ![Gen Sample 3](assets/gen_003.gif) |

*Caption: Ground truth video clips (left) vs. EEG-conditioned VideoLDM reconstructions (right). Each clip is 6 frames at 3 FPS (2 seconds). EEG recorded from 62 channels during passive video viewing.*

### Per-Frame Comparison

> 📌 **Placeholder** — replace with actual frame comparisons

| Ground Truth Frame | Generated Frame | SSIM | CLIP Sim |
|---|---|---|---|
| `![](assets/gt_frame_01.png)` | `![](assets/gen_frame_01.png)` | — | — |
| `![](assets/gt_frame_02.png)` | `![](assets/gen_frame_02.png)` | — | — |

---

## 🔬 Experiments & Ablations

### Ablation Study Design

| Experiment | Description | Key Finding |
|---|---|---|
| **A1** — No DANA | Initialize UNet from pure Gaussian noise | Expected: lower structural coherence |
| **A2** — No ViT Seq2Seq | Use random latent init instead of ViT predictions | Expected: poorer reconstruction fidelity |
| **A3** — No CLIP alignment | Remove Text MLP, use zero context | Expected: semantically incoherent outputs |
| **A4** — Flat path only | Remove band-pooled path from EEG Adapter | Expected: marginal CLIP sim drop |
| **A5** — Band path only | Remove flat spectral path | Expected: significant val_loss increase |
| **A6** — β_fast = β_slow = 0.5 | Fixed DANA beta (no dynamics awareness) | Expected: lower diversity across fast/slow clips |
| **Full Pipeline** | All components enabled | Target: best FVD + CLIP sim |

### W&B Ablation Tracking

```bash
# Run ablation A1 (no DANA)
python pipelines/run_pipeline_test.py \
    --ablation no_dana \
    --wandb_run_name "ablation_no_dana"

# Run ablation A3 (no CLIP)
python pipelines/run_pipeline_test.py \
    --ablation no_clip \
    --wandb_run_name "ablation_no_clip"
```

---

## ⚠ Challenges & Mitigations

| Challenge | Severity | Root Cause | Fix Applied |
|---|---|---|---|
| **DataLoader RAM crash** | 🔴 High | `num_workers > 0` causes memory explosion with per-file `.pt` loading on 291k samples | Reduced to `num_workers=0` or `num_workers=2`; use `TRAIN_SUBSET=20000` |
| **GPU memory overflow (>16 GB)** | 🔴 High | Full 6-frame batch through UNet2DConditionModel | fp16 mixed precision + `GRAD_ACCUM=4` + `BATCH_SIZE=2` |
| **Wrong frame grouping in VAE latents** | 🔴 High | Reshaping `(300k, 4, 32, 32)` → `(50k, 6, 4, 32, 32)` groups unrelated frames | Switched to per-sample `video_sample_XXXXXX.pt` loading (correct grouping) |
| **CLIP alignment instability** | 🔴 High | EEG Adapter initially collapsed to near-zero embeddings | Raw CLIP embedding injection as shortcut; train Adapter + MLP jointly |
| **Mode collapse in generation** | 🟡 Medium | Diffusion model ignores EEG conditioning, generates average frames | Classifier-free guidance (random conditioning dropout p=0.1); guidance scale=7.5 |
| **VAE scale mismatch** | 🟡 Medium | Synthetic latents had std≈0.18 vs real VAE std≈0.9–5.0 | Correct normalization: `latent / 0.18215` per SD convention |
| **Wrong tensor alignment** | 🟡 Medium | Sub-team 3 and Sub-team 4 outputs in different sample orders | Strict alphabetical sort enforced in `AlignedInferenceDataset` |
| **Subject variability** | 🟡 Medium | EEG patterns differ significantly across participants | z-score normalization per channel per epoch; future: subject ID conditioning |

---

## 👥 Team Contributions

| Member(s) | Sub-team | Responsibilities |
|---|---|---|
| [Name TBD] | **Sub-team 1 (EEG Preprocessing)** | MNE-Python bandpass filtering, STFT computation, epoch windowing, spectrogram validation, dataset class (`data/dataset.py`) |
| [Name TBD] | **Sub-team 2 (VideoLDM)** | SD v1.5 UNet fine-tuning, EEGProjection, DANA integration, DDPM/DDIM training loop, mixed precision, EMA, checkpoint management (`training/train_videoldm.py`, `models/diffusion_backbone/`) |
| [Name TBD] | **Sub-team 3 (ViT Seq2Seq)** | Vision Transformer encoder, Seq2Seq LSTM latent predictor, visual latent generation (`models/eeg_encoder/vision_transformer.py`, `scripts/generate_visual_latents.py`) |
| [Name TBD] | **Sub-team 4 (Dynamics + Text MLP)** | Dual-path EEG Adapter, TextProjectorMLP, DynamicsClassifier, frequency-band pooling, W&B sweeps (`models/dynamics_predictor/`, `training/train_text_mlp.py`, `training/train_dynamics_mlp.py`) |
| [Name TBD] | **Evaluation** | SSIM/PSNR/FID/FVD/CLIP-sim scripts, latent-space evaluation, oracle test, ablation experiments (`evaluation/evaluate.py`, `utils/oracle_test.py`) |
| [Name TBD] | **Project Lead / Integration** | Module interface contracts, tensor alignment (`AlignedInferenceDataset`), pipeline integration test, final report coordination (`pipelines/run_pipeline_test.py`) |
| [Name TBD] | **Technical Writing** | Documentation, README, mid-project report, final report (10–15 pages), presentation slides |

---

## 🔮 Future Work

- **Temporal Attention Modules:** Insert 3D attention layers between spatial UNet blocks to enforce inter-frame consistency (full VideoLDM extension)
- **Subject-Adaptive Conditioning:** Add learnable subject ID embedding as auxiliary input to handle inter-subject EEG variability
- **LoRA Fine-tuning:** Use Low-Rank Adaptation for memory-efficient full UNet fine-tuning on limited GPU VRAM
- **Higher Resolution Output:** Scale VAE decoder from 128×128 to 256×256 using SD v2.1 or SDXL backbone
- **Optical Flow Post-Processing:** Apply temporal smoothing between generated frames to reduce flickering artifacts
- **Contrastive EEG Pre-training:** Pre-train EEG Adapter with CLIP contrastive loss on large-scale EEG datasets before task-specific fine-tuning
- **Real-Time Inference:** Optimize DANA + DDIM pipeline for < 1 second per clip (distillation / consistency models)

---

## 🛠 Troubleshooting

**Q: `RuntimeError: CUDA out of memory` during VideoLDM training**

```bash
# Reduce batch size and enable gradient checkpointing
python training/train_videoldm.py --batch_size 1 --grad_accum 8 \
    --gradient_checkpointing --mixed_precision fp16
```

**Q: DataLoader worker process killed (`Killed` message)**

```python
# In train_videoldm.py, set:
DataLoader(..., num_workers=0, pin_memory=False)
# Or add to your bash session:
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

**Q: `KeyError: eeg_sample_XXXXXX.pt not found` during dataset loading**

```bash
# Verify dataset integrity
ls /home/teaching/TEAM_22_DATASET/processed/processed/ | wc -l
# Should be ≥ 291,000 files (3 modalities × 97k samples)
python -c "from data.dataset import EEGVideoDataset; ds = EEGVideoDataset('...', 'train'); print(len(ds))"
```

**Q: Generated videos are all identical (mode collapse)**

```python
# Increase classifier-free guidance scale during inference
guidance_scale = 10.0  # default 7.5; try 10–15 for more diversity
# Also check DANA beta values — if is_fast is always 0 or 1:
print(is_fast.mean())  # should be between 0.2 and 0.8 for a healthy split
```

**Q: `cosine_sim` in metrics_latent.json is near 0 or negative**

This is expected at early training stages. The Text MLP requires ~10–15 epochs to begin producing meaningful CLIP-aligned embeddings. Monitor the `val_cosine_sim` curve in W&B — it should trend positive by epoch 5 with `lr=1e-3`.

**Q: Tensor shape mismatch between Sub-team 3 and Sub-team 4 outputs**

```bash
# Verify alignment
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
- **RunwayML** — Stable Diffusion v1.5 pre-trained weights (`runwayml/stable-diffusion-v1-5`)
- **OpenAI CLIP** — Pre-trained visual-language embedding model
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

[8] Palazzo, S., et al. (2020). Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features. IEEE TPAMI (EEG-ImageNet / EEGCVPR40 dataset).

[9] CVPR 2026 Submission #22. Structured Multivariate Time-Series Modeling for Diffusion-Based EEG-to-Image Reconstruction. (CARD Transformer baseline reference.)
```

---

