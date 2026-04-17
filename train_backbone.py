# train_backbone.py  — Phase 4 REAL DATA training
"""
Trains TemporalUNet on real SEED-DV processed data.
Loads from: /home/teaching/TEAM_22_DATASET/processed/processed/
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path

from dana        import DANAModule
from decoder     import LatentDecoder
from sd_backbone import TemporalUNet, LATENT_CH, LATENT_H, LATENT_W, N_FRAMES, TEXT_DIM
from dataset     import get_dataloader

LATENT_SHAPE = (N_FRAMES, LATENT_CH, LATENT_H, LATENT_W)   # (6, 4, 16, 16)


def train(epochs, batch_size, lr, device, save_dir):

    print(f"[train] device       = {device}")
    print(f"[train] latent shape = {LATENT_SHAPE}")
    print(f"[train] Loading REAL data from processed dataset...")

    # ── Models ────────────────────────────────────────────────────────────
    model   = TemporalUNet(latent_ch=LATENT_CH, text_dim=TEXT_DIM,
                           n_frames=N_FRAMES).to(device)
    dana    = DANAModule(num_timesteps=1000).to(device)
    decoder = LatentDecoder().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_loader = get_dataloader("train", batch_size=batch_size, shuffle=True)
    val_loader   = get_dataloader("val",   batch_size=1,          shuffle=False)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            # Load real data
            video_latents = batch["video"].to(device)  # (B, 6, 4, 16, 16)
            text_emb      = batch["text"].to(device)   # (B, 77, 512)

            B = video_latents.shape[0]

            # is_fast: derive from optical flow if available,
            # otherwise use uniform random for now
            is_fast = torch.rand(B, device=device)

            # Single timestep for batch
            T = int(torch.randint(0, 1000, (1,)).item())

            # DANA noise — returns (z_T, eps_mixed)
            z_T, noise_target = dana(video_latents, is_fast, T)

            # UNet forward
            t_batch    = torch.full((B,), T, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            pred_noise = model(z_T, text_emb, t_batch)

            loss = nn.functional.mse_loss(pred_noise, noise_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(f"  epoch {epoch:03d} step {step:04d} | loss = {loss.item():.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"[train] epoch {epoch:03d}/{epochs:03d} | avg_loss = {avg_loss:.6f}")

    # ── Save checkpoint ───────────────────────────────────────────────────
    ckpt_path = save_path / "temporalunet_real.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": epochs,
        "latent_shape": LATENT_SHAPE,
    }, ckpt_path)
    print(f"[train] checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save_dir",   type=str,   default="checkpoints")
    parser.add_argument("--device",     type=str,   default=None)
    args   = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train(args.epochs, args.batch_size, args.lr, device, args.save_dir)