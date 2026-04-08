"""
Training Loop — EEG Transformer + CLIP Alignment
=================================================
Trains EEGNetTransformer + CLIPAlignmentMLP together.
Target: cosine similarity ≥ 0.55 on validation set.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataset             import get_dataloaders
from models.eegnet_transformer import EEGNetTransformer
from models.clip_alignment     import CLIPAlignmentMLP, InfoNCELoss
from configs.config            import PATHS, DATA, MODEL, TRAIN, TARGETS


def train_one_epoch(eeg_model, align_model, loader,
                    optimizer, loss_fn, device):
    """Run one training epoch."""

    eeg_model.train()
    align_model.train()

    total_loss = 0.0
    total_sim  = 0.0
    n_batches  = 0

    for batch in loader:
        eeg   = batch["eeg"].to(device)    # (B, 62, 51, 9)
        text  = batch["text"].to(device)   # (B, 512)

        # Forward pass
        eeg_embed    = eeg_model(eeg)        # (B, 512)
        aligned_embed = align_model(eeg_embed) # (B, 512)

        # Loss
        loss, sim = loss_fn(aligned_embed, text)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(eeg_model.parameters()) +
            list(align_model.parameters()),
            max_norm = 1.0
        )

        optimizer.step()

        total_loss += loss.item()
        total_sim  += sim
        n_batches  += 1

    return total_loss / n_batches, total_sim / n_batches


@torch.no_grad()
def validate(eeg_model, align_model, loader, loss_fn, device):
    """Run validation."""

    eeg_model.eval()
    align_model.eval()

    total_loss = 0.0
    total_sim  = 0.0
    n_batches  = 0

    for batch in loader:
        eeg  = batch["eeg"].to(device)
        text = batch["text"].to(device)

        eeg_embed     = eeg_model(eeg)
        aligned_embed = align_model(eeg_embed)
        loss, sim     = loss_fn(aligned_embed, text)

        total_loss += loss.item()
        total_sim  += sim
        n_batches  += 1

    return total_loss / n_batches, total_sim / n_batches


def save_checkpoint(eeg_model, align_model, optimizer,
                    epoch, val_sim, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"           : epoch,
        "val_cosine_sim"  : val_sim,
        "eeg_model"       : eeg_model.state_dict(),
        "align_model"     : align_model.state_dict(),
        "optimizer"       : optimizer.state_dict(),
    }, path)
    print(f"      💾 Checkpoint saved: {path}")


def main():
    print("="*60)
    print("🚀 TRAINING: EEGNet Transformer + CLIP Alignment")
    print("="*60)

    device = TRAIN["device"]
    print(f"Device: {device}")

    # ── DataLoaders ────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = PATHS["data_dir"],
        batch_size  = TRAIN["batch_size"],
        num_workers = TRAIN["num_workers"],
    )

    # ── Models ─────────────────────────────────────────────────
    eeg_model = EEGNetTransformer(
        in_channels = DATA["eeg_channels"],
        freq_bins   = DATA["eeg_freq_bins"],
        time_bins   = DATA["eeg_time_bins"],
        embed_dim   = MODEL["embed_dim"],
        num_heads   = MODEL["num_heads"],
        num_layers  = MODEL["num_layers"],
        clip_dim    = MODEL["clip_dim"],
        dropout     = MODEL["dropout"],
    ).to(device)

    align_model = CLIPAlignmentMLP(
        input_dim  = MODEL["clip_dim"],
        hidden_dim = MODEL["mlp_hidden_dim"],
        output_dim = MODEL["clip_dim"],
        dropout    = MODEL["dropout"],
    ).to(device)

    print(f"\nEEG Model params   : {eeg_model.count_parameters():,}")
    print(f"Align Model params : {sum(p.numel() for p in align_model.parameters()):,}")

    # ── Optimizer ──────────────────────────────────────────────
    optimizer = optim.AdamW(
        list(eeg_model.parameters()) +
        list(align_model.parameters()),
        lr           = TRAIN["learning_rate"],
        weight_decay = TRAIN["weight_decay"],
    )

    # ── Scheduler ──────────────────────────────────────────────
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = TRAIN["num_epochs"],
        eta_min = TRAIN["learning_rate"] * 0.01,
    )

    # ── Loss ───────────────────────────────────────────────────
    loss_fn = InfoNCELoss(temperature=0.07)

    # ── Training Loop ──────────────────────────────────────────
    best_val_sim   = 0.0
    no_improve     = 0
    best_ckpt_path = os.path.join(
        PATHS["checkpoint_dir"], "best_model.pt"
    )

    print(f"\n{'='*60}")
    print(f"Starting training for {TRAIN['num_epochs']} epochs")
    print(f"Target cosine similarity: ≥ {TARGETS['clip_cosine_sim']}")
    print(f"{'='*60}\n")

    for epoch in range(1, TRAIN["num_epochs"] + 1):

        # Train
        train_loss, train_sim = train_one_epoch(
            eeg_model, align_model, train_loader,
            optimizer, loss_fn, device
        )

        # Validate
        val_loss, val_sim = validate(
            eeg_model, align_model, val_loader,
            loss_fn, device
        )

        # Scheduler step
        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]["lr"]
        target_met = "✅" if val_sim >= TARGETS["clip_cosine_sim"] else "⏳"

        print(
            f"Epoch [{epoch:3d}/{TRAIN['num_epochs']}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Sim: {train_sim:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Sim: {val_sim:.4f} {target_met} | "
            f"LR: {lr:.6f}"
        )

        # Save best checkpoint
        if val_sim > best_val_sim:
            best_val_sim = val_sim
            no_improve   = 0
            save_checkpoint(
                eeg_model, align_model, optimizer,
                epoch, val_sim, best_ckpt_path
            )
        else:
            no_improve += 1

        # Save periodic checkpoint
        if epoch % TRAIN["save_every"] == 0:
            ckpt_path = os.path.join(
                PATHS["checkpoint_dir"],
                f"checkpoint_epoch_{epoch:03d}.pt"
            )
            save_checkpoint(
                eeg_model, align_model, optimizer,
                epoch, val_sim, ckpt_path
            )

        # Early stopping
        if no_improve >= TRAIN["early_stop"]:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            print(f"   No improvement for {TRAIN['early_stop']} epochs")
            break

    # ── Final Report ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🎉 TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"   Best Val Cosine Sim : {best_val_sim:.4f}")
    print(f"   Target              : {TARGETS['clip_cosine_sim']}")
    status = "✅ TARGET MET!" if best_val_sim >= TARGETS["clip_cosine_sim"] \
             else "❌ Target not met — tune hyperparameters"
    print(f"   Status              : {status}")
    print(f"   Best checkpoint     : {best_ckpt_path}")


if __name__ == "__main__":
    main()