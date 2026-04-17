import os
os.environ["WANDB_API_KEY"] = "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"
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dataset import EEGVideoDataset
from torch.utils.data import DataLoader
from subteam4_models import TextProjectorMLP, EEGAdapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Epoch config ──────────────────────────────────────────────────────────────
# Phase 1 : 50 epochs  — denoising pre-train
# Phase 2 : 125 epochs — real EEG fine-tune
#
# ROOT CAUSE OF EARLY STOPPING BUG (found in v8):
# val_loss = MSE + cos_weight * (1 - cos_sim) + anchor
# cos_weight increases from 0.05 -> 0.50 each epoch.
# Even if cos_sim is STABLE (good), the loss number rises because
# cos_weight multiplier grows. Early stopping was reading this as
# "model getting worse" and stopping at epoch 78.
# In reality val_acc stayed at 0.930 and CosSim at 0.595 the whole time.
#
# FIX: Use val_cos_sim (higher = better) as the checkpoint/stopping metric.
#      val_loss is still logged for reference but NOT used for decisions.
#      Also log val_loss_mse separately so you can see the pure MSE component.
# ─────────────────────────────────────────────────────────────────────────────
P1_EPOCHS        = 50
P2_START         = P1_EPOCHS + 1       # 51
P2_END           = P1_EPOCHS + 125     # 175
P2_EPOCHS_COUNT  = P2_END - P2_START + 1  # 125
ADAPTER_UNFREEZE = P2_START + 20       # epoch 71
COS_ACC_THRESHOLD = 0.5

def cosine_accuracy(out, target, threshold=COS_ACC_THRESHOLD):
    cos_sim = F.cosine_similarity(out, target, dim=1)
    return (cos_sim > threshold).float().mean().item()


def train():
    wandb.init(project="eeg2video-subteam4", name="text-mlp-v9-final")

    data_dir     = "/home/teaching/TEAM_22_DATASET/processed/processed"
    train_loader = DataLoader(
        EEGVideoDataset(data_dir, mode='train'),
        batch_size=128, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        EEGVideoDataset(data_dir, mode='val'),
        batch_size=128, shuffle=False, num_workers=4
    )

    model   = TextProjectorMLP(input_dim=512, hidden_dim=1024, output_dim=512).to(DEVICE)
    adapter = EEGAdapter().to(DEVICE)
    mse_criterion = nn.MSELoss()

    # ── PHASE 1: Skip if pretrained weights exist ─────────────────────────────
    pretrained_path = "text_mlp_pretrained.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print(f"Loaded Phase 1 weights from {pretrained_path} — skipping Phase 1\n")
    else:
        print(f"PHASE 1: Pre-training with Noise ({P1_EPOCHS} epochs)...")
        optimizer_p1 = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_p1, T_max=P1_EPOCHS, eta_min=1e-5
        )
        best_p1_loss = float('inf')

        for epoch in range(1, P2_START):
            model.train()
            t_loss, t_acc_sum, t_batches = 0, 0, 0
            for _, text, _, _ in train_loader:
                text       = text.to(DEVICE)
                noisy_text = text + torch.randn_like(text) * 0.1
                outputs    = model(noisy_text)
                loss       = mse_criterion(outputs, text)
                optimizer_p1.zero_grad(); loss.backward(); optimizer_p1.step()
                t_loss    += loss.item()
                t_acc_sum += cosine_accuracy(outputs.detach(), text)
                t_batches += 1

            scheduler_p1.step()
            model.eval()
            v_loss, v_acc_sum, v_batches = 0, 0, 0
            with torch.no_grad():
                for _, vt, _, _ in val_loader:
                    vt      = vt.to(DEVICE)
                    v_out   = model(vt + torch.randn_like(vt) * 0.1)
                    v_loss += mse_criterion(v_out, vt).item()
                    v_acc_sum += cosine_accuracy(v_out, vt)
                    v_batches += 1

            avg_t = t_loss / len(train_loader)
            avg_v = v_loss / len(val_loader)
            lr    = optimizer_p1.param_groups[0]['lr']
            wandb.log({"epoch": epoch, "phase": "pretrain",
                       "loss": avg_t, "val_loss": avg_v,
                       "train_acc": t_acc_sum/t_batches,
                       "val_acc":   v_acc_sum/v_batches, "lr": lr})
            print(f"Pre-train {epoch:>2}/{P1_EPOCHS} | Loss: {avg_t:.6f} | "
                  f"Val: {avg_v:.6f} | LR: {lr:.2e}")
            if avg_v < best_p1_loss:
                best_p1_loss = avg_v
                torch.save(model.state_dict(), pretrained_path)
        print("Phase 1 done.\n")

    # ── PHASE 2: FINE-TUNING ─────────────────────────────────────────────────
    for p in adapter.parameters():
        p.requires_grad = False

    optimizer_mlp = torch.optim.Adam(model.parameters(), lr=3e-5)
    optimizer_all = torch.optim.Adam(
        list(model.parameters()) + list(adapter.parameters()), lr=3e-5
    )
    t_max_p2 = P2_END - ADAPTER_UNFREEZE   # 104
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_all, T_max=t_max_p2, eta_min=3e-6
    )

    # FIX: track best on CosSim (higher=better), not val_loss (inflated by warmup)
    best_val_cos   = 0.0
    patience_count = 0
    patience_limit = 25
    adapter_unfrozen = False

    print(f"PHASE 2: Fine-tuning ({P2_EPOCHS_COUNT} epochs, "
          f"checkpoint on val_cos_sim, unfreeze at epoch {ADAPTER_UNFREEZE})...")

    for epoch in range(P2_START, P2_END + 1):

        if epoch == ADAPTER_UNFREEZE and not adapter_unfrozen:
            for p in adapter.parameters():
                p.requires_grad = True
            adapter_unfrozen = True
            print(f"\n  >> Epoch {epoch}: Adapter UNFROZEN\n")

        optimizer  = optimizer_all if adapter_unfrozen else optimizer_mlp
        current_lr = optimizer.param_groups[0]['lr']
        progress   = (epoch - P2_START) / max(P2_EPOCHS_COUNT - 1, 1)
        cos_weight = 0.05 + 0.45 * progress   # 0.05 -> 0.50

        # ── Train ─────────────────────────────────────────────────────────
        model.train(); adapter.train()
        t_loss, t_mse, t_acc_sum, t_cos_sum, t_batches = 0, 0, 0, 0, 0

        for eeg, text, _, _ in train_loader:
            eeg_feat = eeg.to(DEVICE)
            text     = text.to(DEVICE)

            adapted  = adapter(eeg_feat)
            out      = model(adapted)
            out_n    = F.normalize(out,  p=2, dim=1)
            text_n   = F.normalize(text, p=2, dim=1)

            loss_mse = mse_criterion(out, text)
            loss_cos = 1 - (out_n * text_n).sum(dim=1).mean()

            # Anchor loss only after unfreeze (v8 fix — kept)
            if adapter_unfrozen:
                loss_anchor = ((adapted.norm(p=2, dim=1) - 1.0) ** 2).mean()
                loss = loss_mse + cos_weight * loss_cos + 0.1 * loss_anchor
            else:
                loss = loss_mse + cos_weight * loss_cos

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()), max_norm=1.0
            )
            optimizer.step()

            t_loss    += loss.item()
            t_mse     += loss_mse.item()
            t_acc_sum += cosine_accuracy(out.detach(), text)
            t_cos_sum += (out_n.detach() * text_n).sum(dim=1).mean().item()
            t_batches += 1

        if adapter_unfrozen:
            scheduler_p2.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval(); adapter.eval()
        v_loss, v_mse, v_acc_sum, v_cos_sum, v_batches = 0, 0, 0, 0, 0

        with torch.no_grad():
            for ve, vt, _, _ in val_loader:
                ve, vt    = ve.to(DEVICE), vt.to(DEVICE)
                v_adapted = adapter(ve)
                v_out     = model(v_adapted)
                v_out_n   = F.normalize(v_out, p=2, dim=1)
                vt_n      = F.normalize(vt,    p=2, dim=1)

                v_loss_cos = 1 - (v_out_n * vt_n).sum(dim=1).mean()
                v_loss_mse = mse_criterion(v_out, vt)

                if adapter_unfrozen:
                    v_anchor = ((v_adapted.norm(p=2, dim=1) - 1.0) ** 2).mean()
                    v_loss  += (v_loss_mse + cos_weight * v_loss_cos + 0.1 * v_anchor).item()
                else:
                    v_loss  += (v_loss_mse + cos_weight * v_loss_cos).item()

                v_mse     += v_loss_mse.item()
                v_acc_sum += cosine_accuracy(v_out, vt)
                v_cos_sum += (v_out_n * vt_n).sum(dim=1).mean().item()
                v_batches += 1

        avg_v_loss = v_loss    / len(val_loader)
        avg_v_mse  = v_mse     / len(val_loader)   # pure MSE — not inflated by warmup
        avg_t_loss = t_loss    / len(train_loader)
        avg_t_mse  = t_mse     / len(train_loader)
        train_acc  = t_acc_sum / t_batches
        val_acc    = v_acc_sum / v_batches
        train_cos  = t_cos_sum / t_batches
        val_cos    = v_cos_sum / v_batches         # THIS is the real alignment metric

        wandb.log({
            "epoch":         epoch,
            "phase":         "finetune",
            "loss":          avg_t_loss,
            "val_loss":      avg_v_loss,
            "train_mse":     avg_t_mse,    # pure MSE train
            "val_mse":       avg_v_mse,    # pure MSE val — not inflated by warmup
            "train_acc":     train_acc,
            "val_acc":       val_acc,
            "train_cos_sim": train_cos,
            "val_cos_sim":   val_cos,      # PRIMARY metric — used for checkpoint
            "cos_weight":    cos_weight,
            "lr":            current_lr,
            "adapter_frozen": 0 if adapter_unfrozen else 1,
        })
        print(
            f"Epoch {epoch:>3}/{P2_END} | "
            f"MSE: {avg_v_mse:.5f} | Val Loss: {avg_v_loss:.5f} | "
            f"Val Acc: {val_acc:.3f} | CosSim: {val_cos:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # FIX: checkpoint and early stop on val_cos_sim (higher = better)
        # val_loss is NOT used for stopping — it rises artificially due to cos_weight warmup
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            patience_count = 0
            torch.save(model.state_dict(),   "text_mlp_final.pth")
            torch.save(adapter.state_dict(), "eeg_adapter.pth")
            print(f"  Best saved | CosSim: {best_val_cos:.4f} | "
                  f"Val Acc: {val_acc:.3f} | Val MSE: {avg_v_mse:.5f}")
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(CosSim no improvement for {patience_limit} epochs)")
                break

    wandb.finish()
    print(f"\nText MLP done. Best val_cos_sim: {best_val_cos:.4f}")

if __name__ == "__main__":
    train()