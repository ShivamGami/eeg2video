import os
os.environ["WANDB_API_KEY"] = "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"
import torch
import torch.nn as nn
import wandb
from dataset import EEGVideoDataset
from torch.utils.data import DataLoader
from subteam4_models import DynamicsClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Label smoothing loss ──────────────────────────────────────────────────────
# Standard BCEWithLogitsLoss makes the model overconfident on noisy EEG labels.
# Label smoothing softens targets: 1.0 -> 0.9, 0.0 -> 0.1
# This reduces the train/val gap seen in previous runs by preventing the model
# from pushing logits to extreme values for ambiguous samples.
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing  = smoothing
        self.bce        = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Smooth: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        soft_targets = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return self.bce(logits, soft_targets)


def train_dynamics():
    wandb.init(project="eeg2video-subteam4", name="dynamics-v5-final")

    data_dir = "/home/teaching/TEAM_22_DATASET/processed/processed"
    train_loader = DataLoader(
        EEGVideoDataset(data_dir, mode='train'),
        batch_size=256, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        EEGVideoDataset(data_dir, mode='val'),
        batch_size=256, shuffle=False, num_workers=4
    )

    # ── Label sanity check ───────────────────────────────────────────────────
    sample_labels = []
    for i, (_, _, _, lbl) in enumerate(train_loader):
        sample_labels.append(lbl)
        if i >= 5: break
    all_lbls = torch.cat(sample_labels)
    print(f"Label sanity | mean={all_lbls.mean():.3f} std={all_lbls.std():.3f}")
    if all_lbls.std() < 0.1:
        print("STOP: Labels collapsed.")
        wandb.finish(); return

    model     = DynamicsClassifier().to(DEVICE)
    criterion = LabelSmoothingBCE(smoothing=0.1)

    # weight_decay=1e-4 (not 1e-3 — that was too aggressive)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # CosineAnnealing over 150 epochs: lr decays 1e-4 -> 1e-5 smoothly.
    # Previous runs used ReduceLROnPlateau which halved lr too early (epoch 11)
    # and starved the model of learning signal. Cosine schedule is more stable.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=150, eta_min=1e-5
    )

    best_val_acc   = 0.0
    patience_count = 0
    patience_limit = 30   # generous — val_acc was still trending up at epoch 100

    print("Training Dynamics Classifier (150 epochs, label smoothing, cosine LR)...")

    for epoch in range(1, 151):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0

        for eeg, _, _, label in train_loader:
            feat  = eeg.to(DEVICE)
            label = label.unsqueeze(1).to(DEVICE).float()

            logits = model(feat)
            loss   = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss   += loss.item()
            # Accuracy uses raw (unsmoothed) 0.5 threshold
            preds     = (torch.sigmoid(logits) > 0.5).float()
            t_correct += (preds == label).sum().item()
            t_total   += label.size(0)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0

        with torch.no_grad():
            for veeg, _, _, vlabel in val_loader:
                vfeat  = veeg.to(DEVICE)
                vlabel = vlabel.unsqueeze(1).to(DEVICE).float()
                vlogits = model(vfeat)
                v_loss += criterion(vlogits, vlabel).item()
                vpreds  = (torch.sigmoid(vlogits) > 0.5).float()
                v_correct += (vpreds == vlabel).sum().item()
                v_total   += vlabel.size(0)

        val_acc    = v_correct / v_total
        train_acc  = t_correct / t_total
        current_lr = optimizer.param_groups[0]['lr']
        acc_gap    = train_acc - val_acc   # track overfitting in WandB

        wandb.log({
            "epoch":      epoch,
            "train_loss": t_loss / len(train_loader),
            "train_acc":  train_acc,
            "val_loss":   v_loss  / len(val_loader),
            "val_acc":    val_acc,
            "acc_gap":    acc_gap,    # NEW: overfitting monitor
            "lr":         current_lr,
        })
        print(
            f"Epoch {epoch:>3}/150 | "
            f"Loss: {t_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"Gap: {acc_gap:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc; patience_count = 0
            torch.save(model.state_dict(), "dynamics_model.pth")
            print(f"  Best saved (Val Acc: {best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience_limit} epochs)")
                break

    wandb.finish()
    print(f"\nDynamics done. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_dynamics()