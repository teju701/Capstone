import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.cityscapes_dataset import CityscapesDataset
from models.segformer_encoder import SegFormerEncoder
from models.decoder.progressive_depth_decoder import ProgressiveDepthDecoder
from models.depth_model import DepthModel

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 4
LR          = 6e-5
EPOCHS      = 30
SAVE_BEST   = "checkpoints/depth/best_depth_model.pth"
SAVE_LAST   = "checkpoints/depth/last_depth_checkpoint.pth"
PRED_DIR    = "logs/depth/predictions"

# Must match depth_utils.py
MIN_DEPTH   = 1.0    # metres
MAX_DEPTH   = 80.0   # metres

os.makedirs("checkpoints/depth", exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Loss: BerHu (reverse Huber)
# Better than SILog for normalised depth:
#   - L1 for small errors (sharp edges)
#   - L2 for large errors (global structure)
# ─────────────────────────────────────────
class BerHuLoss(nn.Module):
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred:   (B, 1, H, W)
        # target: (B, H, W)  — 0 means invalid pixel
        pred   = pred.squeeze(1)
        mask   = target > 0

        pred_v   = pred[mask]
        target_v = target[mask]

        diff = torch.abs(pred_v - target_v)
        c    = self.threshold * diff.max().detach()   # adaptive threshold

        loss = torch.where(diff <= c, diff, (diff ** 2 + c ** 2) / (2.0 * c + 1e-8))
        return loss.mean()


# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────
def abs_rel(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.squeeze(1)
    mask = target > 0
    p, t = pred[mask], target[mask]
    return torch.mean(torch.abs(p - t) / (t + 1e-8)).item()


def rmse_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.squeeze(1)
    mask = target > 0
    p, t = pred[mask], target[mask]
    return torch.sqrt(torch.mean((p - t) ** 2)).item()


def delta_threshold(pred: torch.Tensor, target: torch.Tensor,
                    thr: float = 1.25) -> float:
    """% of pixels where max(pred/target, target/pred) < thr. Higher is better."""
    pred = pred.squeeze(1)
    mask = target > 0
    p, t = pred[mask] + 1e-8, target[mask] + 1e-8
    ratio = torch.max(p / t, t / p)
    return (ratio < thr).float().mean().item()


# ─────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────
def save_depth_prediction(pred: torch.Tensor, epoch: int) -> None:
    depth = pred.squeeze().cpu().numpy()
    plt.imsave(
        os.path.join(PRED_DIR, f"depth_epoch_{epoch:02d}.png"),
        depth, cmap="inferno"
    )


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    # ── Datasets ──────────────────────────
    train_ds = CityscapesDataset("data/cityscapes", "train")
    val_ds   = CityscapesDataset("data/cityscapes", "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────
    encoder = SegFormerEncoder()
    decoder = ProgressiveDepthDecoder(embed_dim=256)
    model   = DepthModel(encoder, decoder).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {total_params / 1e6:.2f}M")
    print(f"[INFO] Device : {DEVICE}")
    print(f"[INFO] Depth targets: normalised [0,1] = [{MIN_DEPTH}, {MAX_DEPTH}] metres")

    # ── Loss, Optimiser, Scheduler ────────
    criterion = BerHuLoss(threshold=0.2)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Resume checkpoint if available ────
    start_epoch = 0
    best_absrel = float("inf")

    if os.path.exists(SAVE_LAST):
        print(f"[INFO] Resuming from checkpoint: {SAVE_LAST}")
        ckpt = torch.load(SAVE_LAST, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_absrel = ckpt.get("best_absrel", float("inf"))
        print(f"[INFO] Resuming from epoch {start_epoch}, best AbsRel: {best_absrel:.4f}")

    # ── Training loop ──────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            img    = batch["image"].to(DEVICE)   # (B, 3, H, W)
            target = batch["depth"].to(DEVICE)   # (B, H, W)  normalised [0,1]

            optimizer.zero_grad()
            pred = model(img)                    # (B, 1, H, W)  sigmoid → [0,1]

            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # ---------- VALIDATE ----------
        model.eval()
        absrel_sum = 0.0
        rmse_sum   = 0.0
        delta_sum  = 0.0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]")):
                img    = batch["image"].to(DEVICE)
                target = batch["depth"].to(DEVICE)

                pred = model(img)

                absrel_sum += abs_rel(pred, target)
                rmse_sum   += rmse_metric(pred, target)
                delta_sum  += delta_threshold(pred, target, thr=1.25)

                if i == 0:
                    save_depth_prediction(pred[0], epoch + 1)

        n          = len(val_loader)
        avg_absrel = absrel_sum / n
        avg_rmse   = rmse_sum   / n
        avg_delta  = delta_sum  / n

        print(f"\n{'='*58}")
        print(f"  Epoch      : {epoch+1}/{EPOCHS}")
        print(f"  Train Loss : {avg_train_loss:.4f}  (BerHu)")
        print(f"  Val AbsRel : {avg_absrel:.4f}  (lower is better, target < 0.15)")
        print(f"  Val RMSE   : {avg_rmse:.4f}  (lower is better)")
        print(f"  Val δ<1.25 : {avg_delta:.4f}  (higher is better, target > 0.85)")
        print(f"{'='*58}\n")

        # Save best model
        if avg_absrel < best_absrel:
            best_absrel = avg_absrel
            torch.save(model.state_dict(), SAVE_BEST)
            print(f"[INFO] Best model saved → AbsRel: {best_absrel:.4f}")

        # Save checkpoint every epoch
        torch.save({
            "epoch"               : epoch,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_absrel"         : best_absrel,
        }, SAVE_LAST)

    print("\n[INFO] Depth training complete.")
    print(f"[INFO] Best Val AbsRel: {best_absrel:.4f}")


if __name__ == "__main__":
    main()