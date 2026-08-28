import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import cv2

from datasets.cityscapes_dataset import CityscapesDataset
from models.segformer_encoder import SegFormerEncoder
from models.segformer_decoder import SegFormerDecoder
from models.seg_model import SegmentationModel

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 19

# Resolution: 1024×1024 — matches pretrained checkpoint resolution.
# This is the single biggest factor for mIoU improvement.
IMG_SIZE    = (1024, 1024)

# Batch size 2 + accumulation steps 4 = effective batch size 8.
# Keeps memory within 24–32 GB VRAM at 1024×1024.
BATCH_SIZE  = 2
ACCUM_STEPS = 4          # gradient accumulation
EFF_BATCH   = BATCH_SIZE * ACCUM_STEPS   # = 8, same as paper

EPOCHS      = 80         # more iterations = better minority class learning
LR_ENCODER  = 6e-6      # pretrained — keep slow
LR_DECODER  = 6e-5      # randomly initialised — faster
POLY_POWER  = 0.9
MIN_LR      = 1e-7

SAVE_BEST   = "checkpoints/segmentation/best_seg_model.pth"
SAVE_LAST   = "checkpoints/segmentation/last_seg_checkpoint.pth"
PRED_DIR    = "logs/segmentation/predictions"

os.makedirs("checkpoints/segmentation", exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Cityscapes colour palette
# ─────────────────────────────────────────
CITYSCAPES_COLORS = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70),   (102, 102, 156),
    (190, 153, 153),(153, 153, 153),(250, 170, 30),  (220, 220, 0),
    (107, 142, 35), (152, 251, 152),(70, 130, 180),  (220, 20, 60),
    (255, 0, 0),    (0, 0, 142),    (0, 0, 70),      (0, 60, 100),
    (0, 80, 100),   (0, 0, 230),    (119, 11, 32)
], dtype=np.uint8)


def save_prediction(mask: torch.Tensor, epoch: int) -> None:
    mask        = mask.cpu().numpy()
    mask        = np.clip(mask, 0, NUM_CLASSES - 1)
    colored     = CITYSCAPES_COLORS[mask]
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    out_path    = os.path.join(PRED_DIR, f"pred_epoch_{epoch:02d}.png")
    os.makedirs(PRED_DIR, exist_ok=True)
    cv2.imwrite(out_path, colored_bgr)
    print(f"[INFO] Saved prediction: {out_path}")


# ─────────────────────────────────────────
# Poly LR scheduler (per-epoch step)
# ─────────────────────────────────────────
class PolyLRScheduler:
    def __init__(self, optimizer, total_steps: int,
                 power: float = 0.9, min_lr: float = 1e-7):
        self.optimizer    = optimizer
        self.total_steps  = total_steps
        self.power        = power
        self.min_lr       = min_lr
        self.current_step = 0
        self.base_lrs     = [g['lr'] for g in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        factor = (1 - self.current_step / self.total_steps) ** self.power
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = max(base_lr * factor, self.min_lr)

    def state_dict(self):
        return {'current_step': self.current_step, 'base_lrs': self.base_lrs}

    def load_state_dict(self, d):
        self.current_step = d['current_step']
        self.base_lrs     = d['base_lrs']


# ─────────────────────────────────────────
# Loss: CE + Dice
# ─────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        prob  = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        t     = targets.clone(); t[~valid] = 0

        one_hot    = F.one_hot(t, C).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        prob       = prob    * valid_mask
        one_hot    = one_hot * valid_mask

        inter = (prob * one_hot).sum(dim=(0, 2, 3))
        union = prob.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, label_smoothing: float = 0.1):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                        label_smoothing=label_smoothing)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        return 0.7 * self.ce(logits, targets) + 0.3 * self.dice(logits, targets)


# ─────────────────────────────────────────
# mIoU
# ─────────────────────────────────────────
def compute_miou(pred, target, num_classes, ignore_index=255):
    pred   = pred.view(-1);   target = target.view(-1)
    valid  = target != ignore_index
    pred   = pred[valid];     target = target[valid]
    ious   = []
    for cls in range(num_classes):
        p = pred == cls;  t = target == cls
        inter = (p & t).sum().item()
        union = p.sum().item() + t.sum().item() - inter
        if union == 0: continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print(f"[INFO] Resolution  : {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print(f"[INFO] Batch size  : {BATCH_SIZE} (accumulation × {ACCUM_STEPS} = effective {EFF_BATCH})")
    print(f"[INFO] Epochs      : {EPOCHS}")
    print(f"[INFO] Enc LR      : {LR_ENCODER}   Dec LR: {LR_DECODER}")
    print(f"[INFO] Device      : {DEVICE}\n")

    # ── Datasets ──────────────────────────
    train_ds = CityscapesDataset("data/cityscapes", "train", img_size=IMG_SIZE)
    val_ds   = CityscapesDataset("data/cityscapes", "val",   img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────
    encoder = SegFormerEncoder()
    decoder = SegFormerDecoder(num_classes=NUM_CLASSES, embed_dim=256, dropout=0.1)
    model   = SegmentationModel(encoder, decoder, NUM_CLASSES).to(DEVICE)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters  : {total/1e6:.2f}M")

    # ── Differential LR ───────────────────
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
        {'params': model.decoder.parameters(), 'lr': LR_DECODER},
    ], weight_decay=1e-4)

    scheduler = PolyLRScheduler(optimizer, total_steps=EPOCHS,
                                power=POLY_POWER, min_lr=MIN_LR)
    criterion = SegmentationLoss(label_smoothing=0.1)

    # ── Resume ────────────────────────────
    start_epoch = 0
    best_miou   = 0.0

    if os.path.exists(SAVE_LAST):
        print(f"[INFO] Resuming from: {SAVE_LAST}")
        ckpt        = torch.load(SAVE_LAST, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"[INFO] Resuming from epoch {start_epoch}, best mIoU: {best_miou:.4f}\n")

    # ── Training loop ──────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # ─── TRAIN ────────────────────────
        model.train()
        total_loss   = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{EPOCHS} [train]")

        for step, batch in pbar:
            img = batch["image"].to(DEVICE)
            gt  = batch["seg"].to(DEVICE)

            out  = model(img)
            # Divide loss by accum steps so gradients are averaged
            loss = criterion(out, gt) / ACCUM_STEPS
            loss.backward()

            total_loss += loss.item() * ACCUM_STEPS   # log unscaled loss

            # Gradient step every ACCUM_STEPS batches
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        enc_lr   = optimizer.param_groups[0]['lr']
        dec_lr   = optimizer.param_groups[1]['lr']

        # ─── VALIDATE ─────────────────────
        model.eval()
        miou_sum = 0.0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader,
                                           desc=f"Epoch {epoch+1}/{EPOCHS} [val]")):
                img = batch["image"].to(DEVICE)
                gt  = batch["seg"].to(DEVICE)

                out  = model(img)
                pred = torch.argmax(out, dim=1)
                miou_sum += compute_miou(pred, gt, NUM_CLASSES)

                if i == 0:
                    save_prediction(pred[0], epoch + 1)

        avg_miou = miou_sum / len(val_loader)

        print(f"\n{'='*60}")
        print(f"  Epoch      : {epoch+1}/{EPOCHS}")
        print(f"  Train Loss : {avg_loss:.4f}  (CE + Dice)")
        print(f"  Val mIoU   : {avg_miou:.4f}")
        print(f"  Enc LR     : {enc_lr:.2e}   Dec LR: {dec_lr:.2e}")
        print(f"{'='*60}\n")

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(model.state_dict(), SAVE_BEST)
            print(f"[INFO] Best model saved → mIoU: {best_miou:.4f}")

        torch.save({
            "epoch"               : epoch,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou"           : best_miou,
        }, SAVE_LAST)

    print("\n[INFO] Segmentation training complete.")
    print(f"[INFO] Best Val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()