"""
train_mtl.py
─────────────
Joint training of the Multi-Task Learning model for simultaneous
semantic segmentation and monocular depth estimation.

Warm-start strategy:
    Encoder     ← loaded from seg baseline checkpoint
    Seg decoder ← loaded from seg baseline checkpoint
    Depth decoder← loaded from depth baseline checkpoint

Loss:
    Uncertainty-weighted combination of CE+Dice (seg) and BerHu (depth).
    Two learnable log-sigma parameters automatically balance the tasks.

Usage:
    python scripts/train_mtl.py
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import cv2

from datasets.cityscapes_dataset import CityscapesDataset
from models.segformer_encoder import SegFormerEncoder
from models.segformer_decoder import SegFormerDecoder
from models.decoder.progressive_depth_decoder import ProgressiveDepthDecoder
from models.mtl_model import MTLModel
from scripts.mtl_loss import MTLLoss

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 19
IMG_SIZE    = (1024, 2048)       # same resolution as baselines

# Batch 2 + accum 4 = effective 8.
# Two decoders use more VRAM than single-task. Batch 2 is safe on 32GB.
# If you see OOM, reduce BATCH_SIZE to 1 and increase ACCUM_STEPS to 8.
BATCH_SIZE  = 2
ACCUM_STEPS = 4
EFF_BATCH   = BATCH_SIZE * ACCUM_STEPS   # = 8

EPOCHS      = 80
LR_ENCODER  = 6e-6      # pretrained — slow
LR_DECODER  = 6e-5      # task heads — faster
LR_SIGMA    = 1e-3      # uncertainty params — slightly faster than decoder
POLY_POWER  = 0.9
MIN_LR      = 1e-7

# ── Baseline checkpoints for warm-start ──
# Both .pth files save model.state_dict() with keys: encoder.*, decoder.*
SEG_CKPT   = "checkpoints/segmentation/best_seg_model.pth"
DEPTH_CKPT = "checkpoints/depth/best_depth_model.pth"

SAVE_BEST   = "checkpoints/mtl/best_mtl_model.pth"
SAVE_LAST   = "checkpoints/mtl/last_mtl_checkpoint.pth"
LOG_DIR     = "logs/mtl"
LOG_CSV     = os.path.join(LOG_DIR, "train_log.csv")
PRED_DIR    = os.path.join(LOG_DIR, "predictions")

os.makedirs("checkpoints/mtl", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ─────────────────────────────────────────
# Per-epoch visualisation: RGB | Seg | Depth
# ─────────────────────────────────────────
def save_mtl_prediction(img_t, seg_t, depth_t, epoch):
    import matplotlib.pyplot as plt
    img = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    seg = seg_t.squeeze(0).cpu().numpy()
    seg = np.clip(seg, 0, NUM_CLASSES - 1)
    seg_col = CITYSCAPES_COLORS[seg]

    depth = depth_t.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].imshow(img);       axes[0].set_title("RGB Input",     fontsize=11); axes[0].axis("off")
    axes[1].imshow(seg_col);   axes[1].set_title("Seg Prediction",fontsize=11, color="navy"); axes[1].axis("off")
    axes[2].imshow(depth, cmap="inferno"); axes[2].set_title("Depth Prediction", fontsize=11, color="darkred"); axes[2].axis("off")
    plt.suptitle(f"MTL Model — Epoch {epoch}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PRED_DIR, f"mtl_epoch_{epoch:02d}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved prediction: {out}")


# ─────────────────────────────────────────
# Poly LR scheduler
# ─────────────────────────────────────────
class PolyLRScheduler:
    def __init__(self, optimizer, total_steps, power=0.9, min_lr=1e-7):
        self.optimizer    = optimizer
        self.total_steps  = total_steps
        self.power        = power
        self.min_lr       = min_lr
        self.current_step = 0
        self.base_lrs     = [g['lr'] for g in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        f = (1 - self.current_step / self.total_steps) ** self.power
        for base_lr, g in zip(self.base_lrs, self.optimizer.param_groups):
            g['lr'] = max(base_lr * f, self.min_lr)

    def state_dict(self):
        return {'current_step': self.current_step, 'base_lrs': self.base_lrs}

    def load_state_dict(self, d):
        self.current_step = d['current_step']
        self.base_lrs     = d['base_lrs']


# ─────────────────────────────────────────
# mIoU via confusion matrix (accurate)
# ─────────────────────────────────────────
class SegMetric:
    def __init__(self, num_classes=19, ignore_index=255):
        self.C            = num_classes
        self.ignore_index = ignore_index
        self.confusion    = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred   = pred.view(-1).cpu()
        target = target.view(-1).cpu()
        valid  = target != self.ignore_index
        pred   = pred[valid];  target = target[valid]
        idx    = target * self.C + pred
        cnt    = torch.bincount(idx, minlength=self.C * self.C)
        self.confusion += cnt.view(self.C, self.C)

    def compute(self):
        C  = self.confusion.float()
        tp = C.diag()
        union = C.sum(0) + C.sum(1) - tp
        iou   = torch.where(union > 0, tp / union, torch.zeros_like(tp))
        return iou[union > 0].mean().item()

    def reset(self):
        self.confusion.zero_()


# ─────────────────────────────────────────
# Depth metrics
# ─────────────────────────────────────────
def compute_depth_metrics(pred, target):
    pred   = pred.squeeze(1)
    mask   = target > 0
    p, t   = pred[mask], target[mask]
    abs_rel = torch.mean(torch.abs(p - t) / (t + 1e-8)).item()
    rmse    = torch.sqrt(torch.mean((p - t) ** 2)).item()
    ratio   = torch.max(p / (t + 1e-8), t / (p + 1e-8))
    delta   = (ratio < 1.25).float().mean().item()
    return abs_rel, rmse, delta


# ─────────────────────────────────────────
# Warm-start loader
# ─────────────────────────────────────────
def warm_start(model, seg_ckpt_path, depth_ckpt_path):
    """
    Load encoder + seg_decoder from seg baseline checkpoint.
    Load depth_decoder from depth baseline checkpoint.

    Baseline checkpoints have state_dict keys:
        encoder.*   → model.encoder.*
        decoder.*   → model.seg_decoder.* OR model.depth_decoder.*
    """
    print(f"[INIT] Warm-starting from baselines...")

    # ── Seg checkpoint → encoder + seg_decoder ────
    seg_state = torch.load(seg_ckpt_path, map_location="cpu")
    enc_w, seg_dec_w = {}, {}
    for k, v in seg_state.items():
        if k.startswith("encoder."):
            enc_w[k[len("encoder."):]] = v
        elif k.startswith("decoder."):
            seg_dec_w[k[len("decoder."):]] = v

    enc_result     = model.encoder.load_state_dict(enc_w, strict=True)
    seg_dec_result = model.seg_decoder.load_state_dict(seg_dec_w, strict=True)
    print(f"[INIT] Encoder loaded from seg checkpoint")
    print(f"[INIT] Seg decoder loaded from seg checkpoint")

    # ── Depth checkpoint → depth_decoder only ─────
    depth_state = torch.load(depth_ckpt_path, map_location="cpu")
    dep_dec_w = {}
    for k, v in depth_state.items():
        if k.startswith("decoder."):
            dep_dec_w[k[len("decoder."):]] = v

    dep_dec_result = model.depth_decoder.load_state_dict(dep_dec_w, strict=True)
    print(f"[INIT] Depth decoder loaded from depth checkpoint")
    print(f"[INIT] Warm-start complete — all weights loaded successfully\n")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print(f"[INFO] Device      : {DEVICE}")
    print(f"[INFO] Resolution  : {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print(f"[INFO] Batch size  : {BATCH_SIZE} × accum {ACCUM_STEPS} = effective {EFF_BATCH}")
    print(f"[INFO] Epochs      : {EPOCHS}")
    print(f"[INFO] Loss        : Uncertainty-weighted CE+Dice + BerHu\n")

    # ── Datasets ──────────────────────────
    train_ds = CityscapesDataset("data/cityscapes", "train", img_size=IMG_SIZE)
    val_ds   = CityscapesDataset("data/cityscapes", "val",   img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────
    encoder       = SegFormerEncoder()
    seg_decoder   = SegFormerDecoder(num_classes=NUM_CLASSES, embed_dim=256, dropout=0.1)
    depth_decoder = ProgressiveDepthDecoder(embed_dim=256)
    model         = MTLModel(encoder, seg_decoder, depth_decoder, NUM_CLASSES).to(DEVICE)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total/1e6:.2f}M")
    seg_p   = sum(p.numel() for p in model.seg_decoder.parameters())
    dep_p   = sum(p.numel() for p in model.depth_decoder.parameters())
    enc_p   = sum(p.numel() for p in model.encoder.parameters())
    print(f"[INFO] Encoder: {enc_p/1e6:.2f}M  |  Seg decoder: {seg_p/1e6:.2f}M  |  Depth decoder: {dep_p/1e6:.2f}M")

    # ── Loss ──────────────────────────────
    criterion = MTLLoss().to(DEVICE)

    # ── Optimizer ─────────────────────────
    # Four parameter groups:
    # 1. Encoder            — pretrained, slow LR
    # 2. Seg decoder        — trained, medium LR
    # 3. Depth decoder      — trained, medium LR
    # 4. Uncertainty params — faster LR (they need to adapt quickly)
    optimizer = AdamW([
        {'params': model.encoder.parameters(),
         'lr': LR_ENCODER, 'name': 'encoder'},
        {'params': model.seg_decoder.parameters(),
         'lr': LR_DECODER, 'name': 'seg_decoder'},
        {'params': model.depth_decoder.parameters(),
         'lr': LR_DECODER, 'name': 'depth_decoder'},
        {'params': [criterion.log_sigma_seg, criterion.log_sigma_depth],
         'lr': LR_SIGMA,  'name': 'uncertainty'},
    ], weight_decay=1e-4)

    scaler    = GradScaler()
    scheduler = PolyLRScheduler(optimizer, total_steps=EPOCHS,
                                power=POLY_POWER, min_lr=MIN_LR)

    # ── Resume or warm-start ──────────────
    start_epoch  = 0
    best_score   = -999.0   # combined score = mIoU - AbsRel (higher is better)

    if os.path.exists(SAVE_LAST):
        # Resume interrupted MTL training
        print(f"[INFO] Resuming MTL training from: {SAVE_LAST}")
        ckpt = torch.load(SAVE_LAST, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        criterion.load_state_dict(ckpt["loss_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_score  = ckpt.get("best_score", -999.0)
        print(f"[INFO] Resumed from epoch {start_epoch}, best score: {best_score:.4f}\n")
    else:
        # Fresh start — warm-start from baselines
        assert os.path.exists(SEG_CKPT),   f"Seg checkpoint not found: {SEG_CKPT}"
        assert os.path.exists(DEPTH_CKPT), f"Depth checkpoint not found: {DEPTH_CKPT}"
        warm_start(model, SEG_CKPT, DEPTH_CKPT)

    # CSV log header
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w") as f:
            f.write("epoch,total_loss,seg_loss,depth_loss,"
                    "w_seg,w_depth,sig_seg,sig_depth,"
                    "val_miou,val_absrel,val_delta,combined_score\n")

    # ── Training loop ──────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # ─── TRAIN ────────────────────────
        model.train()
        criterion.train()

        sum_total = sum_seg = sum_depth = 0.0
        sum_wseg  = sum_wdep = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{EPOCHS} [train]")

        for step, batch in pbar:
            img       = batch["image"].to(DEVICE)
            seg_gt    = batch["seg"].to(DEVICE)
            depth_gt  = batch["depth"].to(DEVICE)

            with autocast():
                seg_pred, depth_pred = model(img)
                total_loss, l_seg, l_depth, w_seg, w_depth = criterion(
                    seg_pred, seg_gt, depth_pred, depth_gt
                )
                total_loss = total_loss / ACCUM_STEPS

            scaler.scale(total_loss).backward()

            sum_total += total_loss.item() * ACCUM_STEPS
            sum_seg   += l_seg
            sum_depth += l_depth
            sum_wseg  += w_seg
            sum_wdep  += w_depth

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(
                loss=f"{total_loss.item()*ACCUM_STEPS:.3f}",
                seg=f"{l_seg:.3f}",
                dep=f"{l_depth:.4f}",
                ws=f"{w_seg:.2f}",
                wd=f"{w_depth:.2f}"
            )

        scheduler.step()
        n = len(train_loader)
        avg_total  = sum_total / n
        avg_seg    = sum_seg   / n
        avg_depth  = sum_depth / n
        avg_wseg   = sum_wseg  / n
        avg_wdep   = sum_wdep  / n
        sig_seg    = float(torch.exp(criterion.log_sigma_seg).item())
        sig_depth  = float(torch.exp(criterion.log_sigma_depth).item())

        enc_lr = optimizer.param_groups[0]['lr']
        dec_lr = optimizer.param_groups[1]['lr']

        # ─── VALIDATE ─────────────────────
        model.eval()
        criterion.eval()

        seg_metric = SegMetric(num_classes=NUM_CLASSES)
        absrel_sum = rmse_sum = delta_sum = 0.0
        vis_saved  = False

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader,
                                           desc=f"Epoch {epoch+1}/{EPOCHS} [val]")):
                img      = batch["image"].to(DEVICE)
                seg_gt   = batch["seg"].to(DEVICE)
                depth_gt = batch["depth"].to(DEVICE)

                with autocast():
                    seg_pred, depth_pred = model(img)

                pred_class = torch.argmax(seg_pred, dim=1)
                seg_metric.update(pred_class, seg_gt)

                ar, rm, dl = compute_depth_metrics(depth_pred, depth_gt)
                absrel_sum += ar
                rmse_sum   += rm
                delta_sum  += dl

                if not vis_saved:
                    save_mtl_prediction(img, pred_class, depth_pred, epoch + 1)
                    vis_saved = True

        nv         = len(val_loader)
        val_miou   = seg_metric.compute()
        val_absrel = absrel_sum / nv
        val_rmse   = rmse_sum   / nv
        val_delta  = delta_sum  / nv

        # Combined score: higher mIoU is better, lower AbsRel is better
        combined_score = val_miou - val_absrel

        print(f"\n{'='*65}")
        print(f"  Epoch       : {epoch+1}/{EPOCHS}")
        print(f"  Total Loss  : {avg_total:.4f}  (uncertainty-weighted)")
        print(f"  Seg Loss    : {avg_seg:.4f}    Depth Loss: {avg_depth:.5f}")
        print(f"  Task weights: w_seg={avg_wseg:.3f}  w_depth={avg_wdep:.3f}")
        print(f"  Sigma values: σ_seg={sig_seg:.4f}  σ_depth={sig_depth:.4f}")
        print(f"  Enc LR      : {enc_lr:.2e}   Dec LR: {dec_lr:.2e}")
        print(f"  ── Validation ──────────────────────────────────────")
        print(f"  Seg mIoU    : {val_miou:.4f}  ({val_miou*100:.2f}%)  "
              f"[baseline: 80.00%]")
        print(f"  Depth AbsRel: {val_absrel:.4f}             "
              f"[baseline: 0.1003]")
        print(f"  Depth RMSE  : {val_rmse:.4f}")
        print(f"  Depth δ<1.25: {val_delta:.4f}")
        print(f"  Combined    : {combined_score:.4f}  (mIoU - AbsRel)")
        print(f"{'='*65}\n")

        # Save best model (based on combined score)
        if combined_score > best_score:
            best_score = combined_score
            torch.save(model.state_dict(), SAVE_BEST)
            print(f"[INFO] Best MTL model saved → "
                  f"mIoU={val_miou*100:.2f}%  AbsRel={val_absrel:.4f}  "
                  f"score={best_score:.4f}")

        # Save checkpoint every epoch for crash recovery
        torch.save({
            "epoch"               : epoch,
            "model_state_dict"    : model.state_dict(),
            "loss_state_dict"     : criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score"          : best_score,
        }, SAVE_LAST)

        # CSV logging
        with open(LOG_CSV, "a") as f:
            f.write(f"{epoch+1},{avg_total:.6f},{avg_seg:.6f},{avg_depth:.6f},"
                    f"{avg_wseg:.4f},{avg_wdep:.4f},{sig_seg:.4f},{sig_depth:.4f},"
                    f"{val_miou:.6f},{val_absrel:.6f},{val_delta:.6f},"
                    f"{combined_score:.6f}\n")

    print("\n[INFO] MTL training complete.")
    print(f"[INFO] Best combined score: {best_score:.4f}")


if __name__ == "__main__":
    main()