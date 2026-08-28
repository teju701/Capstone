"""
evaluate_depth.py
─────────────────
Loads the saved best depth model and evaluates it on the Cityscapes
validation set. Prints a full metrics report and saves it to a text file.

Usage:
    python scripts/evaluate_depth.py

Outputs:
    - Console: full metrics table
    - logs/depth/eval_report.txt : same table saved to disk
    - logs/depth/eval_sample.png : side-by-side RGB | Depth visualisation
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.cityscapes_dataset import CityscapesDataset
from models.segformer_encoder import SegFormerEncoder
from models.decoder.progressive_depth_decoder import ProgressiveDepthDecoder
from models.depth_model import DepthModel

# ─────────────────────────────────────────
# Config — must match training
# ─────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BEST_CKPT   = "checkpoints/depth/best_depth_model.pth"
LOG_DIR     = "logs/depth"
REPORT_PATH = os.path.join(LOG_DIR, "eval_report.txt")
VIS_PATH    = os.path.join(LOG_DIR, "eval_sample.png")

MIN_DEPTH   = 1.0
MAX_DEPTH   = 80.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

os.makedirs(LOG_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    All metrics computed in normalised [0,1] space.
    pred   : (1, 1, H, W)
    target : (1, H, W)
    """
    pred   = pred.squeeze(1)          # (1, H, W)
    mask   = target > 0               # valid pixels only

    p = pred[mask]
    t = target[mask]

    # AbsRel — primary metric
    abs_rel = torch.mean(torch.abs(p - t) / (t + 1e-8)).item()

    # RMSE
    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()

    # MAE
    mae = torch.mean(torch.abs(p - t)).item()

    # SqRel
    sq_rel = torch.mean(((p - t) ** 2) / (t + 1e-8)).item()

    # Delta thresholds — standard depth eval metrics
    ratio = torch.max(p / (t + 1e-8), t / (p + 1e-8))
    d1 = (ratio < 1.25      ).float().mean().item()
    d2 = (ratio < 1.25 ** 2 ).float().mean().item()
    d3 = (ratio < 1.25 ** 3 ).float().mean().item()

    # Valid pixel coverage
    total_pixels = target.numel()
    valid_pixels = mask.sum().item()
    coverage     = valid_pixels / total_pixels

    return {
        "AbsRel"   : abs_rel,
        "RMSE"     : rmse,
        "MAE"      : mae,
        "SqRel"    : sq_rel,
        "δ<1.25"   : d1,
        "δ<1.25²"  : d2,
        "δ<1.25³"  : d3,
        "coverage" : coverage,
    }


# ─────────────────────────────────────────
# Visualisation — saves one sample
# ─────────────────────────────────────────
def save_eval_sample(img_tensor, pred_tensor, target_tensor, path):
    """
    Saves a 3-panel image: RGB | Predicted Depth | Ground Truth Depth
    """
    # De-normalise RGB
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)

    pred   = pred_tensor.squeeze().cpu().numpy()
    target = target_tensor.squeeze().cpu().numpy()

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Panel 1 — RGB input
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.set_title("RGB Input", fontsize=13, fontweight="bold")
    ax1.axis("off")

    # Panel 2 — Predicted depth
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(pred, cmap="inferno", vmin=0, vmax=1)
    ax2.set_title("Predicted Depth", fontsize=13, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                 label="Normalised depth (0=1m, 1=80m)")

    # Panel 3 — Ground truth depth (mask invalid pixels)
    target_masked = np.where(target > 0, target, np.nan)
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(target_masked, cmap="inferno", vmin=0, vmax=1)
    ax3.set_title("Ground Truth Depth", fontsize=13, fontweight="bold")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04,
                 label="Normalised depth (0=1m, 1=80m)")

    plt.suptitle("Depth Estimation Baseline — Evaluation Sample", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved visualisation → {path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print(f"[INFO] Device     : {DEVICE}")
    print(f"[INFO] Checkpoint : {BEST_CKPT}")
    print(f"[INFO] Depth range: [{MIN_DEPTH}, {MAX_DEPTH}] metres\n")

    # ── Dataset ───────────────────────────
    val_ds = CityscapesDataset("data/cityscapes", "val")
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────
    encoder = SegFormerEncoder()
    decoder = ProgressiveDepthDecoder(embed_dim=256)
    model   = DepthModel(encoder, decoder).to(DEVICE)

    # Load best checkpoint
    assert os.path.exists(BEST_CKPT), f"Checkpoint not found: {BEST_CKPT}"
    model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded best model from {BEST_CKPT}\n")

    # ── Evaluation ────────────────────────
    accum = {k: 0.0 for k in
             ["AbsRel", "RMSE", "MAE", "SqRel",
              "δ<1.25", "δ<1.25²", "δ<1.25³", "coverage"]}

    vis_saved = False

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            img    = batch["image"].to(DEVICE)
            target = batch["depth"].to(DEVICE)

            pred = model(img)

            m = compute_metrics(pred, target)
            for k in accum:
                accum[k] += m[k]

            # Save visualisation from the 5th sample (usually a clear scene)
            if i == 4 and not vis_saved:
                save_eval_sample(img, pred, target, VIS_PATH)
                vis_saved = True

    n = len(val_loader)
    avg = {k: v / n for k, v in accum.items()}

    # ── Report ────────────────────────────
    lines = []
    lines.append("=" * 58)
    lines.append("  DEPTH BASELINE — EVALUATION REPORT")
    lines.append("  Cityscapes Val Set  |  500 images  |  512×1024")
    lines.append("=" * 58)
    lines.append(f"  Checkpoint : {BEST_CKPT}")
    lines.append(f"  Depth range: {MIN_DEPTH}–{MAX_DEPTH} metres (normalised)")
    lines.append("-" * 58)
    lines.append(f"  AbsRel     : {avg['AbsRel']:.4f}   ← PRIMARY METRIC (lower better)")
    lines.append(f"  RMSE       : {avg['RMSE']:.4f}   (lower better)")
    lines.append(f"  MAE        : {avg['MAE']:.4f}   (lower better)")
    lines.append(f"  SqRel      : {avg['SqRel']:.4f}   (lower better)")
    lines.append("-" * 58)
    lines.append(f"  δ < 1.25   : {avg['δ<1.25']:.4f}   (higher better, target > 0.85)")
    lines.append(f"  δ < 1.25²  : {avg['δ<1.25²']:.4f}   (higher better, target > 0.95)")
    lines.append(f"  δ < 1.25³  : {avg['δ<1.25³']:.4f}   (higher better, target > 0.99)")
    lines.append("-" * 58)
    lines.append(f"  Valid pixel coverage: {avg['coverage']*100:.1f}%  (Cityscapes disparity sparsity)")
    lines.append("=" * 58)

    report = "\n".join(lines)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n[INFO] Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()