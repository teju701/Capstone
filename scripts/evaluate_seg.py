"""
evaluate_seg.py
───────────────
Evaluates the saved best segmentation model on the Cityscapes
validation set. Reports mIoU with and without Test-Time Augmentation.

Usage:
    python scripts/evaluate_seg.py

Outputs:
    - Console: mIoU comparison table (no TTA vs TTA)
    - logs/segmentation/eval_report.txt
    - logs/segmentation/eval_sample.png  (3 side-by-side panels)
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.cityscapes_dataset import CityscapesDataset
from models.segformer_encoder import SegFormerEncoder
from models.segformer_decoder import SegFormerDecoder
from models.seg_model import SegmentationModel

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 19
BEST_CKPT   = "checkpoints/segmentation/best_seg_model.pth"
LOG_DIR     = "logs/segmentation"
REPORT_PATH = os.path.join(LOG_DIR, "eval_report.txt")
VIS_PATH    = os.path.join(LOG_DIR, "eval_sample.png")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

CITYSCAPES_COLORS = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70),   (102, 102, 156),
    (190, 153, 153),(153, 153, 153),(250, 170, 30),  (220, 220, 0),
    (107, 142, 35), (152, 251, 152),(70, 130, 180),  (220, 20, 60),
    (255, 0, 0),    (0, 0, 142),    (0, 0, 70),      (0, 60, 100),
    (0, 80, 100),   (0, 0, 230),    (119, 11, 32)
], dtype=np.uint8)

os.makedirs(LOG_DIR, exist_ok=True)


# ─────────────────────────────────────────
# mIoU (ignores label 255)
# ─────────────────────────────────────────
def compute_miou(pred: torch.Tensor, target: torch.Tensor,
                 num_classes: int, ignore_index: int = 255) -> float:
    pred   = pred.view(-1)
    target = target.view(-1)
    valid  = target != ignore_index
    pred, target = pred[valid], target[valid]

    ious = []
    for cls in range(num_classes):
        p = pred   == cls
        t = target == cls
        inter = (p & t).sum().item()
        union = p.sum().item() + t.sum().item() - inter
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────
def predict(model, img: torch.Tensor) -> torch.Tensor:
    """Standard single-pass prediction. Returns softmax (B, C, H, W)."""
    with torch.no_grad():
        return torch.softmax(model(img), dim=1)


def predict_tta(model, img: torch.Tensor) -> torch.Tensor:
    """
    Test-Time Augmentation: horizontal flip.
    Averages softmax from original + flipped image.
    Typically +0.5–1.5 mIoU for free.
    """
    with torch.no_grad():
        # Original
        prob_orig = torch.softmax(model(img), dim=1)

        # Horizontal flip
        img_flip  = torch.flip(img, dims=[3])
        prob_flip = torch.softmax(model(img_flip), dim=1)
        # Un-flip prediction back to original orientation
        prob_flip = torch.flip(prob_flip, dims=[3])

        return (prob_orig + prob_flip) / 2.0


# ─────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────
def save_eval_sample(img_tensor, pred_tensor, gt_tensor, path):
    """
    3-panel: RGB input | Predicted segmentation | Ground truth segmentation
    """
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)

    pred_mask = pred_tensor.squeeze(0).cpu().numpy()
    pred_mask = np.clip(pred_mask, 0, NUM_CLASSES - 1)
    pred_col  = CITYSCAPES_COLORS[pred_mask]

    gt = gt_tensor.squeeze(0).cpu().numpy()
    gt_vis = np.where(gt == 255, 0, gt).astype(np.uint8)
    gt_col = CITYSCAPES_COLORS[gt_vis]
    # grey out ignore regions
    ignore_mask = (gt == 255)
    gt_col[ignore_mask] = [128, 128, 128]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img);        axes[0].set_title("RGB Input",          fontsize=13, fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(pred_col);   axes[1].set_title("Predicted (+ TTA)",  fontsize=13, fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(gt_col);     axes[2].set_title("Ground Truth",       fontsize=13, fontweight="bold"); axes[2].axis("off")

    plt.suptitle("Segmentation Baseline — Evaluation Sample", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved visualisation → {path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print(f"[INFO] Device     : {DEVICE}")
    print(f"[INFO] Checkpoint : {BEST_CKPT}\n")

    # ── Dataset ───────────────────────────
    val_ds = CityscapesDataset("data/cityscapes", "val")
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────
    encoder = SegFormerEncoder()
    decoder = SegFormerDecoder(num_classes=NUM_CLASSES, embed_dim=256, dropout=0.1)
    model   = SegmentationModel(encoder, decoder, NUM_CLASSES).to(DEVICE)

    assert os.path.exists(BEST_CKPT), f"Checkpoint not found: {BEST_CKPT}"
    model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded best model.\n")

    # ── Evaluation — both modes ────────────
    miou_no_tta = 0.0
    miou_tta    = 0.0
    vis_saved   = False

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating (no TTA + TTA)")):
        img    = batch["image"].to(DEVICE)
        target = batch["seg"].to(DEVICE)

        # No TTA
        prob_plain = predict(model, img)
        pred_plain = torch.argmax(prob_plain, dim=1)
        miou_no_tta += compute_miou(pred_plain, target, NUM_CLASSES)

        # TTA
        prob_tta = predict_tta(model, img)
        pred_tta = torch.argmax(prob_tta, dim=1)
        miou_tta += compute_miou(pred_tta, target, NUM_CLASSES)

        # Save sample from image 10 (usually a clear urban scene)
        if i == 9 and not vis_saved:
            save_eval_sample(img, pred_tta, target, VIS_PATH)
            vis_saved = True

    n           = len(val_loader)
    avg_no_tta  = miou_no_tta / n
    avg_tta     = miou_tta    / n
    gain        = avg_tta - avg_no_tta

    # ── Report ────────────────────────────
    lines = []
    lines.append("=" * 58)
    lines.append("  SEGMENTATION BASELINE — EVALUATION REPORT")
    lines.append("  Cityscapes Val Set  |  500 images  |  512×1024")
    lines.append("=" * 58)
    lines.append(f"  Checkpoint : {BEST_CKPT}")
    lines.append(f"  Classes    : 19  (ignore index = 255)")
    lines.append("-" * 58)
    lines.append(f"  mIoU (no TTA) : {avg_no_tta:.4f}")
    lines.append(f"  mIoU (+ TTA)  : {avg_tta:.4f}   ← report this")
    lines.append(f"  TTA gain      : +{gain:.4f}")
    lines.append("=" * 58)
    lines.append("  TTA = horizontal flip averaged with original")
    lines.append("=" * 58)

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n[INFO] Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()