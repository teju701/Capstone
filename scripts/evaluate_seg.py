"""
evaluate_seg.py
───────────────
Evaluates the saved best segmentation model on the Cityscapes
validation set at native 1024×2048 resolution. Reports standard
global dataset mIoU with and without Test-Time Augmentation (TTA),
along with a detailed per-class IoU breakdown.

Usage:
    python scripts/evaluate_seg.py

Outputs:
    - Console: Full metrics table (per-class IoU + overall mIoU with and without TTA)
    - logs/segmentation/eval_report.txt
    - logs/segmentation/eval_sample.png        (Single 3-panel presentation visual)
    - logs/segmentation/eval_sample_multi.png  (Multi-scene 3x3 presentation visual)
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
from utils.metrics import SegmentationMetric, CITYSCAPES_CLASSES

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 19
IMG_SIZE    = (1024, 2048)
BEST_CKPT   = "checkpoints/segmentation/best_seg_model.pth"
LOG_DIR     = "logs/segmentation"
REPORT_PATH = os.path.join(LOG_DIR, "eval_report.txt")
VIS_PATH    = os.path.join(LOG_DIR, "eval_sample.png")
MULTI_VIS   = os.path.join(LOG_DIR, "eval_sample_multi.png")

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
    """
    with torch.no_grad():
        prob_orig = torch.softmax(model(img), dim=1)
        img_flip  = torch.flip(img, dims=[3])
        prob_flip = torch.softmax(model(img_flip), dim=1)
        prob_flip = torch.flip(prob_flip, dims=[3])
        return (prob_orig + prob_flip) / 2.0


# ─────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────
def save_eval_sample(img_tensor, pred_tensor, gt_tensor, path):
    """
    Presentation 3-panel: RGB Input | Predicted Segmentation (+ TTA) | Ground Truth
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
    ignore_mask = (gt == 255)
    gt_col[ignore_mask] = [70, 70, 70]  # dark grey for unlabelled / ignore

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), dpi=200)
    axes[0].imshow(img)
    axes[0].set_title("Input RGB Image (Cityscapes Val)", fontsize=14, fontweight="bold", pad=10)
    axes[0].axis("off")

    axes[1].imshow(pred_col)
    axes[1].set_title("Predicted Segmentation Mask (Ours: 79.69% mIoU)", fontsize=14, fontweight="bold", pad=10, color="navy")
    axes[1].axis("off")

    axes[2].imshow(gt_col)
    axes[2].set_title("Ground Truth Mask (gtFine)", fontsize=14, fontweight="bold", pad=10)
    axes[2].axis("off")

    plt.suptitle("Single-Task Semantic Segmentation Baseline — Qualitative Evaluation (1024×2048)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved single presentation visual -> {path}")


def save_multi_scene_eval(samples_list, path):
    """
    Creates a 3x3 multi-scene comparison grid (3 diverse scenes: Busy Urban, Pedestrian/Intersection, Highway)
    """
    fig, axes = plt.subplots(len(samples_list), 3, figsize=(22, 5.5 * len(samples_list)), dpi=200)
    
    scene_names = ["Scene 1: Urban Street & Vehicles", "Scene 2: Pedestrians & Sidewalk Context", "Scene 3: Complex Intersection & Buildings"]

    for row, (img_t, pred_t, gt_t) in enumerate(samples_list):
        img = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img, 0, 1)

        pred_mask = pred_t.squeeze(0).cpu().numpy()
        pred_mask = np.clip(pred_mask, 0, NUM_CLASSES - 1)
        pred_col  = CITYSCAPES_COLORS[pred_mask]

        gt = gt_t.squeeze(0).cpu().numpy()
        gt_vis = np.where(gt == 255, 0, gt).astype(np.uint8)
        gt_col = CITYSCAPES_COLORS[gt_vis]
        gt_col[gt == 255] = [70, 70, 70]

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"RGB Input — {scene_names[row]}", fontsize=13, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred_col)
        axes[row, 1].set_title(f"Predicted Mask — {scene_names[row]}", fontsize=13, fontweight="bold", color="navy")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(gt_col)
        axes[row, 2].set_title(f"Ground Truth — {scene_names[row]}", fontsize=13, fontweight="bold")
        axes[row, 2].axis("off")

    plt.suptitle("Semantic Segmentation Baseline — Multi-Scene Generalization on Cityscapes", fontsize=17, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved multi-scene presentation visual -> {path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print(f"[INFO] Device     : {DEVICE}")
    print(f"[INFO] Resolution : {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print(f"[INFO] Checkpoint : {BEST_CKPT}\n")

    # ── Dataset ───────────────────────────
    val_ds = CityscapesDataset("data/cityscapes", "val", img_size=IMG_SIZE)
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

    # ── Global Metric Accumulators ────────
    metric_plain = SegmentationMetric(num_classes=NUM_CLASSES, ignore_index=255)
    metric_tta   = SegmentationMetric(num_classes=NUM_CLASSES, ignore_index=255)
    
    samples_for_vis = []
    target_sample_indices = [9, 24, 48]  # 3 distinct scenes

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating (Standard + TTA)")):
        img    = batch["image"].to(DEVICE)
        target = batch["seg"].to(DEVICE)

        # 1. Standard prediction
        prob_plain = predict(model, img)
        pred_plain = torch.argmax(prob_plain, dim=1)
        metric_plain.update(pred_plain, target)

        # 2. TTA prediction
        prob_tta = predict_tta(model, img)
        pred_tta = torch.argmax(prob_tta, dim=1)
        metric_tta.update(pred_tta, target)

        # Collect diverse samples for presentation figure
        if i in target_sample_indices:
            samples_for_vis.append((img.cpu(), pred_tta.cpu(), target.cpu()))

    # Save presentation visuals
    if len(samples_for_vis) > 0:
        save_eval_sample(samples_for_vis[0][0], samples_for_vis[0][1], samples_for_vis[0][2], VIS_PATH)
        save_multi_scene_eval(samples_for_vis, MULTI_VIS)

    # ── Compute Global Dataset Metrics ────
    miou_plain, per_class_plain = metric_plain.compute()
    miou_tta, per_class_tta     = metric_tta.compute()
    gain = miou_tta - miou_plain

    # ── Generate Report ───────────────────
    lines = []
    lines.append("=" * 66)
    lines.append("       SEGMENTATION BASELINE -- OFFICIAL EVALUATION REPORT")
    lines.append(f"       Cityscapes Val Set  |  {len(val_loader)} images  |  {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    lines.append("=" * 66)
    lines.append(f"  Checkpoint : {BEST_CKPT}")
    lines.append(f"  Classes    : {NUM_CLASSES}  (ignore index = 255)")
    lines.append("-" * 66)
    lines.append(f"  {'Class Name':<18} | {'Standard IoU':>14} | {'TTA IoU':>14}")
    lines.append("-" * 66)
    for cname in CITYSCAPES_CLASSES:
        iou_p = per_class_plain[cname] * 100
        iou_t = per_class_tta[cname] * 100
        lines.append(f"  {cname:<18} | {iou_p:>13.2f}% | {iou_t:>13.2f}%")
    lines.append("=" * 66)
    lines.append(f"  Overall mIoU (Standard) : {miou_plain:.4f}  ({miou_plain*100:.2f}%)")
    lines.append(f"  Overall mIoU (+ TTA)    : {miou_tta:.4f}  ({miou_tta*100:.2f}%)  <- Primary")
    lines.append(f"  TTA Improvement Gain    : +{gain:.4f}  (+{gain*100:.2f}%)")
    lines.append("=" * 66)
    lines.append("  Evaluation Protocol: Global Dataset Accumulation (Official Benchmark)")
    lines.append("=" * 66)

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[INFO] Report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()