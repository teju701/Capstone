"""
evaluate_mtl.py
───────────────
Full evaluation of the best MTL model on the Cityscapes validation set.
Reports all segmentation and depth metrics, compares against baselines,
and generates presentation-quality visualisations.

Usage:
    python scripts/evaluate_mtl.py

Outputs:
    logs/mtl/eval_report.txt         — full metrics comparison table
    logs/mtl/eval_sample.png         — 5-panel single scene comparison
    logs/mtl/eval_multi.png          — 3-scene grid for presentation
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

from datasets.cityscapes_dataset import CityscapesDataset
from models.segformer_encoder import SegFormerEncoder
from models.segformer_decoder import SegFormerDecoder
from models.decoder.progressive_depth_decoder import ProgressiveDepthDecoder
from models.mtl_model import MTLModel

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 19
IMG_SIZE   = (1024, 2048)
BEST_CKPT  = "checkpoints/mtl/best_mtl_model.pth"
LOG_DIR    = "logs/mtl"
os.makedirs(LOG_DIR, exist_ok=True)

# Baseline results for comparison table
SEG_BASELINE_MIOU   = 0.8000   # with TTA
SEG_BASELINE_PER_CLASS = {
    "road": 98.37, "sidewalk": 86.53, "building": 93.22, "wall": 64.23,
    "fence": 61.60, "pole": 66.71, "traffic light": 72.02, "traffic sign": 80.25,
    "vegetation": 93.00, "terrain": 64.75, "sky": 95.29, "person": 83.10,
    "rider": 62.97, "car": 95.25, "truck": 84.47, "bus": 88.35,
    "train": 81.75, "motorcycle": 69.70, "bicycle": 78.35,
}

DEPTH_BASELINE_ABSR  = 0.1003
DEPTH_BASELINE_RMSE  = 0.0678
DEPTH_BASELINE_MAE   = 0.0334
DEPTH_BASELINE_SQREL = 0.0100
DEPTH_BASELINE_D1    = 0.8956
DEPTH_BASELINE_D2    = 0.9779
DEPTH_BASELINE_D3    = 0.9924

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

CITYSCAPES_COLORS = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70),   (102, 102, 156),
    (190, 153, 153),(153, 153, 153),(250, 170, 30),  (220, 220, 0),
    (107, 142, 35), (152, 251, 152),(70, 130, 180),  (220, 20, 60),
    (255, 0, 0),    (0, 0, 142),    (0, 0, 70),      (0, 60, 100),
    (0, 80, 100),   (0, 0, 230),    (119, 11, 32)
], dtype=np.uint8)

CITYSCAPES_CLASSES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]


# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────
class SegMetric:
    def __init__(self, num_classes=19, ignore_index=255):
        self.C = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, pred, target):
        pred   = pred.view(-1).cpu()
        target = target.view(-1).cpu()
        valid  = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        idx  = target * self.C + pred
        cnt  = torch.bincount(idx, minlength=self.C * self.C)
        self.confusion += cnt.view(self.C, self.C)

    def compute(self):
        C  = self.confusion.float()
        tp = C.diag()
        union = C.sum(0) + C.sum(1) - tp
        iou   = torch.where(union > 0, tp / union, torch.zeros_like(tp))
        per_class = {CITYSCAPES_CLASSES[i]: iou[i].item()
                     for i in range(self.C) if union[i] > 0}
        miou = iou[union > 0].mean().item()
        return miou, per_class


def all_depth_metrics(pred, target):
    pred = pred.squeeze(1)
    mask = target > 0
    p, t = pred[mask] + 1e-8, target[mask] + 1e-8
    abs_rel = torch.mean(torch.abs(p - t) / t).item()
    rmse    = torch.sqrt(torch.mean((p - t) ** 2)).item()
    mae     = torch.mean(torch.abs(p - t)).item()
    sq_rel  = torch.mean(((p - t) ** 2) / t).item()
    ratio   = torch.max(p / t, t / p)
    d1 = (ratio < 1.25      ).float().mean().item()
    d2 = (ratio < 1.25 ** 2 ).float().mean().item()
    d3 = (ratio < 1.25 ** 3 ).float().mean().item()
    coverage = mask.float().mean().item()
    return dict(abs_rel=abs_rel, rmse=rmse, mae=mae, sq_rel=sq_rel,
                d1=d1, d2=d2, d3=d3, coverage=coverage)


# ─────────────────────────────────────────
# TTA
# ─────────────────────────────────────────
def predict_tta(model, img):
    with torch.no_grad():
        with autocast():
            seg1, dep1 = model(img)
            prob1  = torch.softmax(seg1, dim=1)

            img_f  = torch.flip(img, dims=[3])
            seg2, dep2 = model(img_f)
            prob2  = torch.flip(torch.softmax(seg2, dim=1), dims=[3])
            dep2_f = torch.flip(dep2, dims=[3])

    return (prob1 + prob2) / 2.0, (dep1 + dep2_f) / 2.0


# ─────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────
def colorise_seg(pred_mask):
    mask = np.clip(pred_mask.cpu().numpy(), 0, NUM_CLASSES - 1)
    return CITYSCAPES_COLORS[mask]


def denorm(img_tensor):
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def save_single_eval(img_t, seg_pred_t, depth_pred_t, seg_gt_t, depth_gt_t, path):
    """5-panel: RGB | Pred Seg | Pred Depth | GT Seg | GT Depth"""
    img       = denorm(img_t)
    seg_pred  = colorise_seg(seg_pred_t.squeeze(0))
    depth_pred= depth_pred_t.squeeze().cpu().numpy()

    gt = seg_gt_t.squeeze(0).cpu().numpy()
    gt_vis = np.where(gt == 255, 0, gt).astype(np.uint8)
    gt_col = CITYSCAPES_COLORS[gt_vis]; gt_col[gt == 255] = [70, 70, 70]

    gt_depth = depth_gt_t.squeeze(0).cpu().numpy()
    gt_depth_vis = np.where(gt_depth > 0, gt_depth, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    titles = ["RGB Input", "MTL — Seg Prediction", "MTL — Depth Prediction",
              "GT Segmentation", "GT Depth (sparse)", ""]
    data   = [img, seg_pred, depth_pred, gt_col, gt_depth_vis, None]
    cmaps  = [None, None, "inferno", None, "inferno", None]

    for ax, title, d, cmap in zip(axes.flat, titles, data, cmaps):
        if d is None:
            ax.axis("off"); continue
        ax.imshow(d, cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    plt.suptitle("MTL Model — Joint Segmentation & Depth Estimation", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def save_multi_scene(samples, path):
    """3-row × 4-col: RGB | Seg Pred | Depth Pred | GT Seg"""
    fig, axes = plt.subplots(len(samples), 4, figsize=(24, 6 * len(samples)))
    headers = ["RGB Input", "Predicted Segmentation", "Predicted Depth", "Ground Truth Seg"]
    scenes  = [f"Scene {i+1}" for i in range(len(samples))]

    for row, (img_t, seg_t, dep_t, gt_t) in enumerate(samples):
        img      = denorm(img_t)
        seg_col  = colorise_seg(seg_t.squeeze(0))
        depth    = dep_t.squeeze().cpu().numpy()
        gt       = gt_t.squeeze(0).cpu().numpy()
        gt_vis   = np.where(gt == 255, 0, gt).astype(np.uint8)
        gt_col   = CITYSCAPES_COLORS[gt_vis]; gt_col[gt == 255] = [70, 70, 70]

        for col, (d, cmap) in enumerate(
            [(img, None), (seg_col, None), (depth, "inferno"), (gt_col, None)]
        ):
            ax = axes[row, col]
            ax.imshow(d, cmap=cmap)
            if row == 0: ax.set_title(headers[col], fontsize=13, fontweight="bold")
            ax.set_ylabel(scenes[row], fontsize=11, rotation=90)
            ax.axis("off")

    plt.suptitle("MTL Model — Multi-Scene Generalisation on Cityscapes Val Set",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print(f"[INFO] Device     : {DEVICE}")
    print(f"[INFO] Resolution : {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print(f"[INFO] Checkpoint : {BEST_CKPT}\n")

    if not os.path.isfile(BEST_CKPT):
        raise FileNotFoundError(
            f"MTL checkpoint not found: {BEST_CKPT}\n"
            "Train the MTL model first with:\n"
            "  python scripts/train_mtl.py\n"
            "The evaluator requires the state-dict checkpoint written after "
            "the first completed training epoch."
        )

    # ── Dataset ───────────────────────────
    val_ds = CityscapesDataset("data/cityscapes", "val", img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────
    encoder       = SegFormerEncoder()
    seg_decoder   = SegFormerDecoder(num_classes=NUM_CLASSES, embed_dim=256, dropout=0.1)
    depth_decoder = ProgressiveDepthDecoder(embed_dim=256)
    model         = MTLModel(encoder, seg_decoder, depth_decoder, NUM_CLASSES).to(DEVICE)

    model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Best MTL model loaded.\n")

    # ── Evaluation ────────────────────────
    seg_metric = SegMetric(num_classes=NUM_CLASSES)
    depth_acc  = {k: 0.0 for k in
                  ["abs_rel","rmse","mae","sq_rel","d1","d2","d3","coverage"]}

    samples_for_vis = []
    vis_indices = [9, 24, 48]

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating MTL (+ TTA)")):
        img      = batch["image"].to(DEVICE)
        seg_gt   = batch["seg"].to(DEVICE)
        depth_gt = batch["depth"].to(DEVICE)

        seg_prob, depth_pred = predict_tta(model, img)
        pred_class = torch.argmax(seg_prob, dim=1)

        seg_metric.update(pred_class, seg_gt)
        dm = all_depth_metrics(depth_pred, depth_gt)
        for k in depth_acc: depth_acc[k] += dm[k]

        if i in vis_indices:
            samples_for_vis.append((
                img.cpu(), pred_class.cpu(), depth_pred.cpu(), seg_gt.cpu()
            ))

    # ── Save visualisations ───────────────
    if samples_for_vis:
        save_multi_scene(samples_for_vis, os.path.join(LOG_DIR, "eval_multi.png"))

        # Also save detailed 5-panel for first scene
        batch0 = next(iter(DataLoader(
            CityscapesDataset("data/cityscapes", "val", img_size=IMG_SIZE),
            batch_size=1, shuffle=False
        )))
        img0 = batch0["image"].to(DEVICE)
        sg0  = batch0["seg"].to(DEVICE)
        dp0  = batch0["depth"].to(DEVICE)
        with torch.no_grad():
            sp0, dpr0 = predict_tta(model, img0)
        pred0 = torch.argmax(sp0, dim=1)
        save_single_eval(img0, pred0, dpr0, sg0, dp0,
                         os.path.join(LOG_DIR, "eval_sample.png"))

    # ── Metrics ───────────────────────────
    nv = len(val_loader)
    for k in depth_acc: depth_acc[k] /= nv
    val_miou, per_class = seg_metric.compute()

    def pct_change(our, base, lower_better=False):
        if lower_better:
            d = base - our
        else:
            d = our - base
        symbol = "+" if d >= 0 else "-"
        return f"{symbol}{abs(d)*100:.2f}pp"

    # ── Report ────────────────────────────
    lines = []
    lines.append("=" * 68)
    lines.append("         MTL MODEL — OFFICIAL EVALUATION REPORT")
    lines.append(f"         Cityscapes Val Set  |  {nv} images  |  {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    lines.append("=" * 68)
    lines.append(f"  Checkpoint : {BEST_CKPT}")
    lines.append(f"  Evaluation : With Test-Time Augmentation (horizontal flip)")
    lines.append("-" * 68)
    lines.append("  SEGMENTATION RESULTS (mIoU, ignore_index=255)")
    lines.append("-" * 68)
    lines.append(f"  {'Class':<16} | {'MTL IoU':>10} | {'Baseline IoU':>14} | {'Change':>10}")
    lines.append("-" * 68)

    for cls in CITYSCAPES_CLASSES:
        iou = per_class.get(cls, 0.0) * 100
        base_iou = SEG_BASELINE_PER_CLASS.get(cls, 0.0)
        diff = iou - base_iou
        sym = "+" if diff >= 0 else ""
        diff_str = f"{sym}{diff:.2f}%"
        lines.append(f"  {cls:<16} | {iou:>9.2f}% | {base_iou:>13.2f}% | {diff_str:>10}")

    lines.append("=" * 68)
    seg_change = pct_change(val_miou, SEG_BASELINE_MIOU)
    lines.append(f"  mIoU (MTL + TTA)      : {val_miou:.4f}  ({val_miou*100:.2f}%)")
    lines.append(f"  mIoU (Seg baseline)   : {SEG_BASELINE_MIOU:.4f}  ({SEG_BASELINE_MIOU*100:.2f}%)")
    lines.append(f"  Change vs baseline    : {seg_change}")
    lines.append("-" * 68)
    lines.append("  DEPTH RESULTS")
    lines.append("-" * 68)
    lines.append(f"  {'Metric':<20} | {'MTL':>10} | {'Baseline':>10} | {'Change':>10}")
    lines.append("-" * 68)
    lines.append(f"  {'AbsRel (lower better)':<24} | {depth_acc['abs_rel']:>10.4f} | {DEPTH_BASELINE_ABSR:>10.4f} | {pct_change(depth_acc['abs_rel'], DEPTH_BASELINE_ABSR, lower_better=True):>10}")
    lines.append(f"  {'RMSE (lower better)':<24} | {depth_acc['rmse']:>10.4f} | {DEPTH_BASELINE_RMSE:>10.4f} | {pct_change(depth_acc['rmse'], DEPTH_BASELINE_RMSE, lower_better=True):>10}")
    lines.append(f"  {'delta < 1.25 (higher)':<24} | {depth_acc['d1']:>10.4f} | {DEPTH_BASELINE_D1:>10.4f} | {pct_change(depth_acc['d1'], DEPTH_BASELINE_D1):>10}")
    lines.append(f"  {'delta < 1.25^2':<24} | {depth_acc['d2']:>10.4f} | {DEPTH_BASELINE_D2:>10.4f} | {pct_change(depth_acc['d2'], DEPTH_BASELINE_D2):>10}")
    lines.append(f"  {'delta < 1.25^3':<24} | {depth_acc['d3']:>10.4f} | {DEPTH_BASELINE_D3:>10.4f} | {pct_change(depth_acc['d3'], DEPTH_BASELINE_D3):>10}")
    lines.append(f"  {'MAE (lower better)':<24} | {depth_acc['mae']:>10.4f} | {DEPTH_BASELINE_MAE:>10.4f} | {pct_change(depth_acc['mae'], DEPTH_BASELINE_MAE, lower_better=True):>10}")
    lines.append(f"  {'SqRel (lower better)':<24} | {depth_acc['sq_rel']:>10.4f} | {DEPTH_BASELINE_SQREL:>10.4f} | {pct_change(depth_acc['sq_rel'], DEPTH_BASELINE_SQREL, lower_better=True):>10}")
    lines.append(f"  {'Valid pixels':<24} | {depth_acc['coverage']*100:>9.1f}% |      72.9% |            ")
    lines.append("=" * 68)
    combined = val_miou - depth_acc['abs_rel']
    lines.append(f"  Combined score (mIoU - AbsRel) : {combined:.4f}")
    lines.append("=" * 68)
    lines.append("  INTERPRETATION")
    lines.append("  The MTL model performs BOTH segmentation and depth estimation")
    lines.append("  in a single forward pass using ONE shared encoder.")
    lines.append("  Compare MTL efficiency vs running two separate models:")
    lines.append(f"  Seg-only params  : ~24.7M + Depth-only params: ~28.5M = ~53.2M total")
    lines.append(f"  MTL model params : shared encoder + 2 decoders = ~34.8M total (-34.6% params)")
    lines.append(f"  Inference speed  : 1 forward pass (~50% faster than running 2 separate backbones)")
    lines.append("=" * 68)

    report = "\n".join(lines)
    print("\n" + report)
    rpath = os.path.join(LOG_DIR, "eval_report.txt")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[INFO] Report saved -> {rpath}")


if __name__ == "__main__":
    main()