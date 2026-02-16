import torch
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from dataset.cityscapes_dataset import CityscapesDataset
from utils.visualize import visualize_sample

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def save_sample(sample, idx, out_dir):
    """
    Save one preprocessed sample (RGB, segmentation, depth) for demo purposes.
    """
    os.makedirs(out_dir, exist_ok=True)

    # -------- RGB (de-normalize before saving) --------
    img = sample["image"].permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(
        os.path.join(out_dir, f"sample_{idx:02d}_rgb.png"),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )

    # -------- Segmentation --------
    seg = sample["seg"].numpy().astype(np.uint8)
    cv2.imwrite(
        os.path.join(out_dir, f"sample_{idx:02d}_seg.png"),
        seg
    )

    # -------- Depth (normalized for visualization only) --------
    depth = sample["depth"].numpy()
    depth_vis = (depth / (depth.max() + 1e-8) * 255).astype(np.uint8)

    cv2.imwrite(
        os.path.join(out_dir, f"sample_{idx:02d}_depth.png"),
        depth_vis
    )


def main():
    # --------------------------------------------------
    # 1. Initialize Dataset
    # --------------------------------------------------
    dataset = CityscapesDataset(
        root="data/cityscapes",
        split="train"
    )

    print(f"[INFO] Dataset size: {len(dataset)} samples")

    # --------------------------------------------------
    # 2. Wrap Dataset in DataLoader
    # --------------------------------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # --------------------------------------------------
    # 3. Fetch One Batch
    # --------------------------------------------------
    batch = next(iter(dataloader))

    print("[INFO] Batch keys:", batch.keys())
    print("[INFO] Image shape:", batch["image"].shape)
    print("[INFO] Segmentation shape:", batch["seg"].shape)
    print("[INFO] Depth shape:", batch["depth"].shape)

    # --------------------------------------------------
    # 4. Sanity Checks
    # --------------------------------------------------
    assert batch["image"].shape == (4, 3, 512, 1024), "❌ Image shape mismatch"
    assert batch["seg"].shape == (4, 512, 1024), "❌ Segmentation shape mismatch"
    assert batch["depth"].shape == (4, 512, 1024), "❌ Depth shape mismatch"

    assert torch.isfinite(batch["depth"]).all(), "❌ Depth contains NaNs or Infs"

    print("[INFO] All sanity checks passed ✅")

    # --------------------------------------------------
    # 5. Visualize ONE sample
    # --------------------------------------------------
    sample = {
        "image": batch["image"][0],
        "seg": batch["seg"][0],
        "depth": batch["depth"][0]
    }

    visualize_sample(sample)
    print("[INFO] Dataset visualization successful 🎉")

    # --------------------------------------------------
    # 6. SAVE PREPROCESSING DEMO SAMPLES (FOR PANEL)
    # --------------------------------------------------
    output_dir = "outputs/preprocessing_demo"

    for i in range(5):   # Save first 5 samples
        sample = dataset[i]
        save_sample(sample, i + 1, output_dir)

    print(f"[INFO] Saved preprocessing demo samples to: {output_dir}")


if __name__ == "__main__":
    main()