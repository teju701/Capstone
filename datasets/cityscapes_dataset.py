import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils.label_mapping import encode_segmap
from utils.depth_utils import disparity_to_depth


class CityscapesDataset(Dataset):
    """
    Cityscapes dataset loader for segmentation and depth.

    Default resolution: 1024×1024 (square crop — matches the pretrained
    SegFormer-B2 checkpoint: nvidia/segformer-b2-finetuned-cityscapes-1024-1024)

    Augmentation (train only):
        - Random scale in [0.5, 2.0] + random crop to img_size
        - Random horizontal flip
        - Color jitter (brightness, contrast, saturation, hue)
    """

    def __init__(self, root, split='train', img_size=(1024, 1024)):
        self.root     = root
        self.split    = split
        self.img_size = img_size          # (H, W)

        self.images      = []
        self.masks       = []
        self.disparities = []

        left_root = os.path.join(root, 'leftImg8bit', split)
        gt_root   = os.path.join(root, 'gtFine',      split)
        disp_root = os.path.join(root, 'disparity',   split)

        for city in sorted(os.listdir(left_root)):
            city_img  = os.path.join(left_root, city)
            city_gt   = os.path.join(gt_root,   city)
            city_disp = os.path.join(disp_root, city)

            for fname in sorted(os.listdir(city_img)):
                if not fname.endswith('_leftImg8bit.png'):
                    continue
                base = fname.replace('_leftImg8bit.png', '')
                self.images.append(os.path.join(city_img, fname))
                self.masks.append(os.path.join(
                    city_gt, f"{base}_gtFine_labelIds.png"))
                self.disparities.append(os.path.join(
                    city_disp, f"{base}_disparity.png"))

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __len__(self):
        return len(self.images)

    # ── Augmentation helpers ──────────────────────────────────────────────────

    def _random_scale_crop(self, img, mask, depth):
        """
        Random scale in [0.5, 2.0], then crop to self.img_size.
        img: bilinear, mask/depth: nearest neighbour.
        Padding with reflect/constant if scaled size < crop size.
        """
        H, W  = self.img_size
        scale = random.uniform(0.5, 2.0)
        new_h = int(H * scale)
        new_w = int(W * scale)

        img   = cv2.resize(img,   (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_h = max(0, H - new_h)
        pad_w = max(0, W - new_w)
        if pad_h > 0 or pad_w > 0:
            img   = cv2.copyMakeBorder(img,   0, pad_h, 0, pad_w,
                                       cv2.BORDER_REFLECT_101)
            mask  = cv2.copyMakeBorder(mask,  0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=255)
            depth = cv2.copyMakeBorder(depth, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=0)

        h_now, w_now = img.shape[:2]
        top  = random.randint(0, h_now - H)
        left = random.randint(0, w_now - W)

        img   = img  [top:top+H, left:left+W]
        mask  = mask [top:top+H, left:left+W]
        depth = depth[top:top+H, left:left+W]

        return img, mask, depth

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        # 1. Load RGB — resize to img_size first so scale-crop has a base
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]),
                         interpolation=cv2.INTER_LINEAR)

        # 2. Segmentation mask
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                          interpolation=cv2.INTER_NEAREST)
        mask = encode_segmap(mask)

        # 3. Disparity → depth
        disp  = cv2.imread(self.disparities[idx], cv2.IMREAD_UNCHANGED)
        disp  = cv2.resize(disp, (self.img_size[1], self.img_size[0]),
                           interpolation=cv2.INTER_NEAREST)
        depth = disparity_to_depth(disp)

        # ── TRAIN AUGMENTATIONS ───────────────────────────────────────────────
        if self.split == 'train':

            # (a) Random scale + crop
            img, mask, depth = self._random_scale_crop(img, mask, depth)

            # (b) Random horizontal flip
            if random.random() > 0.5:
                img   = np.fliplr(img).copy()
                mask  = np.fliplr(mask).copy()
                depth = np.fliplr(depth).copy()

            # (c) Color jitter — RGB only
            img_pil = TF.to_pil_image(img)
            img_pil = self.color_jitter(img_pil)
            img     = np.array(img_pil)

        # ── Tensors ───────────────────────────────────────────────────────────
        img   = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        img   = self.normalize(img)
        mask  = torch.from_numpy(mask).long()
        depth = torch.from_numpy(depth).float()

        return {"image": img, "seg": mask, "depth": depth}