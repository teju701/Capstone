import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from utils.label_mapping import encode_segmap
from utils.depth_utils import disparity_to_depth

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', img_size=(512, 1024)):
        self.root = root
        self.split = split
        self.img_size = img_size

        self.images = []
        self.masks = []
        self.disparities = []

        left_root = os.path.join(root, 'leftImg8bit', split)
        gt_root = os.path.join(root, 'gtFine', split)
        disp_root = os.path.join(root, 'disparity', split)

        for city in os.listdir(left_root):
            for file in os.listdir(os.path.join(left_root, city)):
                if file.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(left_root, city, file))
                    self.masks.append(
                        os.path.join(
                            gt_root, city,
                            file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                        )
                    )
                    self.disparities.append(
                        os.path.join(
                            disp_root, city,
                            file.replace('_leftImg8bit.png', '_disparity.png')
                        )
                    )

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load RGB image
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size[::-1])

        # 2. Load segmentation mask
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        mask = encode_segmap(mask)

        # 3. Load disparity and convert to depth
        disp = cv2.imread(self.disparities[idx], cv2.IMREAD_UNCHANGED)
        disp = cv2.resize(disp, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        depth = disparity_to_depth(disp)

        # ====================================================
        # 4. DATA AUGMENTATION (ONLY FOR TRAIN)
        # ====================================================
        if self.split == "train" and random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            depth = np.fliplr(depth).copy()

        # ====================================================
        # 5. Convert to tensor / normalize
        # ====================================================
        img = self.img_transform(img)
        mask = torch.from_numpy(mask).long()
        depth = torch.from_numpy(depth).float()

        return {
            "image": img,
            "seg": mask,
            "depth": depth
        }

