import numpy as np
import torch

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
]


class SegmentationMetric:
    """
    Standard academic evaluation metric accumulator for semantic segmentation.
    Accumulates Intersection and Union globally across the entire evaluation dataset
    as required by standard Cityscapes benchmarks (Cityscapes Benchmark, MMSeg, SegFormer).
    """
    def __init__(self, num_classes: int = 19, ignore_index: int = 255, class_names=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or CITYSCAPES_CLASSES
        self.reset()

    def reset(self):
        self.total_inter = np.zeros(self.num_classes, dtype=np.float64)
        self.total_union = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()

        pred = pred.reshape(-1)
        target = target.reshape(-1)

        valid = (target != self.ignore_index)
        p = pred[valid]
        t = target[valid]

        for c in range(self.num_classes):
            p_c = (p == c)
            t_c = (t == c)
            inter = np.logical_and(p_c, t_c).sum()
            union = np.logical_or(p_c, t_c).sum()
            self.total_inter[c] += inter
            self.total_union[c] += union

    def compute(self):
        ious = np.zeros(self.num_classes, dtype=np.float64)
        valid_classes = self.total_union > 0
        ious[valid_classes] = self.total_inter[valid_classes] / self.total_union[valid_classes]
        miou = float(np.mean(ious))
        per_class = {
            self.class_names[c]: float(ious[c])
            for c in range(self.num_classes)
        }
        return miou, per_class
