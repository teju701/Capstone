import torch
import torch.nn as nn


class MTLModel(nn.Module):
    """
    Multi-Task Learning model for joint semantic segmentation
    and monocular depth estimation.

    Architecture:
        - ONE shared SegFormer-B2 encoder (feature extraction done once)
        - TWO task-specific decoders:
            seg_decoder  : All-MLP head → (B, 19, H, W) class logits
            depth_decoder: Progressive Fusion head → (B, 1, H, W) depth

    The encoder runs once per forward pass. Both decoders consume
    the same 4-stage feature maps simultaneously. This is the core
    efficiency argument of the MTL design.
    """

    def __init__(self,
                 encoder: nn.Module,
                 seg_decoder: nn.Module,
                 depth_decoder: nn.Module,
                 num_classes: int = 19):
        super().__init__()
        self.encoder       = encoder
        self.seg_decoder   = seg_decoder
        self.depth_decoder = depth_decoder
        self.num_classes   = num_classes

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) normalised RGB image

        Returns:
            seg_out   : (B, 19, H, W)  — class logits (raw, no softmax)
            depth_out : (B, 1,  H, W)  — sigmoid-activated depth in [0, 1]
        """
        # Shared encoder — runs ONCE
        features = self.encoder(x)          # tuple: (f1, f2, f3, f4)

        # Task-specific decoders — both read the same features
        seg_out   = self.seg_decoder(features)
        depth_out = self.depth_decoder(features)

        return seg_out, depth_out