import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveDepthDecoder(nn.Module):
    """
    Progressive fusion depth decoder for SegFormer-B2 encoder.

    SegFormer-B2 hidden state channels:
        f1 (stage 1): 64   — highest resolution
        f2 (stage 2): 128
        f3 (stage 3): 320
        f4 (stage 4): 512  — lowest resolution

    All skip projections output 256 channels so residual additions
    are always dimension-consistent.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        # Project each encoder stage → embed_dim
        self.proj4 = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(320, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Refinement convs applied after each fusion step
        self.refine3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Final prediction head: embed_dim → 1 (inverse depth)
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _upsample_add(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample x to skip's spatial size, then element-wise add."""
        x = F.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        return x + skip

    def forward(self, features):
        """
        Args:
            features: tuple/list of 4 tensors from SegFormer encoder
                      ordered (stage1, stage2, stage3, stage4)
        Returns:
            depth: (B, 1, H, W) — sigmoid-activated inverse depth in [0, 1]
        """
        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]

        # Bottom-up: start from deepest feature
        x = self.proj4(f4)                          # (B, 256, H/32, W/32)

        x = self._upsample_add(x, self.proj3(f3))  # (B, 256, H/16, W/16)
        x = self.refine3(x)

        x = self._upsample_add(x, self.proj2(f2))  # (B, 256, H/8,  W/8)
        x = self.refine2(x)

        x = self._upsample_add(x, self.proj1(f1))  # (B, 256, H/4,  W/4)
        x = self.refine1(x)

        # Upsample to full input resolution (512 × 1024)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

        depth = torch.sigmoid(self.head(x))         # (B, 1, H, W)  ∈ [0, 1]
        return depth
