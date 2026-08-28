import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerDecoder(nn.Module):
    """
    SegFormer All-MLP Decode Head.

    Fuses all 4 encoder stages from SegFormer-B2:
        f1: (B,  64, H/4,  W/4)
        f2: (B, 128, H/8,  W/8)
        f3: (B, 320, H/16, W/16)
        f4: (B, 512, H/32, W/32)

    Pipeline:
        1. Linear-project each stage to embed_dim (256)
        2. Upsample all to H/4 x W/4
        3. Concatenate → (B, 4*embed_dim, H/4, W/4)
        4. Fuse → (B, embed_dim, H/4, W/4)
        5. Dropout → Classify → (B, num_classes, H/4, W/4)
        6. Upsample 4x → (B, num_classes, H, W)
    """

    def __init__(self, num_classes: int, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        # SegFormer-B2 encoder output channels per stage
        in_channels = [64, 128, 320, 512]

        # Step 1: Project each stage to embed_dim with 1x1 conv
        self.proj1 = nn.Conv2d(in_channels[0], embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels[1], embed_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(in_channels[2], embed_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(in_channels[3], embed_dim, kernel_size=1)

        # Step 4: Fuse concatenated features
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Step 5: Dropout + final classifier
        self.dropout    = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: 4-tuple from SegFormerEncoder — (f1, f2, f3, f4)
        Returns:
            logits: (B, num_classes, H, W) at full input resolution
        """
        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]

        # Upsample all stages to f1 resolution (H/4 x W/4)
        target_size = f1.shape[2:]

        x1 = self.proj1(f1)
        x2 = F.interpolate(self.proj2(f2), size=target_size, mode="bilinear", align_corners=False)
        x3 = F.interpolate(self.proj3(f3), size=target_size, mode="bilinear", align_corners=False)
        x4 = F.interpolate(self.proj4(f4), size=target_size, mode="bilinear", align_corners=False)

        # Concat + fuse
        x = torch.cat([x1, x2, x3, x4], dim=1)   # (B, 4*embed_dim, H/4, W/4)
        x = self.fuse(x)                           # (B, embed_dim,   H/4, W/4)

        # Classify
        x = self.dropout(x)
        x = self.classifier(x)                     # (B, num_classes, H/4, W/4)

        # Final 4x upsample to full resolution — NOT 32x like the old decoder
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return x