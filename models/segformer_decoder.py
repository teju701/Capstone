import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, features):
        x = features[-1]      # last stage
        x = self.classifier(x)

        x = F.interpolate(
            x,
            scale_factor=32,
            mode="bilinear",
            align_corners=False
        )
        return x
