import torch
import torch.nn as nn


class DepthModel(nn.Module):
    """
    Depth estimation model: shared SegFormer encoder + progressive fusion decoder.

    Args:
        encoder: SegFormerEncoder instance
        decoder: ProgressiveDepthDecoder instance
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized RGB image
        Returns:
            depth: (B, 1, H, W) inverse depth in [0, 1]
        """
        features = self.encoder(x)   # 4-tuple of hidden states
        depth = self.decoder(features)
        return depth
