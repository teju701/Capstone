import torch
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, encoder, decoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out
