import torch
import torch.nn as nn
from transformers import SegformerModel

class SegFormerEncoder(nn.Module):
    def __init__(self,
                 model_name="nvidia/segformer-b2-finetuned-cityscapes-1024-1024"):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(model_name)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x,
                                output_hidden_states=True)
        return outputs.hidden_states
