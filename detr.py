'''
Slightly adjusted code from the DETR paper:
https://arxiv.org/abs/2005.12872 (Section A.6)
'''

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class DETR(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, nheads: int,
                 num_encoder_layers: int, num_decoder_layers: int):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        h = self.conv(x)

        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
            ], dim=-1)

        h = h.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(0, 1).unsqueeze(0)
        query = self.query_pos.unsqueeze(0)
        h = self.transformer(pos + h, query)

        return self.linear_class(h), self.linear_bbox(h).sigmoid()


if __name__ == "__main__":
    detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
    detr.eval()
    inputs = torch.randn(1, 3, 512, 512)
    logits, bboxes = detr(inputs)
