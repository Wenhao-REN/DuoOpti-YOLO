import torch
import torch.nn as nn
import torch.nn.functional as F

class SA_C2f(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SA_C2f, self).__init__()


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.LayerNorm([channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )


        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2),
            nn.Sigmoid()
        )


        self.fusion_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):

        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        ca_input = avg_out + max_out
        ca = self.channel_mlp(ca_input)


        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_map, max_map], dim=1)
        sa = self.spatial_att(sa_input)


        out = x * ca * sa
        return x + self.fusion_weight * out
