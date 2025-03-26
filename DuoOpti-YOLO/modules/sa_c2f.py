import torch
import torch.nn as nn

class SA_C2f(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(self.avg_pool(x))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * ca * sa
