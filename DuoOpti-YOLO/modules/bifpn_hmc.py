import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BiFPN_HMC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3 = ConvBnRelu(channels, channels, 3, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.conv3(x)
        ca = self.ca(self.global_pool(x))
        return residual + x * ca
