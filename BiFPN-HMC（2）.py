import torch
import torch.nn as nn
import torch.nn.functional as F

class SeAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg, max_val], dim=1)
        return x * self.conv(out)

class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, act=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            act(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class WeightedFusion(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * x for w, x in zip(weights, inputs))
        return fused

class BiFPN_HMC(nn.Module):
    def __init__(self, channels=64):
        super().__init__()


        self.conv_p3 = ConvBnAct(channels, channels)
        self.conv_p4 = ConvBnAct(channels, channels)
        self.conv_p5 = ConvBnAct(channels, channels)


        self.conv_td4 = ConvBnAct(channels, channels)
        self.conv_td3 = ConvBnAct(channels, channels)


        self.wf_4 = WeightedFusion(2)
        self.wf_3 = WeightedFusion(2)

        self.ca3 = SeAttention(channels)
        self.ca4 = SeAttention(channels)
        self.ca5 = SeAttention(channels)
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()


        self.out_conv3 = ConvBnAct(channels, channels)
        self.out_conv4 = ConvBnAct(channels, channels)
        self.out_conv5 = ConvBnAct(channels, channels)

    def forward(self, p3, p4, p5):

        p3 = self.ca3(self.sa3(self.conv_p3(p3)))
        p4 = self.ca4(self.sa4(self.conv_p4(p4)))
        p5 = self.ca5(self.sa5(self.conv_p5(p5)))


        td4 = self.conv_td4(self.wf_4([p4, F.interpolate(p5, scale_factor=2)]))
        td3 = self.conv_td3(self.wf_3([p3, F.interpolate(td4, scale_factor=2)]))


        out3 = self.out_conv3(td3)
        out4 = self.out_conv4(td4)
        out5 = self.out_conv5(p5)

        return out3, out4, out5
