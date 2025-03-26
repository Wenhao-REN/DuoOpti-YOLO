import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.bifpn_hmc import BiFPN_HMC
from modules.sa_c2f import SA_C2f


class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, act=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            act(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBnAct(in_channels, in_channels // 2, 3, 1, 1)
        self.conv2 = ConvBnAct(in_channels // 2, in_channels // 2, 3, 1, 1)
        self.pred = nn.Conv2d(in_channels // 2, num_classes + 5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pred(x)


class DuoOptiYOLO(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.0):
        super().__init__()
        ch = int(64 * width_mult)


        self.stem = nn.Sequential(
            ConvBnAct(3, ch, 3, 2, 1),
            ConvBnAct(ch, ch, 3, 1, 1)
        )
        self.stage1 = ConvBnAct(ch, ch * 2, 3, 2, 1)
        self.stage2 = ConvBnAct(ch * 2, ch * 4, 3, 2, 1)
        self.stage3 = ConvBnAct(ch * 4, ch * 8, 3, 2, 1)


        self.bifpn1 = BiFPN_HMC(ch * 2)
        self.att1 = SA_C2f(ch * 2)


        self.reduce1 = ConvBnAct(ch * 2, ch * 4, 3, 2, 1)


        self.bifpn2 = BiFPN_HMC(ch * 4)
        self.att2 = SA_C2f(ch * 4)


        self.head_low = DetectionHead(ch * 2, num_classes)
        self.head_high = DetectionHead(ch * 4, num_classes)

    def forward(self, x):

        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

    
        p3, p4, _ = self.bifpn1(x1, x2, x3)
        a_low = self.att1(p3)

        reduced = self.reduce1(a_low)
        p5, p6, _ = self.bifpn2(reduced, x3, x3)
        a_high = self.att2(p5)

        out_low = self.head_low(a_low)
        out_high = self.head_high(a_high)

        return [out_low, out_high]
