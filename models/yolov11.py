import torch
import torch.nn as nn
from models.mfam import MFAM


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True):
        super(C2f, self).__init__()
        self.c = out_channels // 2
        self.cv1 = ConvBlock(in_channels, out_channels, 1, 1, 0)
        self.cv2 = ConvBlock((2 + num_blocks) * self.c, out_channels, 1, 1, 0)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut) for _ in range(num_blocks)])

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(Bottleneck, self).__init__()
        self.cv1 = ConvBlock(in_channels, out_channels, 3, 1, 1)
        self.cv2 = ConvBlock(out_channels, out_channels, 3, 1, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.cv1 = ConvBlock(in_channels, c_, 1, 1, 0)
        self.cv2 = ConvBlock(c_ * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class DetectionHead(nn.Module):
    def __init__(self, nc=80, channels=()):
        super(DetectionHead, self).__init__()
        self.nc = nc
        self.nl = len(channels)
        self.no = nc + 5
        self.stride = torch.tensor([8., 16., 32.])
        
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(x, x, 3, 1, 1),
                ConvBlock(x, x, 3, 1, 1),
                nn.Conv2d(x, 4, 1)
            ) for x in channels
        ])
        
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(x, x, 3, 1, 1),
                ConvBlock(x, x, 3, 1, 1),
                nn.Conv2d(x, self.nc + 1, 1)
            ) for x in channels
        ])

    def forward(self, x):
        outputs = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            outputs.append(torch.cat([box, cls], 1))
        return outputs


class YOLOv11(nn.Module):
    def __init__(self, nc=80, channels=64):
        super(YOLOv11, self).__init__()
        self.nc = nc
        
        self.conv1 = ConvBlock(3, channels, 3, 2, 1)
        self.conv2 = ConvBlock(channels, channels * 2, 3, 2, 1)
        self.c2f1 = C2f(channels * 2, channels * 2, 2)
        
        self.conv3 = ConvBlock(channels * 2, channels * 4, 3, 2, 1)
        self.c2f2 = C2f(channels * 4, channels * 4, 2)
        
        self.conv4 = ConvBlock(channels * 4, channels * 8, 3, 2, 1)
        self.c2f3 = C2f(channels * 8, channels * 8, 2)
        
        self.conv5 = ConvBlock(channels * 8, channels * 16, 3, 2, 1)
        self.c2f4 = C2f(channels * 16, channels * 16, 2)
        self.sppf = SPPF(channels * 16, channels * 16)
        
        self.mfam1 = MFAM(channels * 4)
        self.mfam2 = MFAM(channels * 8)
        self.mfam3 = MFAM(channels * 16)
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f5 = C2f(channels * 24, channels * 8, 2)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f6 = C2f(channels * 12, channels * 4, 2)
        
        self.conv6 = ConvBlock(channels * 4, channels * 4, 3, 2, 1)
        self.c2f7 = C2f(channels * 12, channels * 8, 2)
        
        self.conv7 = ConvBlock(channels * 8, channels * 8, 3, 2, 1)
        self.c2f8 = C2f(channels * 24, channels * 16, 2)
        
        self.head = DetectionHead(nc, (channels * 4, channels * 8, channels * 16))

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.c2f1(c2)
        
        c4 = self.conv3(c3)
        c5 = self.c2f2(c4)
        c5 = self.mfam1(c5)
        
        c6 = self.conv4(c5)
        c7 = self.c2f3(c6)
        c7 = self.mfam2(c7)
        
        c8 = self.conv5(c7)
        c9 = self.c2f4(c8)
        c10 = self.sppf(c9)
        c10 = self.mfam3(c10)
        
        p5 = self.upsample1(c10)
        p5 = torch.cat([p5, c7], 1)
        p5 = self.c2f5(p5)
        
        p4 = self.upsample2(p5)
        p4 = torch.cat([p4, c5], 1)
        p4 = self.c2f6(p4)
        
        p6 = self.conv6(p4)
        p6 = torch.cat([p6, p5], 1)
        p6 = self.c2f7(p6)
        
        p7 = self.conv7(p6)
        p7 = torch.cat([p7, c10], 1)
        p7 = self.c2f8(p7)
        
        outputs = self.head([p4, p6, p7])
        return outputs
