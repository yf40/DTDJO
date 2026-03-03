import torch
import torch.nn as nn
import torch.nn.functional as F


class PSA(nn.Module):
    def __init__(self, channels):
        super(PSA, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 4, channels, kernel_size=1)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        feat1 = self.pool1(x)
        feat2 = F.interpolate(self.pool2(x), size=(h, w), mode='bilinear', align_corners=False)
        feat3 = F.interpolate(self.pool3(x), size=(h, w), mode='bilinear', align_corners=False)
        feat = feat1 + feat2 + feat3
        feat = self.conv1(feat)
        feat = F.relu(feat)
        feat = self.conv2(feat)
        att = self.sigmoid(feat)
        return x * att


class CA(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels // reduction)
        self.conv_h = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn(y)
        y = F.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        att_h = self.sigmoid(self.conv_h(x_h))
        att_w = self.sigmoid(self.conv_w(x_w))
        return x * att_h * att_w


class MSFAM(nn.Module):
    def __init__(self, channels):
        super(MSFAM, self).__init__()
        self.psa = PSA(channels)
        self.ca = CA(channels)
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        psa_out = self.psa(x)
        ca_out = self.ca(x)
        fused = torch.cat([psa_out, ca_out], dim=1)
        out = self.conv(fused)
        return out + x
