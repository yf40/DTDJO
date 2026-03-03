import torch
import torch.nn as nn
import torch.nn.functional as F
from models.msfam import MSFAM


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualSkipConnection(nn.Module):
    def __init__(self, channels):
        super(ResidualSkipConnection, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


class FABlock(nn.Module):
    def __init__(self, channels):
        super(FABlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.conv3 = DepthwiseSeparableConv(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.msfam = MSFAM(channels)

    def forward(self, x):
        feat1 = self.relu(self.conv1(x))
        feat2 = self.relu(self.conv2(feat1))
        feat3 = self.relu(self.conv3(feat2))
        feat = self.msfam(feat3)
        return feat + x


class FFANet(nn.Module):
    def __init__(self, in_channels=3, num_blocks=3, channels=64):
        super(FFANet, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        self.encoder1 = self._make_layer(channels, num_blocks)
        self.down1 = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)
        
        self.encoder2 = self._make_layer(channels * 2, num_blocks)
        self.down2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1)
        
        self.middle = self._make_layer(channels * 4, num_blocks)
        
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
        self.decoder2 = self._make_layer(channels * 4, num_blocks)
        self.rsc2 = ResidualSkipConnection(channels * 4)
        
        self.up1 = nn.ConvTranspose2d(channels * 4, channels, kernel_size=2, stride=2)
        self.decoder1 = self._make_layer(channels * 2, num_blocks)
        self.rsc1 = ResidualSkipConnection(channels * 2)
        
        self.conv_out = nn.Conv2d(channels * 2, in_channels, kernel_size=3, padding=1)

    def _make_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(FABlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat_in = self.conv_in(x)
        
        enc1 = self.encoder1(feat_in)
        down1 = self.down1(enc1)
        
        enc2 = self.encoder2(down1)
        down2 = self.down2(enc2)
        
        middle = self.middle(down2)
        
        up2 = self.up2(middle)
        dec2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.rsc2(dec2)
        
        up1 = self.up1(dec2)
        dec1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.rsc1(dec1)
        
        out = self.conv_out(dec1)
        return out + x
