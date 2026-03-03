import torch
import torch.nn as nn
from models.ffanet import FFANet
from models.yolov11 import YOLOv11


class DTDJO(nn.Module):
    def __init__(self, nc=80, dehaze_channels=64, detect_channels=64):
        super(DTDJO, self).__init__()
        self.dehaze_net = FFANet(in_channels=3, num_blocks=3, channels=dehaze_channels)
        self.detect_net = YOLOv11(nc=nc, channels=detect_channels)

    def forward(self, x):
        dehazed = self.dehaze_net(x)
        detections = self.detect_net(dehazed)
        return dehazed, detections

    def get_dehazed_image(self, x):
        return self.dehaze_net(x)

    def detect_only(self, x):
        return self.detect_net(x)
