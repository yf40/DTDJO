import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        from torchvision import models
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.layers(pred)
        target_features = self.layers(target)
        return F.mse_loss(pred_features, target_features)


class DehazeLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super(DehazeLoss, self).__init__()
        self.mse = MSELoss()
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)
        perceptual_loss = self.perceptual(pred, target)
        total_loss = self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * perceptual_loss
        return total_loss, mse_loss, ssim_loss, perceptual_loss


class YOLOLoss(nn.Module):
    def __init__(self, nc=80):
        super(YOLOLoss, self).__init__()
        self.nc = nc
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_obj = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        device = predictions[0].device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        for i, pred in enumerate(predictions):
            b, _, h, w = pred.shape
            pred = pred.view(b, -1, self.nc + 5, h, w).permute(0, 1, 3, 4, 2).contiguous()
            
            if targets.shape[0] > 0:
                lbox += torch.sum(1 - self.bbox_iou(pred[..., :4], targets[:, 2:6]))
                lobj += self.bce_obj(pred[..., 4], torch.ones_like(pred[..., 4]))
                lcls += self.bce_cls(pred[..., 5:], torch.zeros_like(pred[..., 5:]))

        loss = lbox + lobj + lcls
        return loss, lbox, lobj, lcls

    def bbox_iou(self, box1, box2):
        b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
        b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
        b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter

        iou = inter / (union + 1e-16)
        return iou


class JointLoss(nn.Module):
    def __init__(self, nc=80, dehaze_weight=1.0, detect_weight=1.0):
        super(JointLoss, self).__init__()
        self.dehaze_loss = DehazeLoss()
        self.detect_loss = YOLOLoss(nc)
        self.dehaze_weight = dehaze_weight
        self.detect_weight = detect_weight

    def forward(self, dehazed, detections, clean_imgs, targets):
        dehaze_total, dehaze_mse, dehaze_ssim, dehaze_perceptual = self.dehaze_loss(dehazed, clean_imgs)
        detect_total, detect_box, detect_obj, detect_cls = self.detect_loss(detections, targets)
        
        total_loss = self.dehaze_weight * dehaze_total + self.detect_weight * detect_total
        
        return total_loss, dehaze_total, detect_total
