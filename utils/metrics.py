import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_map(predictions, targets, iou_threshold=0.5, num_classes=80):
    stats = defaultdict(lambda: {'tp': [], 'conf': [], 'pred_cls': [], 'target_cls': []})
    
    for pred, target in zip(predictions, targets):
        if pred is None:
            continue
        
        pred_boxes = pred[:, :4]
        pred_conf = pred[:, 4]
        pred_cls = pred[:, 5].int()
        
        target_boxes = target[:, 2:6]
        target_cls = target[:, 1].int()
        
        correct = np.zeros(len(pred_boxes))
        
        if len(target_boxes):
            detected = []
            for i, pred_box in enumerate(pred_boxes):
                ious = torch.tensor([compute_iou(pred_box.cpu().numpy(), tb.cpu().numpy()) for tb in target_boxes])
                max_iou, max_idx = ious.max(0)
                
                if max_iou > iou_threshold and max_idx not in detected and pred_cls[i] == target_cls[max_idx]:
                    correct[i] = 1
                    detected.append(max_idx)
        
        for c in range(num_classes):
            mask = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_p = mask.sum()
            
            if n_p == 0 and n_gt == 0:
                continue
            
            stats[c]['tp'].extend(correct[mask.cpu()])
            stats[c]['conf'].extend(pred_conf[mask].cpu().numpy())
            stats[c]['pred_cls'].extend([c] * n_p.item())
            stats[c]['target_cls'].extend([c] * n_gt.item())
    
    ap_dict = {}
    for c in range(num_classes):
        if len(stats[c]['tp']) == 0:
            ap_dict[c] = 0
            continue
        
        tp = np.array(stats[c]['tp'])
        conf = np.array(stats[c]['conf'])
        
        i = np.argsort(-conf)
        tp = tp[i]
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(1 - tp)
        
        recall = tp_cumsum / (len(stats[c]['target_cls']) + 1e-6)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap_dict[c] = compute_ap(recall, precision)
    
    mAP = np.mean(list(ap_dict.values()))
    return mAP, ap_dict


def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(img1, img2):
    from pytorch_msssim import ssim
    return ssim(img1, img2, data_range=1.0, size_average=True).item()
