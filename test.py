import os
import argparse
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DTDJO
from data import FoggyDataset, collate_fn
from utils import compute_map, compute_psnr, compute_ssim, non_max_suppression
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def draw_boxes(image, boxes, classes, conf_thres=0.25):
    img = image.copy()
    
    for box in boxes:
        if box[4] < conf_thres:
            continue
        
        x1, y1, x2, y2 = map(int, box[:4])
        cls = int(box[5])
        conf = box[4]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{cls}: {conf:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img


def test(model, dataloader, device, config, output_dir, save_images=False):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    psnr_list = []
    ssim_list = []
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'dehazed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'detections'), exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (imgs, targets, img_paths) in enumerate(tqdm(dataloader, desc='Testing')):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            dehazed, detections = model(imgs)
            
            detections = non_max_suppression(
                detections,
                conf_thres=config['test']['conf_thres'],
                iou_thres=config['test']['iou_thres'],
                max_det=config['test']['max_det']
            )
            
            for i in range(imgs.size(0)):
                psnr = compute_psnr(dehazed[i:i+1], imgs[i:i+1])
                ssim = compute_ssim(dehazed[i:i+1], imgs[i:i+1])
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                if save_images:
                    dehazed_img = dehazed[i].cpu().numpy().transpose(1, 2, 0)
                    dehazed_img = (dehazed_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    dehazed_img = np.clip(dehazed_img, 0, 255).astype(np.uint8)
                    dehazed_img = cv2.cvtColor(dehazed_img, cv2.COLOR_RGB2BGR)
                    
                    img_name = os.path.basename(img_paths[i])
                    cv2.imwrite(os.path.join(output_dir, 'dehazed', img_name), dehazed_img)
                    
                    if detections[i] is not None:
                        detection_img = draw_boxes(
                            dehazed_img,
                            detections[i].cpu().numpy(),
                            None,
                            config['test']['conf_thres']
                        )
                        cv2.imwrite(os.path.join(output_dir, 'detections', img_name), detection_img)
            
            all_predictions.extend(detections)
            all_targets.extend([targets[targets[:, 0] == i, 1:] for i in range(imgs.size(0))])
    
    mAP, ap_dict = compute_map(all_predictions, all_targets, num_classes=config['model']['nc'])
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print(f"\nTest Results:")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"mAP: {mAP:.4f}")
    
    results = {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'mAP': mAP,
        'ap_per_class': ap_dict
    }
    
    import json
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    
    set_seed(config['seed'])
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = DTDJO(
        nc=config['model']['nc'],
        dehaze_channels=config['model']['dehaze_channels'],
        detect_channels=config['model']['detect_channels']
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    test_dataset = FoggyDataset(
        img_dir=config['data']['test_img_dir'],
        label_dir=config['data']['test_label_dir'],
        img_size=config['data']['img_size'],
        augment=False,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    results = test(model, test_loader, device, config, args.output_dir, args.save_images)
    
    print("\nTesting completed!")


if __name__ == '__main__':
    main()
