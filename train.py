import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from models import DTDJO
from data import FoggyDataset, collate_fn
from utils import JointLoss, compute_map, compute_psnr, compute_ssim
from utils import set_seed, save_checkpoint, load_checkpoint, load_config, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    
    total_loss_meter = AverageMeter()
    dehaze_loss_meter = AverageMeter()
    detect_loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (imgs, targets, _) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        clean_imgs = imgs.clone()
        
        optimizer.zero_grad()
        
        dehazed, detections = model(imgs)
        
        loss, dehaze_loss, detect_loss = criterion(dehazed, detections, clean_imgs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss_meter.update(loss.item(), imgs.size(0))
        dehaze_loss_meter.update(dehaze_loss.item(), imgs.size(0))
        detect_loss_meter.update(detect_loss.item(), imgs.size(0))
        
        pbar.set_postfix({
            'total': f'{total_loss_meter.avg:.4f}',
            'dehaze': f'{dehaze_loss_meter.avg:.4f}',
            'detect': f'{detect_loss_meter.avg:.4f}'
        })
    
    return total_loss_meter.avg, dehaze_loss_meter.avg, detect_loss_meter.avg


def validate(model, dataloader, device):
    model.eval()
    
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, targets, _ in tqdm(dataloader, desc='Validating'):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            dehazed, detections = model(imgs)
            
            psnr = compute_psnr(dehazed, imgs)
            ssim = compute_ssim(dehazed, imgs)
            
            psnr_meter.update(psnr, imgs.size(0))
            ssim_meter.update(ssim, imgs.size(0))
            
            all_predictions.extend(detections)
            all_targets.extend([targets[targets[:, 0] == i, 1:] for i in range(imgs.size(0))])
    
    mAP, _ = compute_map(all_predictions, all_targets)
    
    return psnr_meter.avg, ssim_meter.avg, mAP


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
    
    train_dataset = FoggyDataset(
        img_dir=config['data']['train_img_dir'],
        label_dir=config['data']['train_label_dir'],
        img_size=config['data']['img_size'],
        augment=True,
        is_train=True
    )
    
    val_dataset = FoggyDataset(
        img_dir=config['data']['val_img_dir'],
        label_dir=config['data']['val_label_dir'],
        img_size=config['data']['img_size'],
        augment=False,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    criterion = JointLoss(
        nc=config['model']['nc'],
        dehaze_weight=config['train']['dehaze_weight'],
        detect_weight=config['train']['detect_weight']
    )
    
    if config['optimizer']['type'] == 'Adam':
        optimizer = Adam(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay'],
            betas=config['optimizer']['betas']
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=config['train']['learning_rate'],
            momentum=config['train']['momentum'],
            weight_decay=config['train']['weight_decay']
        )
    
    if config['train']['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    start_epoch = 0
    best_map = 0.0
    
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
    
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    
    for epoch in range(start_epoch, config['train']['epochs']):
        train_loss, dehaze_loss, detect_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Dehaze={dehaze_loss:.4f}, Detect={detect_loss:.4f}")
        
        if (epoch + 1) % config['train']['eval_interval'] == 0:
            psnr, ssim, mAP = validate(model, val_loader, device)
            print(f"Validation: PSNR={psnr:.2f}, SSIM={ssim:.4f}, mAP={mAP:.4f}")
            
            if mAP > best_map:
                best_map = mAP
                save_checkpoint(
                    model, optimizer, epoch, train_loss,
                    os.path.join(config['checkpoint']['save_dir'], config['checkpoint']['best_model'])
                )
        
        if (epoch + 1) % config['train']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(config['checkpoint']['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            )
        
        scheduler.step()
    
    save_checkpoint(
        model, optimizer, config['train']['epochs'] - 1, train_loss,
        os.path.join(config['checkpoint']['save_dir'], config['checkpoint']['last_model'])
    )
    
    print(f"Training completed! Best mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()
