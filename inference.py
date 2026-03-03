import argparse
import torch
import cv2
import numpy as np
from models import DTDJO
from utils import load_config, non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.jpg')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def preprocess_image(image_path, img_size=640):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = img_resized
    
    img_tensor = torch.from_numpy(canvas).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, (h, w), (new_h, new_w)


def postprocess_boxes(boxes, orig_size, resized_size, img_size=640):
    orig_h, orig_w = orig_size
    resized_h, resized_w = resized_size
    
    scale_h = orig_h / resized_h
    scale_w = orig_w / resized_w
    
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h
    
    return boxes


def draw_boxes(image, boxes, conf_thres=0.25):
    img = image.copy()
    
    for box in boxes:
        if box[4] < conf_thres:
            continue
        
        x1, y1, x2, y2 = map(int, box[:4])
        cls = int(box[5])
        conf = box[4]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'Class {cls}: {conf:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img


def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = DTDJO(
        nc=config['model']['nc'],
        dehaze_channels=config['model']['dehaze_channels'],
        detect_channels=config['model']['detect_channels']
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    img_tensor, orig_size, resized_size = preprocess_image(args.image, config['data']['img_size'])
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        dehazed, detections = model(img_tensor)
        
        detections = non_max_suppression(
            detections,
            conf_thres=config['test']['conf_thres'],
            iou_thres=config['test']['iou_thres'],
            max_det=config['test']['max_det']
        )
    
    dehazed_img = dehazed[0].cpu().numpy().transpose(1, 2, 0)
    dehazed_img = (dehazed_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    dehazed_img = np.clip(dehazed_img, 0, 255).astype(np.uint8)
    
    dehazed_img = dehazed_img[:resized_size[0], :resized_size[1]]
    dehazed_img = cv2.resize(dehazed_img, (orig_size[1], orig_size[0]))
    dehazed_img = cv2.cvtColor(dehazed_img, cv2.COLOR_RGB2BGR)
    
    if detections[0] is not None:
        boxes = detections[0].cpu().numpy()
        boxes = postprocess_boxes(boxes, orig_size, resized_size, config['data']['img_size'])
        result_img = draw_boxes(dehazed_img, boxes, config['test']['conf_thres'])
    else:
        result_img = dehazed_img
    
    cv2.imwrite(args.output, result_img)
    print(f"Result saved to {args.output}")


if __name__ == '__main__':
    main()
