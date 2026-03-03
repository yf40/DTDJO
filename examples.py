import torch
from models import DTDJO, FFANet, YOLOv11


def example_1_basic_usage():
    print("Example 1: Basic DTDJO Usage")
    
    model = DTDJO(nc=80, dehaze_channels=64, detect_channels=64)
    
    dummy_input = torch.randn(1, 3, 640, 640)
    
    dehazed, detections = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Dehazed shape: {dehazed.shape}")
    print(f"Number of detection scales: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  Detection scale {i}: {det.shape}")
    print()


def example_2_dehaze_only():
    print("Example 2: Dehazing Only")
    
    model = FFANet(in_channels=3, num_blocks=3, channels=64)
    
    foggy_image = torch.randn(1, 3, 640, 640)
    
    clear_image = model(foggy_image)
    
    print(f"Foggy image shape: {foggy_image.shape}")
    print(f"Clear image shape: {clear_image.shape}")
    print()


def example_3_detect_only():
    print("Example 3: Detection Only")
    
    model = YOLOv11(nc=80, channels=64)
    
    image = torch.randn(1, 3, 640, 640)
    
    detections = model(image)
    
    print(f"Input shape: {image.shape}")
    print(f"Number of detection scales: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  Detection scale {i}: {det.shape}")
    print()


def example_4_load_and_save():
    print("Example 4: Load and Save Model")
    
    model = DTDJO(nc=80)
    
    torch.save(model.state_dict(), 'temp_model.pth')
    print("Model saved to temp_model.pth")
    
    new_model = DTDJO(nc=80)
    new_model.load_state_dict(torch.load('temp_model.pth'))
    print("Model loaded from temp_model.pth")
    
    import os
    os.remove('temp_model.pth')
    print("Temporary file removed")
    print()


def example_5_inference_mode():
    print("Example 5: Inference Mode")
    
    model = DTDJO(nc=80)
    model.eval()
    
    with torch.no_grad():
        foggy_image = torch.randn(1, 3, 640, 640)
        dehazed, detections = model(foggy_image)
    
    print(f"Inference completed")
    print(f"Dehazed image shape: {dehazed.shape}")
    print(f"Detections: {len(detections)} scales")
    print()


def example_6_batch_processing():
    print("Example 6: Batch Processing")
    
    model = DTDJO(nc=80)
    
    batch_size = 4
    batch_images = torch.randn(batch_size, 3, 640, 640)
    
    dehazed_batch, detection_batch = model(batch_images)
    
    print(f"Batch size: {batch_size}")
    print(f"Dehazed batch shape: {dehazed_batch.shape}")
    print(f"Detection scales: {len(detection_batch)}")
    print()


def example_7_cuda_usage():
    print("Example 7: CUDA Usage")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DTDJO(nc=80).to(device)
    
    foggy_image = torch.randn(1, 3, 640, 640).to(device)
    
    with torch.no_grad():
        dehazed, detections = model(foggy_image)
    
    print(f"Model on device: {next(model.parameters()).device}")
    print(f"Output on device: {dehazed.device}")
    print()


if __name__ == '__main__':
    print("DTDJO Usage Examples\n" + "="*50 + "\n")
    
    example_1_basic_usage()
    example_2_dehaze_only()
    example_3_detect_only()
    example_4_load_and_save()
    example_5_inference_mode()
    example_6_batch_processing()
    example_7_cuda_usage()
    
    print("All examples completed!")
