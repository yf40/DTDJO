# DTDJO Project Structure

## Directory Tree

```
DTDJO/
├── models/                    # Network architectures
│   ├── __init__.py           # Package initialization
│   ├── dtdjo.py              # Main DTDJO model (combines dehazing + detection)
│   ├── ffanet.py             # Improved FFANet dehazing network
│   ├── yolov11.py            # Improved YOLOv11 detection network
│   ├── msfam.py              # Multi-Scale Feature Attention Module (PSA + CA)
│   └── mfam.py               # Mixed Feature Attention Module (Channel + Spatial)
│
├── data/                      # Data loading and preprocessing
│   ├── __init__.py           # Package initialization
│   └── dataset.py            # FoggyDataset class with augmentation
│
├── utils/                     # Utility functions
│   ├── __init__.py           # Package initialization
│   ├── loss.py               # Loss functions (Dehaze + YOLO + Joint)
│   ├── metrics.py            # Evaluation metrics (mAP, PSNR, SSIM)
│   └── utils.py              # General utilities (seed, checkpoint, NMS)
│
├── configs/                   # Configuration files
│   └── config.yaml           # Main configuration (model, data, train, test)
│
├── checkpoints/               # Saved model checkpoints
│   └── .gitkeep              # Keep directory in git
│
├── train.py                   # Training script
├── test.py                    # Testing script
├── inference.py               # Single image inference script
│
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation script
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
└── __init__.py               # Project initialization
```

## File Descriptions

### Models

1. **dtdjo.py** - Main DTDJO model
   - Combines dehazing and detection networks
   - Forward pass returns dehazed images and detections
   - Methods: forward(), get_dehazed_image(), detect_only()

2. **ffanet.py** - Improved FFANet
   - Multi-scale encoder-decoder architecture
   - Integrates MSFAM for feature attention
   - Uses depthwise separable convolutions
   - Includes residual skip connections (RSC)

3. **yolov11.py** - Improved YOLOv11
   - C2f modules for feature extraction
   - SPPF for multi-scale pooling
   - MFAM modules for small target enhancement
   - FPN-PAN neck for feature fusion

4. **msfam.py** - Multi-Scale Feature Attention Module
   - PSA: Pyramid Squeeze Attention (multi-scale spatial attention)
   - CA: Coordinate Attention (positional correlation)
   - Fusion of both attention mechanisms

5. **mfam.py** - Mixed Feature Attention Module
   - Pre-convolution 1x1 layer
   - Channel attention (avg + max pooling)
   - Spatial attention (avg + max across channels)

### Data

1. **dataset.py** - Dataset loader
   - FoggyDataset class for foggy images
   - Albumentations for augmentation
   - YOLO format label parsing
   - Custom collate_fn for batching

### Utils

1. **loss.py** - Loss functions
   - DehazeLoss: MSE + SSIM + Perceptual
   - YOLOLoss: Box + Objectness + Classification
   - JointLoss: Combined dehazing and detection loss

2. **metrics.py** - Evaluation metrics
   - compute_map(): Mean Average Precision
   - compute_psnr(): Peak Signal-to-Noise Ratio
   - compute_ssim(): Structural Similarity Index

3. **utils.py** - General utilities
   - set_seed(): Reproducibility
   - save/load_checkpoint(): Model persistence
   - non_max_suppression(): Post-processing
   - AverageMeter: Running statistics

### Scripts

1. **train.py** - Training pipeline
   - Configurable training loop
   - Joint optimization of both networks
   - Validation and checkpoint saving
   - Supports resume training

2. **test.py** - Evaluation pipeline
   - Comprehensive testing
   - Saves dehazed images and detections
   - Computes all metrics
   - Exports results to JSON

3. **inference.py** - Single image inference
   - Preprocesses input image
   - Runs dehazing and detection
   - Postprocesses and visualizes results
   - Saves output image

## Usage Flow

1. **Setup**: Install dependencies with `pip install -r requirements.txt`
2. **Data Preparation**: Organize dataset following YOLO format
3. **Configuration**: Modify `configs/config.yaml`
4. **Training**: Run `python train.py --config configs/config.yaml`
5. **Testing**: Run `python test.py --checkpoint checkpoints/best_model.pth`
6. **Inference**: Run `python inference.py --checkpoint checkpoints/best_model.pth --image input.jpg`

## Key Features

- **Modular Design**: Separate dehazing and detection modules
- **Joint Optimization**: End-to-end training of both tasks
- **Attention Mechanisms**: MSFAM for dehazing, MFAM for detection
- **Efficient Architecture**: Depthwise separable convolutions
- **Comprehensive Evaluation**: Multiple metrics (mAP, PSNR, SSIM)
- **Flexible Configuration**: YAML-based settings
- **Production Ready**: Inference script for deployment
