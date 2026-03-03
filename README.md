# DTDJO: Dehazing Target Detection Algorithm Based on Joint Optimization

This repository contains the implementation of DTDJO, a joint optimization framework for foggy water surface target detection.

## Overview

DTDJO is an end-to-end deep learning model that combines:
- **Dehazing Network**: Improved FFANet with MSFAM (Multi-Scale Feature Attention Module), RSC (Residual Skip Connection), and depthwise separable convolutions
- **Detection Network**: Improved YOLOv11 with MFAM (Mixed Feature Attention Module) for enhanced small target detection

## Key Features

- Multi-scale feature attention mechanism (PSA + CA)
- Residual modulated skip connections
- Depthwise separable convolutions for efficiency
- Mixed attention module for small target detection
- Joint optimization of dehazing and detection tasks

## Project Structure

```
DTDJO/
├── models/
│   ├── __init__.py
│   ├── dtdjo.py          # Main DTDJO model
│   ├── ffanet.py         # Improved FFANet dehazing network
│   ├── yolov11.py        # Improved YOLOv11 detection network
│   ├── msfam.py          # Multi-Scale Feature Attention Module
│   └── mfam.py           # Mixed Feature Attention Module
├── data/
│   ├── __init__.py
│   └── dataset.py        # Dataset loader
├── utils/
│   ├── __init__.py
│   ├── loss.py           # Loss functions
│   ├── metrics.py        # Evaluation metrics
│   └── utils.py          # Utility functions
├── configs/
│   └── config.yaml       # Configuration file
├── checkpoints/          # Model checkpoints
├── train.py              # Training script
├── test.py               # Testing script
├── requirements.txt      # Dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DTDJO.git
cd DTDJO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize your dataset as follows:
```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Label format (YOLO format):
```
<class_id> <x_center> <y_center> <width> <height>
```

## Training

Train the model:
```bash
python train.py --config configs/config.yaml --device cuda
```

Resume training:
```bash
python train.py --config configs/config.yaml --resume checkpoints/last_model.pth
```

## Testing

Test the trained model:
```bash
python test.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --output_dir results --save_images
```

## Configuration

Modify `configs/config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Dataset paths
- Loss weights
- Evaluation settings

## Results

The model achieves:
- **WSFOG_VisDrone**: 41.18% mAP (3% improvement over baseline)
- **WSFOG_VOC**: 89.95% mAP (4.11% improvement over baseline)

## Citation

If you use this code, please cite:
```
@article{dtdjo,
  title={Foggy Water Surface Target Detection Model Based on Joint Optimization},
  author={Haihua Zhang and Hongdong Wang},
  journal={Scientific Reports},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- FFANet for the base dehazing architecture
- YOLOv11 for the detection framework
- VisDrone and PASCAL VOC datasets
