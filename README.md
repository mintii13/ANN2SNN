# ANN2SNN: Firing Rate-based SNN Conversion for Anomaly Detection

This repository implements a novel ANN-to-SNN conversion approach for energy-efficient image anomaly detection using reconstruction methods. Our method introduces a firing rate-based reconstruction head that enables high-quality continuous value generation from spiking neural networks.

## Overview

Traditional SNN conversion methods struggle with reconstruction tasks due to the mismatch between discrete spikes and continuous outputs required for image reconstruction. Our approach addresses this challenge through:

- **Firing Rate-based Reconstruction**: Novel temporal spike accumulation mechanism for continuous value approximation
- **Energy Efficiency**: 3,900-10,600× energy reduction compared to ANN baselines
- **Direct Conversion**: No retraining required - works with pre-trained ANN models
- **Competitive Performance**: Superior to specialized SNN methods (FSVAE, ESVAE) on 10/15 MVTec categories

## Requirements

```
torch>=1.12.0
torchvision>=0.13.0
spikingjelly>=0.0.0.0.14
scikit-image>=0.19.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm
```

## Installation

```bash
git clone https://github.com/mintii13/ANN2SNN.git
cd ANN2SNN
pip install -r requirements.txt
```

## Dataset and Checkpoint Setup

### Dataset Structure

1. Download MVTec AD dataset from: https://www.mvtec.com/company/research/datasets/mvtec-ad/

2. Organize your project structure as follows:
```
your_project_root/
├── train.py
├── test.py 
├── conversion_test.py
├── options.py
├── network.py
├── utils.py
├── ssim.py
├── mvtec_anomaly_detection/
│   ├── bottle/
│   │   ├── train/good/
│   │   ├── test/good/
│   │   ├── test/broken_large/
│   │   ├── test/broken_small/
│   │   └── test/contamination/
│   │   └── ground_truth/
│   ├── leather/
│   │   ├── train/good/
│   │   ├── test/good/
│   │   ├── test/color/
│   │   ├── test/cut/
│   │   ├── test/fold/
│   │   ├── test/glue/
│   │   └── test/poke/
│   │   └── ground_truth/
│   └── ... (other 13 categories)
└── results/                  # Will be created automatically
    ├── bottle/
    │   └── chechpoints/
    │       └── ssim_loss/
    │           └── model.pth
    ├── leather/
    │   └── chechpoints/
    │       └── ssim_loss/
    │           └── model.pth
    └── ...
```

3. The `DATASET_PATH` in [options.py](options.py#L46) should point to the mvtec_anomaly_detection folder:
```python
DATASET_PATH = './mvtec_anomaly_detection'  # Relative to your script location
```

### Pre-trained ANN Checkpoints

Download pre-trained ANN models from: https://drive.google.com/drive/folders/1JGw8gxNQ-6AZzRCxhrxRCfTBRY-q7QED?usp=sharing

Extract and organize checkpoints as shown in the structure above. Each category should have:
- `model.pth`: ANN trained model

**Note**: The checkpoint folder is named `chechpoints` (with typo) to match the original code.

## Usage

#### Object Categories (patch_size=256, z_dim=500)

**Bottle**
```bash
# Training
python train.py --name bottle --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate 0.
# Testing ANN
python test.py --name bottle --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --bg_mask W
# SNN Conversion
python conversion_test.py --name bottle --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
```

**Cable**
```bash
# Training
python train.py --name cable --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate 0. --p_horizonal_flip 0. --p_vertical_flip 0.
# Testing ANN
python test.py --name cable --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
# SNN Conversion
python conversion_test.py --name cable --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
```

**Capsule**
```bash
# Training
python train.py --name capsule --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate 0. --p_horizonal_flip 0. --p_vertical_flip 0.
# Testing ANN
python test.py --name capsule --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --bg_mask W
# SNN Conversion
python conversion_test.py --name capsule --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
```

**Hazelnut**
```bash
# Training
python train.py --name hazelnut --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate_crop 0.
# Testing ANN
python test.py --name hazelnut --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --bg_mask B 
# SNN Conversion
python conversion_test.py --name hazelnut --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
```

**Metal Nut**
```bash
# Training
python train.py --name metal_nut --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate_crop 0. --p_horizonal_flip 0. --p_vertical_flip 0.
# Testing ANN
python test.py --name metal_nut --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --bg_mask B 
# SNN Conversion
python conversion_test.py --name metal_nut --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
```

**Pill**
```bash
# Training
python train.py --name pill --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate 0. --p_horizonal_flip 0. --p_vertical_flip 0.
# Testing ANN
python test.py --name pill --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --bg_mask B
# SNN Conversion
python conversion_test.py --name pill --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
```

**Screw (Grayscale)**
```bash
# Training
python train.py --name screw --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --grayscale --do_aug --p_rotate 0.
# Testing ANN
python test.py --name screw --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --grayscale --bg_mask W
# SNN Conversion
python conversion_test.py --name screw --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --grayscale
```

**Toothbrush**
```bash
# Training
python train.py --name toothbrush --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate 0. --p_vertical_flip 0.
# Testing ANN
python test.py --name toothbrush --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
# SNN Conversion
python conversion_test.py --name toothbrush --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
```

**Transistor**
```bash
# Training
python train.py --name transistor --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate 0. --p_vertical_flip 0.
# Testing ANN
python test.py --name transistor --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 
# SNN Conversion
python conversion_test.py --name transistor --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500
```

**Zipper (Grayscale)**
```bash
# Training
python train.py --name zipper --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --grayscale --do_aug --p_rotate 0.
# Testing ANN
python test.py --name zipper --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --grayscale 
# SNN Conversion
python conversion_test.py --name zipper --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --grayscale
```

#### Texture Categories (patch_size=128, z_dim=100)

**Carpet**
```bash
# Training
python train.py --name carpet --loss ssim_loss --im_resize 512 --patch_size 128 --z_dim 100 --do_aug --rotate_angle_vari 10
# Testing ANN
python test.py --name carpet --loss ssim_loss --im_resize 512 --patch_size 128 --z_dim 100
# SNN Conversion
python conversion_test.py --name carpet --loss ssim_loss --im_resize 512 --patch_size 128 --z_dim 100
```

**Grid (Grayscale)**
```bash
# Training
python train.py --name grid --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --grayscale --do_aug 
# Testing ANN
python test.py --name grid --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --grayscale
# SNN Conversion
python conversion_test.py --name grid --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --grayscale
```

**Leather**
```bash
# Training
python train.py --name leather --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --do_aug
# Testing ANN
python test.py --name leather --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100
# SNN Conversion
python conversion_test.py --name leather --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100
```

**Tile**
```bash
# Training
python train.py --name tile --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --do_aug
# Testing ANN
python test.py --name tile --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100
# SNN Conversion
python conversion_test.py --name tile --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100
```

**Wood**
```bash
# Training
python train.py --name wood --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --do_aug --rotate_angle_vari 15
# Testing ANN
python test.py --name wood --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 
# SNN Conversion
python conversion_test.py --name wood --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100
```

### Batch Training (Optional)

For training multiple categories automatically:

```bash
# Train all categories with optimal configurations
python train_all.py --epochs 200
```
