# ANN2SNN: Firing Rate-based SNN Conversion for Anomaly Detection

This repository implements a novel ANN-to-SNN conversion approach for energy-efficient image anomaly detection using reconstruction methods. Our method introduces a firing rate-based reconstruction head that enables high-quality continuous value generation from spiking neural networks.

## Overview

Traditional SNN conversion methods struggle with reconstruction tasks due to the mismatch between discrete spikes and continuous outputs required for image reconstruction. Our approach addresses this challenge through:

- **Firing Rate-based Reconstruction**: Novel temporal spike accumulation mechanism for continuous value approximation
- **Energy Efficiency**: 3,900-10,600× energy reduction compared to ANN baselines
- **Direct Conversion**: No retraining required - works with pre-trained ANN models
- **Competitive Performance**: Superior to specialized SNN methods (FSVAE, ESVAE) on 10/15 MVTec categories

## Results

### Performance Comparison: ANN vs SNN Conversion

The following table shows AUC performance across different time steps for SNN conversion compared to the original ANN baseline:

| Class | AUC | ANN ReLU | Time step 1 | Time step 10 | Time step 20 | Time step 30 | Time step 40 | Time step 50 | Time step 60 | Time step 70 | Time step 80 | Time step 90 | Time step 100 |
|-------|--------|----------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|
| **wood** | image | 0.9754 | 0.9702 | 0.9737 | 0.9789 | 0.9781 | 0.9798 | 0.9816 | 0.9807 | 0.9798 | 0.9807 | 0.9798 | 0.9798 |
| | pixel | 0.7140 | 0.7102 | 0.7118 | 0.7137 | 0.7136 | 0.7136 | 0.7138 | 0.7138 | 0.7137 | 0.7137 | 0.7138 | 0.7139 |
| **tile** | image | 0.8315 | 0.8777 | 0.8615 | 0.8442 | 0.8434 | 0.8395 | 0.8355 | 0.8359 | 0.8377 | 0.8369 | 0.8366 | 0.8359 |
| | pixel | 0.4326 | 0.4342 | 0.4323 | 0.4322 | 0.4323 | 0.4323 | 0.4325 | 0.4326 | 0.4324 | 0.4324 | 0.4323 | 0.4323 |
| **leather** | image | 0.8767 | 0.9474 | 0.9474 | 0.9151 | 0.8736 | 0.8702 | 0.8658 | 0.8665 | 0.8672 | 0.8709 | 0.8706 | 0.8736 |
| | pixel | 0.8701 | 0.9094 | 0.9094 | 0.8980 | 0.8808 | 0.8844 | 0.8831 | 0.8828 | 0.8826 | 0.8822 | 0.8816 | 0.8813 |
| **grid** | image | 0.7318 | 0.7310 | 0.7318 | 0.7327 | 0.7327 | 0.7327 | 0.7327 | 0.7327 | 0.7327 | 0.7327 | 0.7327 | 0.7327 |
| | pixel | 0.6452 | 0.6453 | 0.6453 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 |
| **carpet** | image | 0.5666 | 0.8006 | 0.6260 | 0.5831 | 0.5726 | 0.5662 | 0.5610 | 0.5626 | 0.5686 | 0.5626 | 0.5658 | 0.5598 |
| | pixel | 0.5141 | 0.5169 | 0.5148 | 0.5144 | 0.5142 | 0.5142 | 0.5142 | 0.5142 | 0.5142 | 0.5142 | 0.5141 | 0.5141 |
| **zipper** | image | 0.6867 | 0.4716 | 0.4716 | 0.5735 | 0.6205 | 0.6552 | 0.6886 | 0.6943 | 0.6862 | 0.6959 | 0.6914 | 0.6959 |
| | pixel | 0.7982 | 0.7820 | 0.7820 | 0.7830 | 0.7960 | 0.7970 | 0.7708 | 0.7998 | 0.8046 | 0.8083 | 0.8102 | 0.8106 |
| **transistor** | image | 0.6329 | 0.4471 | 0.4471 | 0.4554 | 0.6208 | 0.6508 | 0.6525 | 0.6617 | 0.6429 | 0.6392 | 0.6442 | 0.6425 |
| | pixel | 0.7842 | 0.5811 | 0.5811 | 0.5849 | 0.6961 | 0.7310 | 0.7511 | 0.7621 | 0.7687 | 0.7718 | 0.7747 | 0.7762 |
| **toothbrush** | image | 0.9833 | 0.4694 | 0.4694 | 0.4667 | 0.4000 | 0.4444 | 0.3972 | 0.4083 | 0.4417 | 0.4417 | 0.4333 | 0.3667 |
| | pixel | 0.9726 | 0.2471 | 0.2471 | 0.1975 | 0.2069 | 0.2515 | 0.3133 | 0.3850 | 0.5169 | 0.5920 | 0.6665 | 0.7171 |
| **screw** | image | 0.5436 | 0.6331 | 0.6331 | 0.8354 | 0.6194 | 0.5794 | 0.5589 | 0.5382 | 0.5032 | 0.5024 | 0.4933 | 0.4970 |
| | pixel | 0.9712 | 0.9288 | 0.9288 | 0.9011 | 0.9090 | 0.9287 | 0.9428 | 0.9511 | 0.9581 | 0.9616 | 0.9638 | 0.9656 |
| **pill** | image | 0.4905 | 0.3707 | 0.3707 | 0.3740 | 0.3762 | 0.3647 | 0.4394 | 0.4539 | 0.4798 | 0.4744 | 0.5802 | 0.5038 |
| | pixel | 0.9041 | 0.3723 | 0.3723 | 0.3698 | 0.3410 | 0.3407 | 0.3780 | 0.4401 | 0.5136 | 0.5973 | 0.6677 | 0.7076 |
| **metal_nut** | image | 0.5029 | 0.3309 | 0.3368 | 0.5699 | 0.3866 | 0.4301 | 0.4883 | 0.4863 | 0.5161 | 0.5244 | 0.5108 | 0.5235 |
| | pixel | 0.7764 | 0.2920 | 0.2747 | 0.3357 | 0.5477 | 0.6602 | 0.7087 | 0.7341 | 0.7459 | 0.7529 | 0.7575 | 0.7612 |
| **hazelnut** | image | 0.8886 | 0.8979 | 0.8979 | 0.9543 | 0.9321 | 0.9239 | 0.9129 | 0.9054 | 0.9014 | 0.8996 | 0.8964 | 0.8950 |
| | pixel | 0.9636 | 0.8788 | 0.8788 | 0.9696 | 0.9758 | 0.9754 | 0.9742 | 0.9731 | 0.9721 | 0.9713 | 0.9706 | 0.9700 |
| **capsule** | image | 0.6613 | 0.3554 | 0.3510 | 0.3690 | 0.2932 | 0.6055 | 0.6474 | 0.7012 | 0.6709 | 0.6434 | 0.6350 | 0.6398 |
| | pixel | 0.8761 | 0.6467 | 0.7004 | 0.6609 | 0.6476 | 0.7886 | 0.8582 | 0.8836 | 0.8986 | 0.9070 | 0.9129 | 0.9156 |
| **cable** | image | 0.6265 | 0.5287 | 0.5287 | 0.5013 | 0.4957 | 0.5371 | 0.5560 | 0.5836 | 0.5864 | 0.6091 | 0.6293 | 0.6419 |
| | pixel | 0.7468 | 0.7190 | 0.7190 | 0.7222 | 0.7584 | 0.7407 | 0.7394 | 0.7439 | 0.7482 | 0.7509 | 0.7515 | 0.7529 |
| **bottle** | image | 0.9183 | 0.5698 | 0.5698 | 0.2841 | 0.3865 | 0.4183 | 0.4889 | 0.5984 | 0.8397 | 0.8889 | 0.9206 | 0.9238 |
| | pixel | 0.8121 | 0.4350 | 0.4350 | 0.4121 | 0.5332 | 0.6126 | 0.6549 | 0.6814 | 0.7002 | 0.7157 | 0.7288 | 0.7380 |

### Key Observations

- **Convergence Pattern**: SNN performance generally improves with increasing time steps, approaching ANN baseline performance
- **Category Variability**: Some categories (wood, grid, hazelnut) show stable performance across time steps, while others (bottle, toothbrush, pill) demonstrate significant improvement with longer time steps
- **Pixel vs Image Level**: Pixel-level AUC tends to show more gradual improvement with time steps compared to image-level metrics
- **Energy vs Performance Trade-off**: While longer time steps improve accuracy, they also increase energy consumption. Time steps 50-100 often provide the best balance

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
ANN2SNN/
├── train.py                # ANN training script
├── test.py                 # ANN testing and evaluation
├── conversion_test.py      # ANN-to-SNN conversion and SNN evaluation
├── options.py              # Configuration parameters
├── network.py              # Autoencoder architecture
├── utils.py                # Utility functions
├── ssim.py                 # SSIM loss implementation
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
