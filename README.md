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
│   ├── leather/
│   │   ├── train/good/
│   │   ├── test/good/
│   │   ├── test/color/
│   │   ├── test/cut/
│   │   ├── test/fold/
│   │   ├── test/glue/
│   │   └── test/poke/
│   └── ... (other 13 categories)
├── ground_truth/             # Optional: for pixel-level evaluation
│   ├── bottle/
│   ├── leather/
│   └── ...
└── results/                  # Will be created automatically
    ├── bottle/
    │   └── chechpoints/
    │       └── ssim_loss/
    │           ├── model.pth
    │           └── best_model.pth
    ├── leather/
    │   └── chechpoints/
    │       └── ssim_loss/
    │           ├── model.pth
    │           └── best_model.pth
    └── ...
```

3. The `DATASET_PATH` in [options.py](options.py#L46) should point to the mvtec_anomaly_detection folder:
```python
DATASET_PATH = './mvtec_anomaly_detection'  # Relative to your script location
```

### Pre-trained ANN Checkpoints

Download pre-trained ANN models from: https://drive.google.com/drive/folders/1JGw8gxNQ-6AZzRCxhrxRCfTBRY-q7QED?usp=sharing

Extract and organize checkpoints as shown in the structure above. Each category should have:
- `model.pth`: Latest trained model
- `best_model.pth`: Best performing model (optional)

**Note**: The checkpoint folder is named `chechpoints` (with typo) to match the original code structure./datasets/mvtec-ad/

3. Set the `DATASET_PATH` in [options.py](options.py#L46):
```python
DATASET_PATH = 'your_project_root/mvtec_anomaly_detection'
```

### Pre-trained ANN Checkpoints

Download pre-trained ANN models from: https://drive.google.com/drive/folders/1JGw8gxNQ-6AZzRCxhrxRCfTBRY-q7QED?usp=sharing

Organize checkpoints in your project:
```
your_project_root/
├── results/
│   ├── bottle/chechpoints/ssim_loss/model.pth
│   ├── leather/chechpoints/ssim_loss/model.pth
│   ├── wood/chechpoints/ssim_loss/model.pth
│   ├── tile/chechpoints/ssim_loss/model.pth
│   ├── hazelnut/chechpoints/ssim_loss/model.pth
│   └── ... (other categories)
```

**Note**: The checkpoint folder is named `chechpoints` (with typo) to match the original code.

## Usage

### Step 1: Train ANN Baseline

First train the ANN autoencoder using SSIM loss:

**Texture Categories (patch_size=128, z_dim=100):**
```bash
# Leather
python train.py --name leather --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --do_aug

# Wood  
python train.py --name wood --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --do_aug --rotate_angle_vari 15

# Tile
python train.py --name tile --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --do_aug

# Carpet
python train.py --name carpet --loss ssim_loss --im_resize 512 --patch_size 128 --z_dim 100 --do_aug --rotate_angle_vari 10

# Grid (grayscale)
python train.py --name grid --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100 --grayscale --do_aug
```

**Object Categories (patch_size=256, z_dim=500):**
```bash
# Bottle
python train.py --name bottle --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate 0.

# Hazelnut
python train.py --name hazelnut --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate_crop 0.

# Toothbrush
python train.py --name toothbrush --loss ssim_loss --im_resize 266 --patch_size 256 --z_dim 500 --do_aug --p_rotate 0. --p_vertical_flip 0.
```

### Step 2: Test ANN Baseline

```bash
# Test ANN performance
python test.py --name leather --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100
```

### Step 3: Convert to SNN and Evaluate

```bash
# Convert ANN to SNN and evaluate across different timesteps
python conversion_test.py --name leather --loss ssim_loss --im_resize 256 --patch_size 128 --z_dim 100
```

This will:
- Convert the trained ANN to SNN using SpikingJelly
- Evaluate performance across timesteps T ∈ {1,10,20,...,100}
- Generate energy consumption analysis
- Save reconstruction visualizations

## Key Features

### Firing Rate-based Reconstruction

Our core contribution addresses the spike-to-continuous conversion challenge:

```python
# Temporal spike accumulation
y_rate = (1/T) * sum(spikes[t] for t in range(T))

# Sigmoid mapping to pixel intensities  
reconstruction = sigmoid(y_rate)
```

### Energy Efficiency Analysis

The framework automatically calculates energy consumption using neuromorphic hardware baselines:
- **SNN**: 77 fJ per synaptic operation (SOP)
- **ANN**: 12.5 pJ per floating-point operation (FLOP)

### Adaptive Timestep Selection

- **Standard**: T=100 timesteps for most categories
- **Complex objects**: T=300 for fine-grained details (e.g., Toothbrush)
- **Trade-off**: Higher T improves accuracy but increases latency

## Results

Our approach achieves competitive anomaly detection performance while providing substantial energy savings:

| Category | ANN (Image/Pixel) | ANN2SNN (Image/Pixel) | Energy Reduction |
|----------|-------------------|----------------------|------------------|
| Wood     | 0.975/0.714      | 0.980/0.714          | 7,240×           |
| Leather  | 0.877/0.870      | 0.874/0.881          | 10,588×          |
| Bottle   | 0.918/0.812      | 0.925/0.738          | 12,376×          |

## File Structure

```
ANN2SNN/
├── train.py              # ANN training script
├── test.py               # ANN testing and evaluation
├── conversion_test.py    # ANN-to-SNN conversion and SNN evaluation
├── network.py            # Autoencoder architecture
├── options.py            # Configuration parameters
├── utils.py              # Utility functions
├── ssim.py               # SSIM loss implementation
└── debug_snn_conversion.py # Debugging tools
```

## Debugging and Analysis

Use the debugging tools to analyze conversion quality:

```bash
python debug_snn_conversion.py --name leather
```

This provides:
- SNN conversion verification
- Temporal behavior analysis
- Voltage scaler diagnostics
- Performance comparison insights
