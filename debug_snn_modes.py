#!/usr/bin/env python
"""
debug_snn_modes.py - Debug SNN thresholds and firing rates for multiple snn_modes and timesteps.

Usage:
    python debug_snn_modes.py --dataset mvtec --name bottle --modes 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 --timesteps 4 8 16 32 64
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from glob import glob
import cv2
from spikingjelly.activation_based import ann2snn, functional
import random
from PIL import Image

# ------------------------------
# Copy các class và hàm cần thiết từ s2ad_validate.py (giữ nguyên)
# ------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class BackboneEncoder(nn.Module):
    def __init__(self, backbone='resnet18', layers='layer23'):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self._build_backbone(backbone)
        
    def _build_backbone(self, backbone):
        if backbone in ['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2']:
            if backbone == 'resnet18':
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self.feat_channels = {'layer1': 64, 'layer2': 128, 'layer3': 256}
            elif backbone == 'resnet34':
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                self.feat_channels = {'layer1': 64, 'layer2': 128, 'layer3': 256}
            elif backbone == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                self.feat_channels = {'layer1': 256, 'layer2': 512, 'layer3': 1024}
            elif backbone == 'wide_resnet50_2':
                model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
                self.feat_channels = {'layer1': 256, 'layer2': 512, 'layer3': 1024}
            elif backbone == 'wide_resnet101_2':
                model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1)
                self.feat_channels = {'layer1': 256, 'layer2': 512, 'layer3': 1024}
            self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.is_resnet = True
            return
        
        self.is_resnet = False
        if backbone.startswith('vgg'):
            variants = {
                'vgg11': (models.vgg11, 8, 15, 22),
                'vgg13': (models.vgg13, 6, 11, 16),
                'vgg16': (models.vgg16, 8, 15, 22),
                'vgg19': (models.vgg19, 9, 16, 25)
            }
            creator, idx1, idx2, idx3 = variants[backbone]
            model = creator(weights='IMAGENET1K_V1').features
            self.feat_channels = {'layer1': 256, 'layer2': 512, 'layer3': 512}
            self.output_indices = [idx1, idx2, idx3]
        elif backbone == 'alexnet':
            model = models.alexnet(weights='IMAGENET1K_V1').features
            self.feat_channels = {'layer1': 192, 'layer2': 256, 'layer3': 256}
            self.output_indices = [4, 7, 9]
        elif backbone == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            self.feat_channels = {'layer1': 32, 'layer2': 96, 'layer3': 320}
            self.output_indices = [3, 10, 17]
        elif backbone == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1).features
            self.feat_channels = {'layer1': 40, 'layer2': 112, 'layer3': 160}
            self.output_indices = [3, 8, 12]
        elif backbone in ['densenet121', 'densenet169']:
            if backbone == 'densenet121':
                model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
            else:
                model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1).features
            self.feat_channels = {'layer1': 256, 'layer2': 512, 'layer3': 1024}
            self.output_indices = [4, 6, 8]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.features = model
        self.output_indices = sorted(self.output_indices)
    
    def forward(self, x):
        if self.is_resnet:
            x = self.stem(x)
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            outputs = [f1, f2, f3]
        else:
            outputs = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.output_indices:
                    outputs.append(x)
            while len(outputs) < 3:
                outputs.append(x)
            if len(outputs) > 3:
                outputs = outputs[:3]
        if self.layers == 'layer1':
            return (outputs[0],)
        elif self.layers == 'layer2':
            return (outputs[1],)
        elif self.layers == 'layer3':
            return (outputs[2],)
        elif self.layers == 'layer12':
            return (outputs[0], outputs[1])
        elif self.layers == 'layer23':
            return (outputs[1], outputs[2])
        elif self.layers == 'layer123':
            return tuple(outputs)
        else:
            return (outputs[1], outputs[2])

def get_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, split='train', img_size=256, max_samples=None):
        self.transform = get_transform(img_size)
        self.img_size = img_size
        if split == 'train':
            pattern = os.path.join(root, category, 'train', 'good', '*')
            self.files = sorted(glob(pattern))
            self.labels = [0] * len(self.files)
            self.gt_paths = [None] * len(self.files)
        else:
            self.files, self.labels, self.gt_paths = [], [], []
            test_root = os.path.join(root, category, 'test')
            for subfolder in sorted(os.listdir(test_root)):
                fpath = os.path.join(test_root, subfolder)
                if not os.path.isdir(fpath):
                    continue
                lbl = 0 if subfolder == 'good' else 1
                for f in sorted(glob(os.path.join(fpath, '*'))):
                    self.files.append(f)
                    self.labels.append(lbl)
                    self.gt_paths.append(None)
        if max_samples and max_samples < len(self.files):
            self.files = self.files[:max_samples]
            self.labels = self.labels[:max_samples]
            self.gt_paths = self.gt_paths[:max_samples]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), self.labels[idx], self.gt_paths[idx] or ''

class VisADataset(torch.utils.data.Dataset):
    def __init__(self, root, category, split='train', img_size=256, max_samples=None):
        self.transform = get_transform(img_size)
        self.img_size = img_size
        base = os.path.join(root, 'visa_pytorch', category)
        if split == 'train':
            pattern = os.path.join(base, 'train', 'good', '*')
            self.files = sorted(glob(pattern))
            self.labels = [0] * len(self.files)
            self.gt_paths = [None] * len(self.files)
        else:
            self.files, self.labels, self.gt_paths = [], [], []
            test_root = os.path.join(base, 'test')
            good_dir = os.path.join(test_root, 'good')
            if os.path.exists(good_dir):
                for f in sorted(glob(os.path.join(good_dir, '*'))):
                    self.files.append(f)
                    self.labels.append(0)
                    self.gt_paths.append(None)
            bad_dir = os.path.join(test_root, 'bad')
            if os.path.exists(bad_dir):
                for f in sorted(glob(os.path.join(bad_dir, '*'))):
                    self.files.append(f)
                    self.labels.append(1)
                    self.gt_paths.append(None)
        if max_samples and max_samples < len(self.files):
            self.files = self.files[:max_samples]
            self.labels = self.labels[:max_samples]
            self.gt_paths = self.gt_paths[:max_samples]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), self.labels[idx], self.gt_paths[idx] or ''

def get_dataset_class(dataset_name):
    if dataset_name == 'mvtec':
        return MVTecDataset
    elif dataset_name == 'visa':
        return VisADataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_layer_indices_and_names(layers):
    mapping = {
        'layer1': ([0], ['layer1']),
        'layer2': ([0], ['layer2']),
        'layer3': ([0], ['layer3']),
        'layer12': ([0, 1], ['layer1', 'layer2']),
        'layer23': ([0, 1], ['layer2', 'layer3']),
        'layer123': ([0, 1, 2], ['layer1', 'layer2', 'layer3']),
    }
    return mapping.get(layers, ([0, 1], ['layer2', 'layer3']))

def build_snn_encoder(ann_encoder, calib_loader, device, mode='max'):
    ann_encoder.eval()
    class AdapterLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                yield batch[0], batch[1]
        def __len__(self):
            return len(self.loader)
    adapter = AdapterLoader(calib_loader)
    if mode == 'max':
        converter_mode = 'max'
    else:
        try:
            converter_mode = float(mode)
        except ValueError:
            converter_mode = mode
    converter = ann2snn.Converter(
        dataloader=adapter,
        device=device,
        mode=converter_mode,
        momentum=0.1
    )
    snn_encoder = converter(ann_encoder)
    return snn_encoder

def get_firing_rates(snn_encoder, img_tensor, device, timesteps, layers='layer23'):
    functional.reset_net(snn_encoder)
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    spike_acc = {name: None for name in layer_names}
    with torch.no_grad():
        for t in range(timesteps):
            outputs = snn_encoder(img_tensor)
            for idx, name in zip(layer_indices, layer_names):
                feat = outputs[idx]
                spike_acc[name] = feat if t == 0 else spike_acc[name] + feat
    rates = {}
    for name in layer_names:
        rates[name] = spike_acc[name] / timesteps
    return rates

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------
# Hàm debug chính
# ------------------------------
def debug_snn(snn_encoder, normal_loader, device, timestep, layers='layer23'):
    """In firing rate trung bình trên một batch normal."""
    print(f"\n[DEBUG] Firing rate with timestep={timestep}:")
    sample_imgs, _, _ = next(iter(normal_loader))
    sample_imgs = sample_imgs.to(device)
    rates = get_firing_rates(snn_encoder, sample_imgs, device, timestep, layers)
    for name, rate in rates.items():
        print(f"  Layer {name}: mean={rate.mean().item():.6f}, std={rate.std().item():.6f}, max={rate.max().item():.6f}")

def main():
    parser = argparse.ArgumentParser(description='Debug SNN thresholds and firing rates')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--name', type=str, required=True, help='Category name')
    parser.add_argument('--data_path', type=str, default='/home/minhtringuyen/ANN2SNN/datasets')
    parser.add_argument('--backbone', type=str, default='vgg16')
    parser.add_argument('--layers', type=str, default='layer123')
    parser.add_argument('--calib_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--timesteps', type=int, nargs='+', default=[4,8,16,32,64], help='List of timesteps')
    parser.add_argument('--modes', type=float, nargs='+', required=True,
                        help='snn_mode values to test (e.g., 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2)')
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}, Category: {args.name}")
    print(f"Backbone: {args.backbone}, Layers: {args.layers}")
    print(f"Timesteps: {args.timesteps}")
    print(f"snn_modes: {args.modes}")

    # Build ANN encoder
    ann_encoder = BackboneEncoder(backbone=args.backbone, layers=args.layers).to(device)
    ann_encoder.eval()

    # Load normal dataset
    dataset_class = get_dataset_class(args.dataset)
    data_root = os.path.join(args.data_path, args.dataset)
    full_normal_ds = dataset_class(data_root, args.name, 'train', img_size=args.img_size)
    print(f"Normal images: {len(full_normal_ds)}")

    if args.calib_samples > 0 and args.calib_samples < len(full_normal_ds):
        subset_indices = list(range(args.calib_samples))
        normal_subset = Subset(full_normal_ds, subset_indices)
        calib_loader = DataLoader(normal_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        normal_loader = DataLoader(normal_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"Using {args.calib_samples} samples for calibration and normal stats")
    else:
        calib_loader = DataLoader(full_normal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        normal_loader = DataLoader(full_normal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print("Using full normal set")

    # Lưu kết quả vào file debug.txt
    output_file = f"debug_{args.dataset}_{args.name}_{args.backbone}_{args.layers}.txt"
    with open(output_file, 'w') as f:
        f.write(f"SNN Debug: Dataset={args.dataset}, Category={args.name}, Backbone={args.backbone}, Layers={args.layers}\n")
        f.write(f"Timesteps: {args.timesteps}\n")
        f.write(f"snn_modes: {args.modes}\n\n")

        for snn_mode in args.modes:
            print(f"\n======= snn_mode = {snn_mode} =======")
            f.write(f"\n======= snn_mode = {snn_mode} =======\n")
            # Chuyển đổi SNN
            snn_encoder = build_snn_encoder(ann_encoder, calib_loader, device, mode=str(snn_mode))
            snn_encoder.eval()

            # In ngưỡng các neuron IF
            f.write("Thresholds (v_threshold):\n")
            print("Thresholds (v_threshold):")
            for name, module in snn_encoder.named_modules():
                if hasattr(module, 'v_threshold'):
                    msg = f"  {name}: v_threshold = {module.v_threshold:.6f}"
                    print(msg)
                    f.write(msg + "\n")
            f.write("\n")

            # Với mỗi timestep, tính firing rate trên một batch normal
            for T in args.timesteps:
                f.write(f"Timestep = {T}:\n")
                sample_imgs, _, _ = next(iter(normal_loader))
                sample_imgs = sample_imgs.to(device)
                rates = get_firing_rates(snn_encoder, sample_imgs, device, T, args.layers)
                for name, rate in rates.items():
                    msg = f"  Layer {name}: mean={rate.mean().item():.6f}, std={rate.std().item():.6f}, max={rate.max().item():.6f}"
                    print(msg)
                    f.write(msg + "\n")
                f.write("\n")
            f.write("\n" + "="*50 + "\n")
            # Giải phóng bộ nhớ
            del snn_encoder
            torch.cuda.empty_cache()

    print(f"\nAll debug information saved to {output_file}")

if __name__ == '__main__':
    main()