#!/usr/bin/env python
"""
energy_validate.py - Tính năng lượng cho A2AD (ANN) và S2AD (SNN) trên MVTec/VisA
Dựa trên:
- ANN: FLOPs, năng lượng = FLOPs * 12.5 pJ (12.5e-6 µJ)
- SNN: tổng số spike (mỗi spike = 1 SynOp), năng lượng = spikes * 77 fJ (77e-6 µJ)
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from thop import profile, clever_format

# Import các class và hàm từ s2ad_validate.py
from s2ad_validate import (
    BackboneEncoder, get_transform, MVTecDataset, VisADataset,
    get_dataset_class, build_snn_encoder, get_layer_indices_and_names,
    functional, seed_everything
)

# Các hằng số năng lượng
FLOP_ENERGY_pJ = 12.5          # pJ per FLOP (ước lượng cho GPU 32-bit)
FLOP_ENERGY_uJ = FLOP_ENERGY_pJ / 1e6   # µJ per FLOP
SPIKE_ENERGY_fJ = 77           # fJ per spike (ROLLS neuromorphic processor)
SPIKE_ENERGY_uJ = SPIKE_ENERGY_fJ / 1e9 # µJ per spike

# Cấu hình mặc định
DEFAULT_DATA_ROOT = '/home/minhtringuyen/ANN2SNN/datasets'
DEFAULT_BACKBONE = 'vgg16'
DEFAULT_LAYERS = 'layer123'
DEFAULT_CALIB_SAMPLES = -1   # dùng toàn bộ train set
DEFAULT_SNN_MODE = 'max'
DEFAULT_TIMESTEPS = [4, 8, 16, 32, 64]

# ============================================================
# Hàm tính năng lượng cho ANN (A2AD)
# ============================================================
def compute_ann_energy(model, sample_input, device):
    """Tính FLOPs và năng lượng (µJ) cho ANN model."""
    model.eval()
    flops, _ = profile(model, inputs=(sample_input.to(device),), verbose=False)
    energy_uJ = flops * FLOP_ENERGY_uJ
    return flops, energy_uJ

# ============================================================
# Hàm tính năng lượng cho SNN (S2AD)
# ============================================================
def count_spikes_snn(snn_encoder, test_loader, device, timesteps, layers='layer23'):
    """
    Đếm tổng số spike (SynOps) trên toàn bộ test set.
    Trả về: avg_spikes_per_sample, total_spikes, total_samples
    """
    snn_encoder.eval()
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    total_spikes = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, _, _ in tqdm(test_loader, desc=f"Timestep {timesteps}"):
            imgs = imgs.to(device)
            B = imgs.shape[0]
            functional.reset_net(snn_encoder)
            for t in range(timesteps):
                outputs = snn_encoder(imgs)
                for idx, name in zip(layer_indices, layer_names):
                    feat = outputs[idx]
                    spike = (feat > 0).float()   # spike nhị phân (0/1)
                    total_spikes += spike.sum().item()
            total_samples += B

    avg_spikes = total_spikes / total_samples
    return avg_spikes, total_spikes, total_samples

def compute_snn_energy(snn_encoder, test_loader, device, timesteps, layers='layer23'):
    """Tính năng lượng (µJ) cho SNN dựa trên spike count."""
    avg_spikes, _, _ = count_spikes_snn(snn_encoder, test_loader, device, timesteps, layers)
    energy_uJ = avg_spikes * SPIKE_ENERGY_uJ
    return avg_spikes, energy_uJ

# ============================================================
# Main: chạy cho tất cả các category
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Tính năng lượng cho A2AD và S2AD')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--categories', type=str, nargs='+', default=None, help='Danh sách category (mặc định tất cả)')
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE, help='Backbone architecture')
    parser.add_argument('--layers', type=str, default=DEFAULT_LAYERS, help='Layer selection')
    parser.add_argument('--calib_samples', type=int, default=DEFAULT_CALIB_SAMPLES, help='Số mẫu calibration ( -1 = toàn bộ)')
    parser.add_argument('--snn_mode', type=str, default=DEFAULT_SNN_MODE, help='Chế độ chuyển đổi SNN')
    parser.add_argument('--timesteps', type=int, nargs='+', default=DEFAULT_TIMESTEPS, help='Các timestep cần đánh giá')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT, help='Root chứa dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device)

    # Lấy danh sách categories
    if args.categories is None:
        if args.dataset == 'mvtec':
            categories = [
                'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
            ]
        else:
            categories = [
                'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
            ]
    else:
        categories = args.categories

    # Bảng kết quả
    results = []

    for category in categories:
        print(f"\n{'='*70}")
        print(f"Processing category: {category}")
        print(f"{'='*70}")

        # Load test dataset
        dataset_class = get_dataset_class(args.dataset)
        data_root = os.path.join(args.data_root, args.dataset)
        test_ds = dataset_class(data_root, category, split='test', img_size=args.img_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # Xây dựng ANN encoder
        ann_encoder = BackboneEncoder(backbone=args.backbone, layers=args.layers).to(device)
        ann_encoder.eval()

        # Lấy một sample input để tính FLOPs
        sample_input, _, _ = test_ds[0]
        sample_input = sample_input.unsqueeze(0).to(device)

        # Tính FLOPs và năng lượng ANN
        flops, ann_energy_uJ = compute_ann_energy(ann_encoder, sample_input, device)
        print(f"  ANN (A2AD): FLOPs = {flops/1e9:.2f} G, Energy = {ann_energy_uJ:.2f} µJ")

        # Chuẩn bị calibration loader cho SNN conversion (dùng toàn bộ train set)
        train_ds = dataset_class(data_root, category, split='train', img_size=args.img_size)
        if args.calib_samples > 0 and args.calib_samples < len(train_ds):
            calib_subset = Subset(train_ds, list(range(args.calib_samples)))
            calib_loader = DataLoader(calib_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        else:
            calib_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # Convert ANN -> SNN (chỉ một lần, dùng mode=args.snn_mode)
        print("  Converting ANN to SNN...")
        snn_encoder = build_snn_encoder(ann_encoder, calib_loader, device, mode=args.snn_mode)
        snn_encoder.eval()

        # Tính năng lượng cho từng timestep
        for T in args.timesteps:
            avg_spikes, energy_uJ = compute_snn_energy(snn_encoder, test_loader, device, T, args.layers)
            print(f"  S2AD T={T}: avg spikes/sample = {avg_spikes:.2f}, Energy = {energy_uJ:.5f} µJ")
            results.append({
                'dataset': args.dataset,
                'category': category,
                'backbone': args.backbone,
                'layers': args.layers,
                'snn_mode': args.snn_mode,
                'timestep': T,
                'ann_energy_uJ': ann_energy_uJ,
                'snn_energy_uJ': energy_uJ,
                'saving_factor': ann_energy_uJ / energy_uJ if energy_uJ > 0 else float('inf')
            })

    # In bảng tổng hợp
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Dataset':<8} {'Category':<15} {'Timestep':<8} {'ANN Energy (µJ)':<18} {'SNN Energy (µJ)':<18} {'Saving Factor':<15}")
    for r in results:
        print(f"{r['dataset']:<8} {r['category']:<15} {r['timestep']:<8} {r['ann_energy_uJ']:<18.2f} {r['snn_energy_uJ']:<18.6f} {r['saving_factor']:<15.0f}")

if __name__ == '__main__':
    main()