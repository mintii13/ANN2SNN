#!/usr/bin/env python
"""
A2AD - Statistical ANN-based Anomaly Detection
===================================================
No training required. Uses activation deviation from normal statistics.
This is the ANN version of S2AD, no conversion to SNN, no timesteps.

Usage:
  python a2ad_validate.py --name bottle --data_path /path/to/mvtec
  python a2ad_validate.py --name leather --layers layer123 --use_membrane --wandb
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from glob import glob
import cv2
from sklearn.metrics import roc_auc_score
import random
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc
from scipy.ndimage import label as connected_components
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import matplotlib.cm as cm
import setproctitle
setproctitle.setproctitle("Minh Tri is running A2AD...")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BackboneEncoder(nn.Module):
    def __init__(self, backbone='resnet18', layers='layer23'):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self._build_backbone(backbone)
        
    def _build_backbone(self, backbone):
        # ========== RESNET & WIDE RESNET ==========
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
            # Cắt lấy các thành phần
            self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.is_resnet = True
            return
        
        # ========== CÁC BACKBONE DẠNG SEQUENTIAL (VGG, AlexNet, MobileNet, DenseNet) ==========
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


# ═══════════════════════════════════════════════════════════════════════════
# DATASETS (giống hệt S2AD)
# ═══════════════════════════════════════════════════════════════════════════

def get_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MVTecDataset(Dataset):
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
            gt_root = os.path.join(root, category, 'ground_truth')
            for subfolder in sorted(os.listdir(test_root)):
                fpath = os.path.join(test_root, subfolder)
                if not os.path.isdir(fpath):
                    continue
                lbl = 0 if subfolder == 'good' else 1
                for f in sorted(glob(os.path.join(fpath, '*'))):
                    self.files.append(f)
                    self.labels.append(lbl)
                    if lbl == 1:
                        fname = os.path.splitext(os.path.basename(f))[0]
                        gt_path = None
                        for ext in ['.png', '.bmp', '.jpg']:
                            p = os.path.join(gt_root, subfolder, fname + '_mask' + ext)
                            if os.path.exists(p):
                                gt_path = p
                                break
                            p = os.path.join(gt_root, subfolder, fname + ext)
                            if os.path.exists(p):
                                gt_path = p
                                break
                        self.gt_paths.append(gt_path)
                    else:
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


class VisADataset(Dataset):
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
            gt_root = os.path.join(base, 'ground_truth')
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
                    fname = os.path.splitext(os.path.basename(f))[0]
                    mask_path = os.path.join(gt_root, 'bad', fname + '_mask.png')
                    if not os.path.exists(mask_path):
                        for ext in ['.png', '.bmp', '.jpg']:
                            alt = os.path.join(gt_root, 'bad', fname + ext)
                            if os.path.exists(alt):
                                mask_path = alt
                                break
                    self.gt_paths.append(mask_path if os.path.exists(mask_path) else None)
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


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

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


def compute_normal_stats(encoder, normal_loader, device, layers='layer23'):
    """Compute mean, std, MAD from normal activations (ANN)."""
    encoder.eval()
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    
    # PASS 1: mean and std
    sum_acts = {name: None for name in layer_names}
    sum_sq_acts = {name: None for name in layer_names}
    count = 0
    
    with torch.no_grad():
        for imgs, _, _ in normal_loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            outputs = encoder(imgs)
            for idx, name in zip(layer_indices, layer_names):
                feat = outputs[idx]  # [B, C, H, W]
                if sum_acts[name] is None:
                    sum_acts[name] = feat.sum(dim=0).cpu()
                    sum_sq_acts[name] = (feat ** 2).sum(dim=0).cpu()
                else:
                    sum_acts[name] += feat.sum(dim=0).cpu()
                    sum_sq_acts[name] += (feat ** 2).sum(dim=0).cpu()
            count += B
    
    means = {}
    stats = {}
    for name in layer_names:
        mean = sum_acts[name] / count
        var = (sum_sq_acts[name] / count) - (mean ** 2)
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var + 1e-8)
        means[name] = mean
        stats[name] = {'mean': mean, 'std': std}
    
    # PASS 2: MAD
    sum_abs_dev = {name: 0.0 for name in layer_names}
    count = 0
    with torch.no_grad():
        for imgs, _, _ in normal_loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            outputs = encoder(imgs)
            for idx, name in zip(layer_indices, layer_names):
                feat = outputs[idx]
                abs_dev = torch.abs(feat - means[name].to(device)).mean().item()
                sum_abs_dev[name] += abs_dev * B
            count += B
    
    for name in layer_names:
        mad = sum_abs_dev[name] / count
        stats[name]['mad'] = mad
        print(f'    {name}: mean={stats[name]["mean"].mean().item():.4f}, '
              f'std={stats[name]["std"].mean().item():.4f}, mad={mad:.6f}')
    
    return stats


def get_activations(encoder, img_tensor, device, layers='layer23'):
    """Extract activations for a single image."""
    encoder.eval()
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    with torch.no_grad():
        outputs = encoder(img_tensor)
    acts = {}
    for idx, name in zip(layer_indices, layer_names):
        acts[name] = outputs[idx]
    return acts


def score_image(encoder, img_tensor, normal_stats, device, layers='layer23',
                img_size=256, combine_method='simple'):
    """Compute anomaly map from activation deviations."""
    encoder.eval()
    img_tensor = img_tensor.to(device)
    acts = get_activations(encoder, img_tensor, device, layers)
    
    deviations = {}
    for layer_name, act in acts.items():
        mean = normal_stats[layer_name]['mean'].to(device)
        std = normal_stats[layer_name]['std'].to(device)
        z_score = (act[0] - mean) / std
        deviation = torch.abs(z_score).mean(dim=0)   # mean over channels
        deviations[layer_name] = deviation
    
    if len(deviations) == 1:
        score_spatial = list(deviations.values())[0]
    else:
        target_name = list(deviations.keys())[0]
        target_res = deviations[target_name].shape
        if combine_method == 'simple':
            combined = torch.zeros_like(deviations[target_name])
            weight_sum = 0
            for layer_name, dev in deviations.items():
                if dev.shape != target_res:
                    dev = F.interpolate(dev.unsqueeze(0).unsqueeze(0),
                                        size=target_res, mode='bilinear',
                                        align_corners=False).squeeze()
                combined += dev
                weight_sum += 1
            score_spatial = combined / weight_sum
        else:  # mad_weighted
            weighted_sum = None
            total_weight = 0.0
            for layer_name, dev in deviations.items():
                if dev.shape != target_res:
                    dev = F.interpolate(dev.unsqueeze(0).unsqueeze(0),
                                        size=target_res, mode='bilinear',
                                        align_corners=False).squeeze()
                mad = normal_stats[layer_name]['mad']
                weight = 1.0 / (mad + 1e-8)
                total_weight += weight
                if weighted_sum is None:
                    weighted_sum = dev * weight
                else:
                    weighted_sum += dev * weight
            score_spatial = weighted_sum / total_weight
    
    score_map = F.interpolate(
        score_spatial.unsqueeze(0).unsqueeze(0).float(),
        size=(img_size, img_size), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    
    return score_map, float(np.max(score_map))


def compute_pro_metric(gt_masks, anomaly_maps, fpr_limit=0.3):
    if not gt_masks or not anomaly_maps:
        return 0.0
    all_amaps = np.array(anomaly_maps)
    all_masks = np.array(gt_masks)
    normal_scores = all_amaps[all_masks == 0]
    total_normal_pixels = len(normal_scores)
    if total_normal_pixels == 0:
        return 0.0
    thresholds = np.linspace(all_amaps.min(), all_amaps.max(), 100)
    normal_scores_sorted = np.sort(normal_scores)
    regions_list = []
    for mask in all_masks:
        labeled, num_regions = connected_components(mask)
        regions_list.append([labeled == reg_id for reg_id in range(1, num_regions + 1)])
    fprs, pros = [], []
    for t in thresholds:
        fp_count = total_normal_pixels - np.searchsorted(normal_scores_sorted, t)
        fpr = fp_count / total_normal_pixels
        fprs.append(fpr)
        overlaps = []
        for img_idx, regions in enumerate(regions_list):
            for region_mask in regions:
                region_scores = all_amaps[img_idx][region_mask]
                if region_scores.size > 0:
                    overlap_ratio = (region_scores >= t).sum() / region_scores.size
                    overlaps.append(overlap_ratio)
        pros.append(np.mean(overlaps) if overlaps else 0.0)
    fprs = np.array(fprs)
    pros = np.array(pros)
    idxes = fprs <= fpr_limit
    fprs_valid = fprs[idxes]
    pros_valid = pros[idxes]
    if len(fprs_valid) < 2:
        return 0.0
    fprs_normalized = (fprs_valid - fprs_valid.min()) / (fprs_valid.max() - fprs_valid.min() + 1e-8)
    pro_auc = auc(fprs_normalized, pros_valid)
    return float(pro_auc)


def evaluate(encoder, test_dataset, normal_stats, device, layers='layer23',
             img_size=256, combine_method='mad_weighted',
             save_maps=False, maps_dir=None, category_name=''):
    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []
    gt_masks, anomaly_maps = [], []
    
    if hasattr(test_dataset, 'files'):
        file_paths = test_dataset.files
    else:
        file_paths = [None] * len(test_dataset)
    
    for i in range(len(test_dataset)):
        img_t, lbl, gt_path = test_dataset[i]
        img_t = img_t.unsqueeze(0)
        score_map, img_score = score_image(encoder, img_t, normal_stats, device,
                                           layers, img_size, combine_method)
        img_scores.append(img_score)
        img_labels.append(lbl)
        
        # Save anomaly maps if requested
        if save_maps and maps_dir:
            subfolder_name = "unknown"
            if file_paths[i]:
                path_parts = os.path.normpath(file_paths[i]).split(os.sep)
                if 'test' in path_parts:
                    test_idx = path_parts.index('test')
                    if test_idx + 1 < len(path_parts):
                        subfolder_name = path_parts[test_idx + 1]
                elif 'bad' in path_parts:
                    bad_idx = path_parts.index('bad')
                    if bad_idx + 1 < len(path_parts):
                        subfolder_name = path_parts[bad_idx + 1]
                    else:
                        subfolder_name = 'bad'
                else:
                    subfolder_name = 'good' if lbl == 0 else 'abnormal'
            else:
                subfolder_name = 'good' if lbl == 0 else 'abnormal'
            save_dir = os.path.join(maps_dir, subfolder_name)
            if file_paths[i] and os.path.exists(file_paths[i]):
                orig_img = cv2.imread(file_paths[i])
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                orig_img = cv2.resize(orig_img, (img_size, img_size))
            else:
                mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
                std = torch.tensor(IMAGENET_STD).view(3,1,1)
                orig_img = img_t[0].cpu() * std + mean
                orig_img = orig_img.clamp(0,1).permute(1,2,0).numpy()
                orig_img = (orig_img * 255).astype(np.uint8)
            gt_mask = None
            if lbl == 1 and gt_path and os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, 0)
                if gt_mask is not None:
                    gt_mask = cv2.resize(gt_mask, (img_size, img_size))
                    gt_mask = (gt_mask > 127).astype(np.uint8) * 255
            save_anomaly_map(orig_img, score_map, gt_mask, save_dir, i)
        
        if lbl == 1 and gt_path:
            gt = cv2.imread(gt_path, 0)
            if gt is not None:
                gt = cv2.resize(gt, (img_size, img_size))
                gt_bin = (gt > 127).astype(int)
                pix_scores.extend(score_map.flatten())
                pix_labels.extend(gt_bin.flatten())
                gt_masks.append(gt_bin)
                anomaly_maps.append(score_map)
    
    # Metrics
    img_auc = roc_auc_score(img_labels, img_scores) if len(set(img_labels)) == 2 else None
    img_ap = average_precision_score(img_labels, img_scores) if len(set(img_labels)) == 2 else None
    prec, rec, _ = precision_recall_curve(img_labels, img_scores)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    img_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
    pix_auc = roc_auc_score(pix_labels, pix_scores) if pix_labels else None
    pix_ap = average_precision_score(pix_labels, pix_scores) if pix_labels else None
    pprec, prec_rec, _ = precision_recall_curve(pix_labels, pix_scores)
    pf1_scores = 2 * (pprec * prec_rec) / (pprec + prec_rec + 1e-8)
    pix_f1 = np.max(pf1_scores) if len(pf1_scores) > 0 else 0.0
    pro_score = compute_pro_metric(gt_masks, anomaly_maps) if gt_masks else 0.0
    
    metrics = {
        'img_auc': img_auc or 0.0,
        'img_ap': img_ap or 0.0,
        'img_f1': img_f1,
        'pix_auc': pix_auc or 0.0,
        'pix_ap': pix_ap or 0.0,
        'pix_f1': pix_f1,
        'pro': pro_score,
    }
    return metrics, img_scores, img_labels


def save_anomaly_map(original_img, score_map, gt_mask, save_dir, idx):
    img_pil = Image.fromarray(original_img)
    smin, smax = score_map.min(), score_map.max()
    if smax > smin:
        score_norm = (score_map - smin) / (smax - smin)
    else:
        score_norm = score_map
    cmap = cm.jet(score_norm)
    anomaly_colored = (cmap[:, :, :3] * 255).astype(np.uint8)
    anomaly_pil = Image.fromarray(anomaly_colored)
    os.makedirs(save_dir, exist_ok=True)
    img_pil.save(os.path.join(save_dir, f'{idx:04d}_img.png'))
    blended = Image.blend(img_pil, anomaly_pil, alpha=0.4)
    blended.save(os.path.join(save_dir, f'{idx:04d}_blend.png'))
    if gt_mask is not None:
        gt_pil = Image.fromarray(gt_mask).convert('RGB')
        gt_pil.save(os.path.join(save_dir, f'{idx:04d}_mask.png'))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='A2AD - Statistical ANN-based Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'],
                        help='Dataset to use')
    parser.add_argument('--name', type=str, required=True, help='Category name')
    parser.add_argument('--data_path', type=str, default='/home/minhtringuyen/ANN2SNN/datasets')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--calib_samples', type=int, default=500)
    parser.add_argument('--layers', type=str, default='layer23',
                        choices=['layer1', 'layer2', 'layer3', 'layer12', 'layer23', 'layer123'])
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2',
                                 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet',
                                 'mobilenet_v2', 'mobilenet_v3_large', 'densenet121', 'densenet169'],
                        help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='./a2ad_results')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='A2AD')
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--combine_method', type=str, default='mad_weighted',
                        choices=['simple', 'mad_weighted'])
    parser.add_argument('--save_anomaly_maps', action='store_true')
    parser.add_argument('--maps_root', type=str, default='./anomaly_maps_a2ad')
    return parser.parse_args()


def main():
    seed_everything(42)
    g = torch.Generator()
    g.manual_seed(42)
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('=' * 60)
    print('A2AD - Statistical ANN-based Anomaly Detection (no SNN, no timesteps)')
    print(f'  Category: {args.name}')
    print(f'  Device: {device}')
    print(f'  Layers: {args.layers}')
    print(f'  Calibration samples: {args.calib_samples}')
    print('=' * 60)
    
    # WandB init
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        if args.wandb_key:
            os.environ['WANDB_API_KEY'] = args.wandb_key
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name if args.wandb_run_name else f'{args.name}_{args.layers}',
                config=vars(args),
                mode='offline' if args.wandb_offline else 'online'
            )
            print(f"  WandB logging enabled: {wandb_run.project}/{wandb_run.name}")
        except Exception as e:
            print(f'WandB init failed: {e}')
    
    # Build encoder
    print(f'\n[1/3] Building encoder (backbone={args.backbone})...')
    encoder = BackboneEncoder(backbone=args.backbone, layers=args.layers).to(device)
    encoder.eval()
    
    dummy = torch.randn(2, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        outputs = encoder(dummy)
        layer_indices, layer_names = get_layer_indices_and_names(args.layers)
        for idx, name in zip(layer_indices, layer_names):
            print(f'  {name} max activation: {outputs[idx].max().item():.3f}')
    
    # Load datasets
    dataset_class = get_dataset_class(args.dataset)
    data_root = os.path.join(args.data_path, args.dataset)
    print('\n[2/3] Loading normal dataset...')
    full_normal_ds = dataset_class(data_root, args.name, 'train', img_size=args.img_size)
    print(f'  Full normal set: {len(full_normal_ds)} images')
    
    if args.calib_samples > 0 and args.calib_samples < len(full_normal_ds):
        subset_indices = list(range(args.calib_samples))
        normal_subset = Subset(full_normal_ds, subset_indices)
        calib_loader = DataLoader(normal_subset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=2, generator=g, worker_init_fn=lambda wid: np.random.seed(42+wid))
        normal_loader = DataLoader(normal_subset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=2, generator=g, worker_init_fn=lambda wid: np.random.seed(42+wid))
        print(f'  Using {args.calib_samples} samples for both calibration and normal statistics')
    else:
        calib_loader = DataLoader(full_normal_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=2, generator=g, worker_init_fn=lambda wid: np.random.seed(42+wid))
        normal_loader = DataLoader(full_normal_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=2, generator=g, worker_init_fn=lambda wid: np.random.seed(42+wid))
        print('  Using full normal set for both calibration and normal statistics')
    
    # Compute normal statistics
    print('\n[3/3] Computing normal statistics and evaluating...')
    normal_stats = compute_normal_stats(encoder, normal_loader, device, args.layers)
    
    # Load test dataset
    test_ds = dataset_class(data_root, args.name, 'test', img_size=args.img_size)
    n_anomaly = sum(test_ds.labels)
    n_normal = len(test_ds.labels) - n_anomaly
    print(f'  Test set: {len(test_ds)} images ({n_anomaly} anomaly, {n_normal} normal)')
    
    # Evaluate (single forward, no timesteps)
    if args.save_anomaly_maps:
        config_str = f"{args.backbone}_{args.combine_method}{args.layers}"
        maps_dir = os.path.join(args.maps_root, config_str, args.name)
    else:
        maps_dir = None
    
    metrics, img_scores, img_labels = evaluate(
        encoder, test_ds, normal_stats, device, args.layers,
        args.img_size, args.combine_method,
        save_maps=args.save_anomaly_maps, maps_dir=maps_dir, category_name=args.name
    )
    
    # Print results
    print('\n' + '=' * 90)
    print('RESULTS:')
    print(f'  Image AUC: {metrics["img_auc"]:.4f}')
    print(f'  Image AP:  {metrics["img_ap"]:.4f}')
    print(f'  Image F1:  {metrics["img_f1"]:.4f}')
    print(f'  Pixel AUC: {metrics["pix_auc"]:.4f}')
    print(f'  Pixel AP:  {metrics["pix_ap"]:.4f}')
    print(f'  Pixel F1:  {metrics["pix_f1"]:.4f}')
    print(f'  PRO:       {metrics["pro"]:.4f}')
    print('=' * 90)
    
    # Log to WandB
    if wandb_run:
        wandb_run.config.update({
            'category': args.name,
            'layers': args.layers,
            'calib_samples': args.calib_samples,
            'img_size': args.img_size,
            'combine_method': args.combine_method,
        })
        wandb_run.summary.update(metrics)
        wandb_run.finish()
    
    # Save results to file
    out_path = os.path.join(args.save_dir, f'{args.name}_results.txt')
    with open(out_path, 'w') as f:
        f.write(f"A2AD Results - {args.name}\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Layers: {args.layers}\n")
        f.write(f"Calibration samples: {args.calib_samples}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Combine method: {args.combine_method}\n")
        f.write(f"\nImage AUC: {metrics['img_auc']:.4f}\n")
        f.write(f"Image AP:  {metrics['img_ap']:.4f}\n")
        f.write(f"Image F1:  {metrics['img_f1']:.4f}\n")
        f.write(f"Pixel AUC: {metrics['pix_auc']:.4f}\n")
        f.write(f"Pixel AP:  {metrics['pix_ap']:.4f}\n")
        f.write(f"Pixel F1:  {metrics['pix_f1']:.4f}\n")
        f.write(f"PRO:       {metrics['pro']:.4f}\n")
    print(f'\nResults saved: {out_path}')


if __name__ == '__main__':
    main()