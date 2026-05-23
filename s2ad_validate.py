"""
S2AD - Statistical SNN-based Anomaly Detection
===================================================
No training required. Uses firing rate deviation from normal statistics.

Key features:
  - Pure spiking neural network (fully neuromorphic compatible)
  - Statistical anomaly detection (no training, only statistics from normal data)
  - Configurable feature layers (layer1, layer2, layer3, or combinations)
  - Multi-timestep evaluation and comparison
  - Membrane potential integration option
  - WandB logging support

Usage:
  python s2ad.py --name bottle --data_path /path/to/mvtec
  python s2ad.py --name leather --layers layer123 --timesteps 8 16 32 --use_membrane --wandb
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
from spikingjelly.activation_based import ann2snn, functional
import random
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc
from scipy.ndimage import label as connected_components
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import matplotlib.cm as cm
import setproctitle
setproctitle.setproctitle("python") 

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
            self.output_indices = [4, 7, 9]   # sau conv2, conv3, conv5
        elif backbone == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            self.feat_channels = {'layer1': 32, 'layer2': 96, 'layer3': 320}
            self.output_indices = [3, 10, 17]  # blocks 4, 10, 17
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
            self.output_indices = [4, 6, 8]   # sau dense block 2, 3, 4
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.features = model
        # Đảm bảo output_indices sắp xếp tăng dần (để layer1 là sớm nhất)
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
            # Đảm bảo có đúng 3 output (nếu thiếu thì lấy output cuối cùng lặp lại)
            while len(outputs) < 3:
                outputs.append(x)
            if len(outputs) > 3:
                outputs = outputs[:3]
        
        # Chọn output dựa trên self.layers
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
# SECTION 2 - Dataset
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
    """
    VisA dataset loader compatible with anomalib's downloaded structure:
        root/visa/visa_pytorch/category/
            train/good/
            test/good/ and test/bad/
            ground_truth/bad/
    """
    def __init__(self, root, category, split='train', img_size=256, max_samples=None):
        self.transform = get_transform(img_size)
        self.img_size = img_size
        
        # Dữ liệu thực tế nằm trong thư mục con visa_pytorch
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
            
            # Ảnh normal
            good_dir = os.path.join(test_root, 'good')
            if os.path.exists(good_dir):
                for f in sorted(glob(os.path.join(good_dir, '*'))):
                    self.files.append(f)
                    self.labels.append(0)
                    self.gt_paths.append(None)
            
            # Ảnh anomaly
            bad_dir = os.path.join(test_root, 'bad')
            if os.path.exists(bad_dir):
                for f in sorted(glob(os.path.join(bad_dir, '*'))):
                    self.files.append(f)
                    self.labels.append(1)
                    fname = os.path.splitext(os.path.basename(f))[0]
                    mask_path = os.path.join(gt_root, 'bad', fname + '_mask.png')
                    if not os.path.exists(mask_path):
                        # Thử các đuôi khác nếu cần
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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 - SNN Conversion
# ═══════════════════════════════════════════════════════════════════════════

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
            converter_mode = float(mode)   # "0.98" → 0.98
        except ValueError:
            converter_mode = mode   # fallback (sẽ không dùng đến)
    
    converter = ann2snn.Converter(
        dataloader=adapter,
        device=device,
        mode=converter_mode,
        momentum=0.1
    )
    snn_encoder = converter(ann_encoder)
    # ÉP BUỘC OUTPUT LÀ SPIKE
    for module in snn_encoder.modules():
        if hasattr(module, 'output'):
            module.output = True
        if hasattr(module, 'out_spike'):
            module.out_spike = True

    # Kiểm tra nhanh output có phải spike không
    with torch.no_grad():
        test_input, _, _ = next(iter(calib_loader))  # cần pass calib_loader vào
        test_input = test_input[:1].to(device)
        test_out = snn_encoder(test_input)
        if isinstance(test_out, tuple):
            test_out = test_out[0]
        max_val = test_out.max().item()
        print(f"  Spike test: max output = {max_val} (must be <= 1.0)")
        if max_val > 1.0:
            print("  WARNING: Still not spike! Upgrade spikingjelly or check configuration.")
        else:
            print("  SUCCESS: SNN outputs spikes (0/1).")
    print(f"  ANN2SNN conversion complete (mode={converter_mode})")
    return snn_encoder


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 - Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_layer_indices_and_names(layers):
    """Convert layer string to list of indices and names (0-indexed based on output order)."""
    mapping = {
        'layer1': ([0], ['layer1']),
        'layer2': ([0], ['layer2']),
        'layer3': ([0], ['layer3']),
        'layer12': ([0, 1], ['layer1', 'layer2']),
        'layer23': ([0, 1], ['layer2', 'layer3']),
        'layer123': ([0, 1, 2], ['layer1', 'layer2', 'layer3']),
    }
    return mapping.get(layers, ([0, 1], ['layer2', 'layer3']))

def _get_spike_features(snn_encoder, imgs, timesteps, layer_indices, layer_names, device):
    """
    Extract true spike firing rate from cumulative membrane potential.
    For SpikingJelly 0.0.0.0.14 where ann2snn outputs cumulative membrane.
    """
    imgs = imgs.to(device)
    functional.reset_net(snn_encoder)
    
    # Lấy ngưỡng v_threshold (mặc định 1.0)
    v_th = 1.0
    for module in snn_encoder.modules():
        if hasattr(module, 'v_threshold'):
            v_th = float(module.v_threshold)
            break
    
    spike_counts = {}
    prev_cumulative = {}  # lưu cumulative của timestep trước
    
    with torch.no_grad():
        for t in range(timesteps):
            outputs = snn_encoder(imgs)  # tuple các cumulative membrane tensors
            
            for idx, name in zip(layer_indices, layer_names):
                cumulative = outputs[idx]  # shape (B, C, H, W)
                
                if t == 0:
                    # Timestep đầu: số spike = floor(cumulative / v_th), tối đa 1
                    n_fires = torch.floor(cumulative.clamp(min=0) / v_th).clamp(max=1)
                    spike_counts[name] = n_fires.float()
                else:
                    # Timestep sau: tính delta so với bước trước
                    delta = cumulative - prev_cumulative[name]
                    n_fires = torch.floor(delta.clamp(min=0) / v_th).clamp(max=1)
                    spike_counts[name] += n_fires.float()
                
                # Lưu cumulative hiện tại cho bước tiếp theo
                prev_cumulative[name] = cumulative.clone()
    
    # Tính firing rate trung bình
    rates = {}
    for name in layer_names:
        rates[name] = spike_counts[name] / timesteps
    return rates
# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 - Normal Statistics Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_normal_stats(snn_encoder, normal_loader, device, timesteps, layers='layer23'):
    snn_encoder.eval()
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    
    # PASS 1: mean, std, và max
    sum_rates = {name: None for name in layer_names}
    sum_sq_rates = {name: None for name in layer_names}
    max_rates = {name: 0.0 for name in layer_names}
    count = 0
    
    with torch.no_grad():
        for imgs, _, _ in normal_loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            functional.reset_net(snn_encoder)
            spike_acc = {name: None for name in layer_names}
            for t in range(timesteps):
                outputs = snn_encoder(imgs)
                for idx, name in zip(layer_indices, layer_names):
                    feat = outputs[idx]
                    spike = (feat > 0).float()
                    if spike_acc[name] is None:
                        spike_acc[name] = spike
                    else:
                        spike_acc[name] += spike
            for name in layer_names:
                rate = spike_acc[name] / timesteps
                if sum_rates[name] is None:
                    sum_rates[name] = rate.sum(dim=0).cpu()
                    sum_sq_rates[name] = (rate ** 2).sum(dim=0).cpu()
                else:
                    sum_rates[name] += rate.sum(dim=0).cpu()
                    sum_sq_rates[name] += (rate ** 2).sum(dim=0).cpu()
                # Cập nhật giá trị lớn nhất của rate (trên toàn bộ batch)
                current_max = rate.max().item()
                if current_max > max_rates[name]:
                    max_rates[name] = current_max
            count += B
    
    # Tính mean, std
    means = {}
    stats = {}
    for name in layer_names:
        mean = sum_rates[name] / count
        var = (sum_sq_rates[name] / count) - (mean ** 2)
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var + 1e-8)
        means[name] = mean
        stats[name] = {
            'mean': mean,
            'std': std,
            'max_rate': max_rates[name]
        }
    
    # PASS 2: MAD (giữ nguyên)
    sum_abs_dev = {name: 0.0 for name in layer_names}
    count = 0
    with torch.no_grad():
        for imgs, _, _ in normal_loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            functional.reset_net(snn_encoder)
            spike_acc = {name: None for name in layer_names}
            for t in range(timesteps):
                outputs = snn_encoder(imgs)
                for idx, name in zip(layer_indices, layer_names):
                    feat = outputs[idx]
                    spike = (feat > 0).float()
                    if spike_acc[name] is None:
                        spike_acc[name] = spike
                    else:
                        spike_acc[name] += spike
            for name in layer_names:
                rate = spike_acc[name] / timesteps
                abs_dev = torch.abs(rate - means[name].to(device)).mean().item()
                sum_abs_dev[name] += abs_dev * B
            count += B
    
    for name in layer_names:
        mad = sum_abs_dev[name] / count
        stats[name]['mad'] = mad
        print(f'    {name}: mean={stats[name]["mean"].mean().item():.4f}, '
              f'max_rate={stats[name]["max_rate"]:.4f}, '
              f'std={stats[name]["std"].mean().item():.4f}, mad={mad:.6f}')
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 - Anomaly Scoring
# ═══════════════════════════════════════════════════════════════════════════

def get_firing_rates(snn_encoder, img_tensor, device, timesteps, layers='layer23'):
    functional.reset_net(snn_encoder)
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    spike_acc = {name: None for name in layer_names}
    with torch.no_grad():
        for t in range(timesteps):
            outputs = snn_encoder(img_tensor)
            for idx, name in zip(layer_indices, layer_names):
                feat = outputs[idx]
                spike = (feat > 0).float()   # spike thực
                if spike_acc[name] is None:
                    spike_acc[name] = spike
                else:
                    spike_acc[name] += spike
    rates = {}
    for name in layer_names:
        rates[name] = spike_acc[name] / timesteps
    return rates


def score_image(snn_encoder, img_tensor, normal_stats, device, timesteps,
                layers='layer23', img_size=256, use_membrane=False, combine_method='simple'):
    snn_encoder.eval()
    img_tensor = img_tensor.to(device)
    
    rates = get_firing_rates(snn_encoder, img_tensor, device, timesteps, layers)
    
    deviations = {}
    for layer_name, rate in rates.items():
        mean = normal_stats[layer_name]['mean'].to(device)
        std = normal_stats[layer_name]['std'].to(device)
        z_score = (rate[0] - mean) / std
        deviation = torch.abs(z_score).mean(dim=0)
        deviations[layer_name] = deviation
    
    if len(deviations) == 1:
        score_spatial = list(deviations.values())[0]
    else:
        # Xác định layer có độ phân giải cao nhất (lấy cái đầu tiên làm target)
        target_name = list(deviations.keys())[0]
        target_res = deviations[target_name].shape
        
        if combine_method == 'simple':
            # Cách cũ: average đơn giản sau khi upsample
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
        
        else:  # combine_method == 'mad_weighted'
            # Phương pháp mới: trọng số tỉ lệ nghịch với MAD
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
    
    if use_membrane:
        v_score = _get_membrane_score(snn_encoder)
        if v_score is not None:
            v_score_up = F.interpolate(
                v_score.unsqueeze(0).unsqueeze(0),
                size=score_spatial.shape, mode='bilinear', align_corners=False
            ).squeeze()
            v_norm = v_score_up / (v_score_up.max() + 1e-8)
            score_spatial = score_spatial * (1.0 + 0.5 * v_norm)
    
    score_map = F.interpolate(
        score_spatial.unsqueeze(0).unsqueeze(0).float(),
        size=(img_size, img_size), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    
    return score_map, float(np.max(score_map))


def _get_membrane_score(snn_encoder):
    for name, module in snn_encoder.named_modules():
        if 'IFNode' in str(type(module)) and hasattr(module, 'v') and module.v is not None:
            v = module.v
            if v is not None:
                return v.pow(2).mean(dim=1).squeeze(0)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 - Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def compute_pro_metric(gt_masks, anomaly_maps, fpr_limit=0.3):
    """
    Tính PRO (Per-Region Overlap) AUC up to FPR Limit bằng Numpy thuần cực nhanh.
    """
    if not gt_masks or not anomaly_maps:
        return 0.0

    all_amaps = np.array(anomaly_maps)
    all_masks = np.array(gt_masks)

    # Gom điểm số của vùng không bị lỗi (normal) để tính FPR
    normal_scores = all_amaps[all_masks == 0]
    total_normal_pixels = len(normal_scores)
    if total_normal_pixels == 0:
        return 0.0

    # Lấy 100 threshold chia đều từ min đến max của bộ điểm
    thresholds = np.linspace(all_amaps.min(), all_amaps.max(), 100)
    
    # Sắp xếp normal_score để dùng binary search cho nhanh
    normal_scores_sorted = np.sort(normal_scores)

    # Lấy trước các Region (Tránh vòng lặp tính đi tính lại)
    regions_list = []
    for mask in all_masks:
        labeled, num_regions = connected_components(mask)
        regions_list.append([labeled == reg_id for reg_id in range(1, num_regions + 1)])

    fprs = []
    pros = []

    for t in thresholds:
        # 1. Tính FPR bằng thuật toán tìm kiếm nhị phân (Siêu nhanh)
        fp_count = total_normal_pixels - np.searchsorted(normal_scores_sorted, t)
        fpr = fp_count / total_normal_pixels
        fprs.append(fpr)

        # 2. Tính PRO trung bình cho ngưỡng này
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

    # Lọc các điểm nằm trong giới hạn FPR (expect_fpr = 0.3)
    idxes = fprs <= fpr_limit
    fprs_valid = fprs[idxes]
    pros_valid = pros[idxes]

    if len(fprs_valid) < 2:
        return 0.0

    fprs_normalized = (fprs_valid - fprs_valid.min()) / (fprs_valid.max() - fprs_valid.min() + 1e-8)

    # Tích phân diện tích trên trục FPR đã được chuẩn hóa
    pro_auc = auc(fprs_normalized, pros_valid)
    
    return float(pro_auc)

def evaluate(snn_encoder, test_dataset, normal_stats, device, timesteps,
             layers='layer23', img_size=256, use_membrane=False, combine_method='mad_weighted',
             save_maps=False, maps_dir=None, category_name=''):
    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []
    gt_masks = []
    anomaly_maps = []
    
    # Lấy danh sách đường dẫn ảnh từ dataset (nếu có)
    if hasattr(test_dataset, 'files'):
        file_paths = test_dataset.files
    else:
        file_paths = [None] * len(test_dataset)
    
    for i in range(len(test_dataset)):
        img_t, lbl, gt_path = test_dataset[i]
        img_t = img_t.unsqueeze(0)
        
        score_map, img_score = score_image(
            snn_encoder, img_t, normal_stats, device, timesteps,
            layers, img_size, use_membrane, combine_method
        )
        
        img_scores.append(img_score)
        img_labels.append(lbl)
        
        # ==================== LƯU ANOMALY MAP (phân theo thư mục con) ====================
        if save_maps and maps_dir:
            # Xác định tên thư mục con dựa trên đường dẫn ảnh
            subfolder_name = "unknown"
            if file_paths[i]:
                path_parts = os.path.normpath(file_paths[i]).split(os.sep)
                # Tìm 'test' (MVTec) hoặc 'bad' (VisA)
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
            
            # Đường dẫn lưu: maps_dir/subfolder_name
            save_dir = os.path.join(maps_dir, subfolder_name)
            
            # Đọc ảnh gốc (RGB)
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
            
            # Ground truth (nếu có)
            gt_mask = None
            if lbl == 1 and gt_path and os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, 0)
                if gt_mask is not None:
                    gt_mask = cv2.resize(gt_mask, (img_size, img_size))
                    gt_mask = (gt_mask > 127).astype(np.uint8) * 255
            
            save_anomaly_map(orig_img, score_map, gt_mask, save_dir, i)
        # =========================================================
        
        if lbl == 1 and gt_path:
            gt = cv2.imread(gt_path, 0)
            if gt is not None:
                gt = cv2.resize(gt, (img_size, img_size))
                gt_bin = (gt > 127).astype(int)
                pix_scores.extend(score_map.flatten())
                pix_labels.extend(gt_bin.flatten())
                gt_masks.append(gt_bin)
                anomaly_maps.append(score_map)
    
    # Image metrics
    img_auc = roc_auc_score(img_labels, img_scores) if len(set(img_labels)) == 2 else None
    img_ap = average_precision_score(img_labels, img_scores) if len(set(img_labels)) == 2 else None
    
    prec, rec, _ = precision_recall_curve(img_labels, img_scores)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    img_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
    
    # Pixel metrics
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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 - Main
# ═══════════════════════════════════════════════════════════════════════════
def get_dataset_class(dataset_name):
    if dataset_name == 'mvtec':
        return MVTecDataset
    elif dataset_name == 'visa':
        return VisADataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def parse_args():
    parser = argparse.ArgumentParser(description='S2AD - Statistical SNN-based Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'],
                    help='Dataset to use: mvtec or visa')
    parser.add_argument('--name', type=str, required=True, help='MVTec category')
    parser.add_argument('--data_path', type=str, default='/home/minhtringuyen/ANN2SNN/datasets')
    parser.add_argument('--img_size', type=int, default=256)
    
    parser.add_argument('--timesteps', type=int, nargs='+', default=[16],
                        help='List of timesteps to evaluate (e.g., 8 16 32 64)')
    parser.add_argument('--calib_samples', type=int, default=500)
    parser.add_argument('--snn_mode', type=str, default='max',
                    help='ANN2SNN mode: "max" or a percentile value (e.g., "0.99", "0.98")')
    
    parser.add_argument('--layers', type=str, default='layer23',
                        choices=['layer1', 'layer2', 'layer3', 'layer12', 'layer23', 'layer123'])
    parser.add_argument('--backbone', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet',
                             'mobilenet_v2', 'mobilenet_v3_large', 'densenet121', 'densenet169'],
                    help='Backbone architecture')
    
    parser.add_argument('--use_membrane', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='./s2ad_results_simple')
    
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='S2AD')
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='Custom run name for WandB')
    parser.add_argument('--combine_method', type=str, default='simple',
                    choices=['simple', 'mad_weighted'],
                    help='Method to combine multi-layer deviations')
    parser.add_argument('--save_anomaly_maps', action='store_true', help='Save anomaly map images for test samples')
    parser.add_argument('--maps_root', type=str, default='./anomaly_maps', help='Root directory to save anomaly maps')
    return parser.parse_args()

def save_anomaly_map(original_img, score_map, gt_mask, save_dir, idx):
    """
    original_img: numpy array (H,W,3) uint8 (RGB)
    score_map: 2D numpy float (H,W)
    gt_mask: 2D numpy uint8 (0/255) hoặc None
    """
    # Chuẩn bị ảnh gốc (PIL)
    img_pil = Image.fromarray(original_img)
    
    # Chuẩn hóa anomaly map và tạo colormap jet
    smin, smax = score_map.min(), score_map.max()
    if smax > smin:
        score_norm = (score_map - smin) / (smax - smin)
    else:
        score_norm = score_map
    cmap = cm.jet(score_norm)
    # Bỏ kênh alpha, chuyển về uint8
    anomaly_colored = (cmap[:, :, :3] * 255).astype(np.uint8)
    anomaly_pil = Image.fromarray(anomaly_colored)
    
    # 1. Lưu ảnh gốc
    os.makedirs(save_dir, exist_ok=True)
    img_pil.save(os.path.join(save_dir, f'{idx:04d}_img.png'))
    
    # 2. Lưu ảnh blend (chồng anomaly map lên ảnh gốc)
    blended = Image.blend(img_pil, anomaly_pil, alpha=0.4)
    blended.save(os.path.join(save_dir, f'{idx:04d}_blend.png'))
    
    # 3. Lưu ground truth (nếu có)
    if gt_mask is not None:
        # gt_mask đang là (H,W) uint8 (0/255), chuyển thành ảnh xám 3 kênh để lưu
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

def debug_snn(snn_encoder, normal_loader, device, timestep, layers='layer23'):
    print("\n[DEBUG] SNN thresholds (v_threshold of IF nodes):")
    found = False
    for name, module in snn_encoder.named_modules():
        if hasattr(module, 'v_threshold'):
            print(f"  {name}: v_threshold = {module.v_threshold:.6f}")
            found = True
    if not found:
        print("  WARNING: No v_threshold found.")
    
    print(f"\n[DEBUG] Reconstructed spike firing rate on a normal batch (T={timestep}):")
    sample_imgs, _, _ = next(iter(normal_loader))
    sample_imgs = sample_imgs.to(device)
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    rates = _get_spike_features(snn_encoder, sample_imgs, timestep, layer_indices, layer_names, device)
    for name in layer_names:
        rate = rates[name]
        print(f"  Layer {name}: mean={rate.mean().item():.6f}, max={rate.max().item():.6f}, std={rate.std().item():.6f}")

def main():
    seed_everything(42)
    g = torch.Generator()
    g.manual_seed(42)

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('=' * 60)
    print(f'S2AD - Statistical SNN-based Anomaly Detection')
    print(f'  Category: {args.name}')
    print(f'  Device: {device}')
    print(f'  Timesteps to test: {args.timesteps}')
    print(f'  Layers: {args.layers}')
    print(f'  Use membrane: {args.use_membrane}')
    print(f'  Calibration samples: {args.calib_samples}')
    print('=' * 60)
    
    # Initialize WandB
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        if args.wandb_key:
            os.environ['WANDB_API_KEY'] = args.wandb_key
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name if args.wandb_run_name else f'{args.name}_{args.layers}_mem{args.use_membrane}',
                config=vars(args),
                mode='offline' if args.wandb_offline else 'online'
            )
            print(f"  WandB logging enabled: {wandb_run.project}/{wandb_run.name}")
        except Exception as e:
            print(f'WandB init failed: {e}')
    
    # Build ANN encoder
    print(f'\n[1/4] Building encoder (backbone={args.backbone})...')
    ann_encoder = BackboneEncoder(backbone=args.backbone, layers=args.layers).to(device)
    ann_encoder.eval()
    
    dummy = torch.randn(2, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        outputs = ann_encoder(dummy)
        layer_indices, layer_names = get_layer_indices_and_names(args.layers)
        for idx, name in zip(layer_indices, layer_names):
            print(f'  {name} max activation: {outputs[idx].max().item():.3f}')
    
    # Load normal dataset - CHỈ 1 LẦN
    dataset_class = get_dataset_class(args.dataset)
    data_root = os.path.join(args.data_path, args.dataset)
    print('\n[2/4] Loading normal dataset...')
    full_normal_ds = dataset_class(data_root, args.name, 'train', img_size=args.img_size)
    print(f'  Full normal set: {len(full_normal_ds)} images')
    
        # Tạo subset chung cho calibration và normal statistics
    if args.calib_samples > 0 and args.calib_samples < len(full_normal_ds):
        subset_indices = list(range(args.calib_samples))
        normal_subset = Subset(full_normal_ds, subset_indices)
        calib_loader = DataLoader(
            normal_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
        )
        normal_loader = DataLoader(
            normal_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
        )
        print(f'  Using {args.calib_samples} samples for both calibration and normal statistics')
    else:
        calib_loader = DataLoader(
            full_normal_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
        )
        normal_loader = DataLoader(
            full_normal_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
        )
        print(f'  Using full normal set for both calibration and normal statistics')

    
    # ANN2SNN conversion - dùng calib_loader
    print('\n[3/4] Converting ANN to SNN...')
    snn_encoder = build_snn_encoder(ann_encoder, calib_loader, device, mode=args.snn_mode)
    # Debug unique values in SNN output (chỉ chạy một lần cho một batch nhỏ)
    with torch.no_grad():
        sample_imgs, _, _ = next(iter(normal_loader))
        sample_imgs = sample_imgs.to(device)
        functional.reset_net(snn_encoder)
        out1 = snn_encoder(sample_imgs)
        out2 = snn_encoder(sample_imgs)
        for idx, name in zip(layer_indices, layer_names):
            v1 = out1[idx]
            v2 = out2[idx]
            print(f"{name} step1 unique values: {v1.unique()[:10]}")
            print(f"{name} step2 unique values: {v2.unique()[:10]}")
    snn_encoder.eval()

    # # ========== DEBUG ==========
    # if args.timesteps:
    #     debug_snn(snn_encoder, normal_loader, device, args.timesteps[2], args.layers)
    # # ===========================
    # ================================
    
    # Load test dataset
    test_ds = dataset_class(data_root, args.name, 'test', img_size=args.img_size)
    n_anomaly = sum(test_ds.labels)
    n_normal = len(test_ds.labels) - n_anomaly
    print(f'  Normal set: {len(full_normal_ds)} images (for statistics)')
    print(f'  Test set: {len(test_ds)} images ({n_anomaly} anomaly, {n_normal} normal)')
    
    # Evaluate for each timestep - dùng normal_loader cho statistics
    print('\n[4/4] Evaluating across timesteps...')
    print(f'\n{"Timestep":>8} | {"Img AUC":>8} | {"Img AP":>8} | {"Img F1":>8} | {"Pix AUC":>8} | {"Pix AP":>8} | {"Pix F1":>8} | {"PRO":>8}')
    print('-' * 90)
    
    results = {}
    firing_rate_stats = {}
    
    for T in args.timesteps:
        print(f'\n  Testing with T={T}...')
        
        # Compute normal statistics - dùng normal_loader (TOÀN BỘ)
        normal_stats = compute_normal_stats(snn_encoder, normal_loader, device, T, args.layers)
        firing_rate_stats[T] = {}
        for name, stats in normal_stats.items():
            mean_val = stats['mean'].mean().item()
            std_val = stats['std'].mean().item()
            max_val = stats['max_rate']
            firing_rate_stats[T][name] = {'mean': mean_val, 'std': std_val, 'max': max_val}
        
        # Evaluate
        if args.save_anomaly_maps:
            # Tạo config_str: ví dụ "vgg16_mad_weightedlayer123"
            config_str = f"{args.backbone}_{args.combine_method}{args.layers}"
            snn_str = f"snnmode_{args.snn_mode}".replace('.', '_')
            maps_dir = os.path.join(args.maps_root, config_str, snn_str, args.name, f"T{T}")
        else:
            maps_dir = None

        metrics, img_scores, img_labels = evaluate(
            snn_encoder, test_ds, normal_stats, device, T,
            args.layers, args.img_size, args.use_membrane, args.combine_method,
            save_maps=args.save_anomaly_maps,
            maps_dir=maps_dir,
            category_name=args.name
        )
        
        img_auc_val = metrics['img_auc'] if metrics['img_auc'] else 0.0
        pix_auc_val = metrics['pix_auc'] if metrics['pix_auc'] else 0.0
        
        results[T] = {
            'img_auc': metrics['img_auc'],
            'img_ap': metrics['img_ap'],
            'img_f1': metrics['img_f1'],
            'pix_auc': metrics['pix_auc'],
            'pix_ap': metrics['pix_ap'],
            'pix_f1': metrics['pix_f1'],
            'pro': metrics['pro'],
        }
        
        print(f'  {T:8d} | {results[T]["img_auc"]:8.4f} | {results[T]["img_ap"]:8.4f} | {results[T]["img_f1"]:8.4f} | {results[T]["pix_auc"]:8.4f} | {results[T]["pix_ap"]:8.4f} | {results[T]["pix_f1"]:8.4f} | {results[T]["pro"]:8.4f}')
    
    # Print summary table
    print('\n' + '=' * 90)
    print('SUMMARY RESULTS:')
    print(f'{"Timestep":>8} | {"Img AUC":>8} | {"Img AP":>8} | {"Img F1":>8} | {"Pix AUC":>8} | {"Pix AP":>8} | {"Pix F1":>8} | {"PRO":>8}')
    print('-' * 90)
    for T in sorted(results.keys()):
        print(f'{T:8d} | {results[T]["img_auc"]:8.4f} | {results[T]["img_ap"]:8.4f} | {results[T]["img_f1"]:8.4f} | {results[T]["pix_auc"]:8.4f} | {results[T]["pix_ap"]:8.4f} | {results[T]["pix_f1"]:8.4f} | {results[T]["pro"]:8.4f}')
    print('=' * 90)
    
    # Log to WandB (giữ nguyên phần này)
    if wandb_run:
        timesteps = sorted(results.keys())
        img_aucs = [results[T]['img_auc'] for T in timesteps]
        pix_aucs = [results[T]['pix_auc'] for T in timesteps]
        
        wandb_run.config.update({
            'category': args.name,  
            'layers': args.layers,
            'use_membrane': args.use_membrane,
            'snn_mode': args.snn_mode,
            'calib_samples': args.calib_samples,
            'img_size': args.img_size,
            'combine_method': args.combine_method,
        })
        
        for T in timesteps:
            for layer_name, stats in firing_rate_stats[T].items():
                wandb_run.summary[f'firing_rate/T{T}/{layer_name}_mean'] = stats['mean']
                wandb_run.summary[f'firing_rate/T{T}/{layer_name}_std'] = stats['std']
        
        img_table = wandb.Table(data=[[T, results[T]['img_auc']] for T in timesteps], 
                                columns=["Timestep", "Image AUC"])
        wandb_run.log({
            "Image AUC vs Timestep": wandb.plot.line(
                img_table, "Timestep", "Image AUC",
                title=f"Image AUC vs Timestep - {args.name}",
                stroke="blue"
            )
        })
        
        pix_table = wandb.Table(data=[[T, results[T]['pix_auc']] for T in timesteps], 
                                columns=["Timestep", "Pixel AUC"])
        wandb_run.log({
            "Pixel AUC vs Timestep": wandb.plot.line(
                pix_table, "Timestep", "Pixel AUC",
                title=f"Pixel AUC vs Timestep - {args.name}"
            )
        })
        
        log_dict = {}
        for T in timesteps:
            log_dict[f'img_auc_T{T}'] = results[T]['img_auc']
            log_dict[f'img_ap_T{T}'] = results[T]['img_ap']
            log_dict[f'img_f1_T{T}'] = results[T]['img_f1']
            log_dict[f'pix_auc_T{T}'] = results[T]['pix_auc']
            log_dict[f'pix_ap_T{T}'] = results[T]['pix_ap']
            log_dict[f'pix_f1_T{T}'] = results[T]['pix_f1']
            log_dict[f'pro_T{T}'] = results[T]['pro']
        wandb_run.log(log_dict)
        
        wandb_run.summary['best_img_auc'] = max(img_aucs)
        wandb_run.summary['best_pix_auc'] = max(pix_aucs)
        wandb_run.summary['best_timestep'] = timesteps[img_aucs.index(max(img_aucs))]
        
        print(f'\n  WandB run: https://wandb.ai/{wandb_run.project}/runs/{wandb_run.id}')
        wandb_run.finish()
    
    # Save results to file
    out_path = os.path.join(args.save_dir, f'{args.name}_results.txt')
    with open(out_path, 'w') as f:
        f.write(f"S2AD Results - {args.name}\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Layers: {args.layers}\n")
        f.write(f"Use membrane: {args.use_membrane}\n")
        f.write(f"Calibration samples: {args.calib_samples}\n")
        f.write(f"\n{'Timestep':>8} | {'Img AUC':>8} | {'Img AP':>8} | {'Img F1':>8} | {'Pix AUC':>8} | {'Pix AP':>8} | {'Pix F1':>8} | {'PRO':>8}\n")
        f.write('-' * 95 + '\n')
        for T in sorted(results.keys()):
            f.write(f'{T:8d} | {results[T]["img_auc"]:8.4f} | {results[T]["img_ap"]:8.4f} | {results[T]["img_f1"]:8.4f} | {results[T]["pix_auc"]:8.4f} | {results[T]["pix_ap"]:8.4f} | {results[T]["pix_f1"]:8.4f} | {results[T]["pro"]:8.4f}\n')
        f.write(f"\n\nFiring Rate Statistics:\n")
        f.write(f"{'=' * 50}\n")
        for T in sorted(firing_rate_stats.keys()):
            f.write(f"\nTimestep T={T}:\n")
            for layer_name, stats in firing_rate_stats[T].items():
                f.write(f"  {layer_name}: mean={stats['mean']:.6f}, max={stats['max']:.6f}, std={stats['std']:.6f}\n")
    print(f'\nResults saved: {out_path}')


if __name__ == '__main__':
    main()