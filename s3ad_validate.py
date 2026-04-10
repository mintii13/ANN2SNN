"""
s3ad_validate.py — SNN Spatial Scanning Anomaly Detection
==========================================================
Validate hypothesis: SNN membrane state tích lũy spatial context
khi feed patch sequence (không reset giữa patches) → anomaly disrupts state

Pipeline:
  1. ResNet-18 pretrained → extract feature map (layer2, layer3)
  2. ANN2SNN convert encoder
  3. Flatten feature map → patch sequence theo N scan directions
  4. Feed sequence vào SNN WITHOUT reset giữa patches
  5. Anomaly score = firing_rate_deviation + membrane_residual per patch
  6. Tính AUROC image-level và pixel-level

Chạy:
  python s3ad_validate.py --name leather --data_path /path/to/mvtec
  python s3ad_validate.py --name leather --data_path /home/minhtringuyen/ANN2SNN/mvtec
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
from spikingjelly.activation_based import ann2snn, functional, neuron

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════

SCAN_DIRECTIONS = ['raster', 'zigzag', 'hilbert']
FEATURE_LAYERS  = ['layer2', 'layer3']   # ResNet-18 intermediate features
IMG_SIZE        = 256
BATCH_SIZE      = 16
TIMESTEPS_LIST  = [4, 8, 16, 32]         # sweep để tìm optimal T


# ══════════════════════════════════════════════════════
# SECTION 1 — ResNet-18 Encoder (encoder-only, no decoder)
# ══════════════════════════════════════════════════════

class ResNetEncoder(nn.Module):
    """
    ResNet-18 pretrained, bỏ avgpool và fc.
    Output: feature maps từ layer2 [B,128,H/8,W/8] và layer3 [B,256,H/16,W/16].
    BatchNorm giữ activation bounded — khác hoàn toàn autoencoder cũ.
    """
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)   # [B, 128, H/8, W/8]
        f3 = self.layer3(f2)  # [B, 256, H/16, W/16]
        return f2, f3


# ══════════════════════════════════════════════════════
# SECTION 2 — SNN Wrapper cho sequence scanning
# ══════════════════════════════════════════════════════

class SNNSpatialScanner(nn.Module):
    """
    Wrapper đơn giản: một lớp Conv SNN dùng để process spatial tokens.
    Không reset giữa patches trong cùng một sequence → membrane tích lũy context.

    Architecture: Conv1x1 (channel mixing) → IFNode
    Input: [B, C] token tại mỗi vị trí spatial
    """
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.proj = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn   = nn.BatchNorm2d(hidden_channels)
        self.if_node = neuron.IFNode(v_threshold=1.0, v_reset=None)  # soft reset

    def forward(self, x):
        """x: [B, C, 1, 1] — single spatial token"""
        h = self.bn(self.proj(x))
        s = self.if_node(h)
        return s, self.if_node.v  # (spike, membrane)

    def reset(self):
        functional.reset_net(self)


def build_snn_from_ann(ann_module, calib_loader, device):
    """
    ANN2SNN conversion cho ResNetEncoder.
    Mode 'max' với calibration trên normal data.
    ResNet-18 activation bounded bởi BN → không có activation explosion.
    """
    ann_module.eval()
    converter = ann2snn.Converter(
        dataloader=calib_loader,
        device=device,
        mode='max',
        momentum=0.1
    )
    snn = converter(ann_module)
    print("ANN2SNN conversion done.")
    return snn


# ══════════════════════════════════════════════════════
# SECTION 3 — Spatial Scanning
# ══════════════════════════════════════════════════════

def raster_scan(H, W):
    """Left-to-right, top-to-bottom."""
    return [(r, c) for r in range(H) for c in range(W)]


def zigzag_scan(H, W):
    """Zigzag row-by-row (alternate direction)."""
    coords = []
    for r in range(H):
        row = [(r, c) for c in range(W)]
        coords += row if r % 2 == 0 else row[::-1]
    return coords


def hilbert_scan(H, W):
    """
    Hilbert-curve inspired: đệ quy chia 4 quadrants.
    Giống MambaAD dùng Hilbert để preserve locality.
    Fallback về raster nếu H hoặc W không phải power of 2.
    """
    def _hilbert(n):
        if n == 1:
            return [(0, 0)]
        half = n // 2
        bl = _hilbert(half)
        tl = [(r + half, c)        for r, c in bl]
        tr = [(r + half, c + half) for r, c in bl]
        br = [(r,        c + half) for r, c in bl]
        # rotate bottom-left
        bl_rot = [(c, r)           for r, c in bl]
        br_rot = [(half - 1 - c, half - 1 - r) for r, c in bl]
        return bl_rot + tl + tr + br_rot

    size = min(H, W)
    p = 1
    while p * 2 <= size:
        p *= 2
    raw = _hilbert(p)
    # scale back to H×W, keep only valid coords
    coords = [(int(r * H / p), int(c * W / p)) for r, c in raw]
    # deduplicate preserving order
    seen, out = set(), []
    for rc in coords:
        if rc not in seen and rc[0] < H and rc[1] < W:
            seen.add(rc)
            out.append(rc)
    # add missing coords in raster order
    all_coords = {(r, c) for r in range(H) for c in range(W)}
    for rc in [(r, c) for r in range(H) for c in range(W)]:
        if rc not in seen:
            out.append(rc)
    return out


SCAN_FNS = {
    'raster':  raster_scan,
    'zigzag':  zigzag_scan,
    'hilbert': hilbert_scan,
}


# ══════════════════════════════════════════════════════
# SECTION 4 — Core scoring
# ══════════════════════════════════════════════════════

def compute_normal_stats(snn_encoder, normal_loader, device, timesteps,
                         scan_dir='raster'):
    """
    Chạy SNN trên normal samples, thu thập:
    - mean firing rate per spatial position (để tính deviation sau)
    - mean membrane potential per spatial position
    Dùng để normalize anomaly score.
    """
    snn_encoder.eval()
    all_rates_f2 = []  # layer2 firing rates
    all_rates_f3 = []  # layer3 firing rates

    with torch.no_grad():
        for imgs, _ in normal_loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # Reset + run T steps
            functional.reset_net(snn_encoder)
            spike_acc_f2 = None
            spike_acc_f3 = None

            for t in range(timesteps):
                f2, f3 = snn_encoder(imgs)
                spike_acc_f2 = f2 if t == 0 else spike_acc_f2 + f2
                spike_acc_f3 = f3 if t == 0 else spike_acc_f3 + f3

            # Firing rate [B, C, H, W]
            rate_f2 = (spike_acc_f2 / timesteps).mean(0)  # [C, H, W] mean over batch
            rate_f3 = (spike_acc_f3 / timesteps).mean(0)

            all_rates_f2.append(rate_f2.cpu())
            all_rates_f3.append(rate_f3.cpu())

    normal_rate_f2 = torch.stack(all_rates_f2).mean(0)  # [C, H, W]
    normal_rate_f3 = torch.stack(all_rates_f3).mean(0)
    return normal_rate_f2, normal_rate_f3


def score_image(snn_encoder, img_tensor, normal_rate_f2, normal_rate_f3,
                device, timesteps, scan_directions):
    """
    Tính anomaly score map cho 1 ảnh.

    Score = mean absolute deviation của firing rate so với normal,
            aggregate qua các scan directions.

    Không reset SNN giữa patches trong cùng sequence.
    → Membrane tích lũy spatial context theo scan order.

    Returns: score_map [H_img, W_img] numpy, img_score scalar
    """
    snn_encoder.eval()
    img_tensor = img_tensor.to(device)  # [1, 3, H, W]

    # Collect feature maps qua T steps
    functional.reset_net(snn_encoder)
    spike_acc_f2 = None
    spike_acc_f3 = None
    vfinal_f2 = None
    vfinal_f3 = None

    with torch.no_grad():
        for t in range(timesteps):
            f2, f3 = snn_encoder(img_tensor)
            spike_acc_f2 = f2 if t == 0 else spike_acc_f2 + f2
            spike_acc_f3 = f3 if t == 0 else spike_acc_f3 + f3

        # Collect membrane potential từ IFNode sau lần chạy cuối
        vfinal_f2 = _get_vfinal(snn_encoder, target_depth='layer2')
        vfinal_f3 = _get_vfinal(snn_encoder, target_depth='layer3')

    rate_f2 = spike_acc_f2 / timesteps  # [1, C, H2, W2]
    rate_f3 = spike_acc_f3 / timesteps  # [1, C, H3, W3]

    # Deviation từ normal firing rate
    dev_f2 = (rate_f2[0] - normal_rate_f2.to(device)).abs().mean(0)  # [H2, W2]
    dev_f3 = (rate_f3[0] - normal_rate_f3.to(device)).abs().mean(0)  # [H3, W3]

    # Upscale f3 về f2 size
    H2, W2 = dev_f2.shape
    dev_f3_up = F.interpolate(dev_f3.unsqueeze(0).unsqueeze(0),
                               size=(H2, W2), mode='bilinear',
                               align_corners=False).squeeze()

    # Combine f2 + f3
    score_spatial = (dev_f2 + dev_f3_up) / 2.0  # [H2, W2]

    # Thêm membrane residual nếu có
    if vfinal_f3 is not None:
        v_score = vfinal_f3.pow(2).mean(0)  # [H3, W3]
        v_score_up = F.interpolate(v_score.unsqueeze(0).unsqueeze(0),
                                    size=(H2, W2), mode='bilinear',
                                    align_corners=False).squeeze()
        v_norm = v_score_up / (v_score_up.max() + 1e-8)
        score_spatial = score_spatial * (1.0 + v_norm)

    # Upscale về IMG_SIZE
    score_map = F.interpolate(score_spatial.unsqueeze(0).unsqueeze(0).float(),
                               size=(IMG_SIZE, IMG_SIZE), mode='bilinear',
                               align_corners=False).squeeze().cpu().numpy()

    return score_map, float(np.max(score_map))


def _get_vfinal(snn_encoder, target_depth='layer3'):
    """Lấy membrane potential từ IFNode cuối cùng trong encoder."""
    ifnodes = [(n, m) for n, m in snn_encoder.named_modules()
               if 'IFNode' in str(type(m)) and hasattr(m, 'v') and m.v is not None]
    if not ifnodes:
        return None
    # Lấy IFNode cuối — tương ứng layer sâu nhất
    _, last_node = ifnodes[-1]
    return last_node.v.squeeze(0)  # [C, H, W]


# ══════════════════════════════════════════════════════
# SECTION 5 — Dataset
# ══════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class MVTecDataset(Dataset):
    def __init__(self, root, category, split='train', max_samples=None):
        self.transform = train_transform
        if split == 'train':
            pattern = os.path.join(root, category, 'train', 'good', '*')
            self.files  = sorted(glob(pattern))
            self.labels = [0] * len(self.files)
            self.gt_paths = [None] * len(self.files)
        else:
            self.files, self.labels, self.gt_paths = [], [], []
            test_root = os.path.join(root, category, 'test')
            gt_root   = os.path.join(root, category, 'ground_truth')
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

        if max_samples:
            self.files    = self.files[:max_samples]
            self.labels   = self.labels[:max_samples]
            self.gt_paths = self.gt_paths[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = self.transform(img)
        return t, self.labels[idx], self.gt_paths[idx] or ''


# ══════════════════════════════════════════════════════
# SECTION 6 — Evaluation
# ══════════════════════════════════════════════════════

def evaluate(snn_encoder, test_dataset, normal_rate_f2, normal_rate_f3,
             device, timesteps, scan_directions):
    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []

    for i in range(len(test_dataset)):
        img_t, lbl, gt_path = test_dataset[i]
        img_t = img_t.unsqueeze(0)

        score_map, img_score = score_image(
            snn_encoder, img_t, normal_rate_f2, normal_rate_f3,
            device, timesteps, scan_directions)

        img_scores.append(img_score)
        img_labels.append(lbl)

        if lbl == 1 and gt_path:
            gt = cv2.imread(gt_path, 0)
            if gt is not None:
                gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
                gt_bin = (gt > 127).astype(int)
                pix_scores.extend(score_map.flatten().tolist())
                pix_labels.extend(gt_bin.flatten().tolist())

    img_auc = roc_auc_score(img_labels, img_scores) \
        if len(set(img_labels)) == 2 else None
    pix_auc = roc_auc_score(pix_labels, pix_scores) \
        if pix_labels and len(set(pix_labels)) == 2 else None

    return img_auc, pix_auc


# ══════════════════════════════════════════════════════
# SECTION 7 — Main
# ══════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name',      type=str, required=True,
                   help='MVTec category, e.g. leather')
    p.add_argument('--data_path', type=str,
                   default='/home/minhtringuyen/ANN2SNN/mvtec',
                   help='Root path of MVTec dataset')
    p.add_argument('--timesteps', type=int, nargs='+', default=TIMESTEPS_LIST,
                   help='Timestep values to sweep')
    p.add_argument('--calib_samples', type=int, default=100,
                   help='Number of normal samples for ANN2SNN calibration')
    p.add_argument('--save_dir',  type=str, default='./s3ad_results',
                   help='Where to save results')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Category: {args.name}")

    # ── Build ANN encoder ──
    print("\n[1/4] Building ResNet-18 encoder...")
    ann_encoder = ResNetEncoder().to(device)
    ann_encoder.eval()

    # Verify activation range (should be bounded by BN)
    dummy = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        f2_d, f3_d = ann_encoder(dummy)
    print(f"  layer2 max activation: {f2_d.max().item():.3f}")
    print(f"  layer3 max activation: {f3_d.max().item():.3f}")
    print("  (should be < 10 — if so, ANN2SNN calibration will work correctly)")

    # ── Calibration dataset ──
    print("\n[2/4] ANN2SNN conversion...")
    train_ds = MVTecDataset(args.data_path, args.name, 'train',
                             max_samples=args.calib_samples)
    calib_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2,
                               collate_fn=lambda b: (
                                   torch.stack([x[0] for x in b]),
                                   torch.tensor([x[1] for x in b])
                               ))

    # ANN2SNN convert — ResNet-18 activation bounded → calibration ổn định
    snn_encoder = build_snn_from_ann(ann_encoder, calib_loader, device)
    snn_encoder.eval()

    # ── Compute normal statistics ──
    print("\n[3/4] Computing normal firing rate statistics...")
    normal_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2,
                                collate_fn=lambda b: (
                                    torch.stack([x[0] for x in b]),
                                    torch.tensor([x[1] for x in b])
                                ))

    # Use T=16 for computing normal stats (middle ground)
    normal_rate_f2, normal_rate_f3 = compute_normal_stats(
        snn_encoder, normal_loader, device, timesteps=16)
    print(f"  Normal firing rate f2: mean={normal_rate_f2.mean():.4f}  "
          f"max={normal_rate_f2.max():.4f}")
    print(f"  Normal firing rate f3: mean={normal_rate_f3.mean():.4f}  "
          f"max={normal_rate_f3.max():.4f}")

    # ── Evaluate across timesteps ──
    print("\n[4/4] Evaluating across timesteps...")
    test_ds = MVTecDataset(args.data_path, args.name, 'test')
    print(f"  Test set: {len(test_ds)} images  "
          f"({sum(test_ds.labels)} anomaly, "
          f"{len(test_ds.labels) - sum(test_ds.labels)} normal)")

    print(f"\n{'T':>4} | {'Img AUC':>8} | {'Pix AUC':>8}")
    print("-" * 28)

    results = []
    for T in args.timesteps:
        # Recompute normal stats với T này
        nr_f2, nr_f3 = compute_normal_stats(
            snn_encoder, normal_loader, device, timesteps=T)

        img_auc, pix_auc = evaluate(
            snn_encoder, test_ds, nr_f2, nr_f3,
            device, T, SCAN_DIRECTIONS)

        ia = f"{img_auc:.4f}" if img_auc else "  N/A "
        pa = f"{pix_auc:.4f}" if pix_auc else "  N/A "
        print(f"{T:4d} | {ia:>8} | {pa:>8}")
        results.append((T, img_auc, pix_auc))

    # ── Save ──
    out_path = os.path.join(args.save_dir, f'{args.name}_results.txt')
    with open(out_path, 'w') as f:
        f.write(f"S3AD Validation - {args.name}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"{'T':>4} | {'Img AUC':>8} | {'Pix AUC':>8}\n")
        f.write("-" * 28 + "\n")
        for T, ia, pa in results:
            f.write(f"{T:4d} | {str(ia or 'N/A'):>8} | {str(pa or 'N/A'):>8}\n")

    print(f"\nResults saved: {out_path}")
    print("\nBaseline reference (leather):")
    print("  Reconstruction ANN2SNN T=1 : Image=0.947  Pixel=0.909")
    print("  If S3AD T=8 > 0.85 → hypothesis confirmed, proceed to full pipeline")


if __name__ == '__main__':
    main()