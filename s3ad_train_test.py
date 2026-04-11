"""
s3ad_train_test.py — S3AD: SNN Spatial Scanning Anomaly Detection
==================================================================
Full spiking pipeline — no ANN at inference, neuromorphic deployable.

Architecture:
  - SNN Encoder: ResNet-18 ANN2SNN (frozen after conversion)
  - SNN Scanner: SNNSpatialScanner trained on normal data (feature reconstruction loss)
  - Anomaly score: firing rate deviation + membrane residual across N scan directions

Train:  python s3ad_train_test.py --mode train --name leather --data_path /path/to/mvtec
Test:   python s3ad_train_test.py --mode test  --name leather --data_path /path/to/mvtec
Both:   python s3ad_train_test.py --mode both  --name leather --data_path /path/to/mvtec
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from glob import glob
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from spikingjelly.activation_based import ann2snn, functional, neuron

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ══════════════════════════════════════════════════════
# SECTION 1 — Args
# ══════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='S3AD: SNN Spatial Scanning Anomaly Detection')

    # Run mode
    p.add_argument('--mode',         type=str,   default='both',
                   choices=['train', 'test', 'both'],
                   help='Run mode: train only / test only / both')

    # Dataset
    p.add_argument('--name',         type=str,   required=True,
                   help='MVTec category name, e.g. leather')
    p.add_argument('--data_path',    type=str,
                   default='/home/minhtringuyen/ANN2SNN/mvtec',
                   help='Root path of MVTec dataset')
    p.add_argument('--img_size',     type=int,   default=256,
                   help='Resize all images to this size')

    # SNN Encoder
    p.add_argument('--timesteps',    type=int,   default=16,
                   help='Timesteps for SNN encoder forward pass')
    p.add_argument('--calib_samples',type=int,   default=100,
                   help='Number of normal samples for ANN2SNN calibration')
    p.add_argument('--snn_mode',     type=str,   default='max',
                   choices=['max', '99percent'],
                   help='ANN2SNN calibration mode')

    # SNN Scanner
    p.add_argument('--hidden_channels', type=int, default=None,
                   help='Hidden channels in SNNSpatialScanner (default=same as input)')
    p.add_argument('--scan_directions', type=str, nargs='+',
                   default=['raster', 'zigzag', 'hilbert'],
                   choices=['raster', 'zigzag', 'hilbert'],
                   help='Spatial scan directions to use')

    # Training
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--eval_every',   type=int,   default=10,
                   help='Run test evaluation every N epochs')
    p.add_argument('--loss',         type=str,   default='mse',
                   choices=['mse', 'cosine', 'mse+cosine'],
                   help='Scanner reconstruction loss')

    # Paths
    p.add_argument('--save_dir',     type=str,   default='./s3ad_checkpoints',
                   help='Directory to save scanner checkpoints')
    p.add_argument('--result_dir',   type=str,   default='./s3ad_results',
                   help='Directory to save evaluation results')
    p.add_argument('--resume',       type=str,   default=None,
                   help='Path to scanner checkpoint to resume from')

    # WandB
    p.add_argument('--wandb',        action='store_true',
                   help='Enable WandB logging')
    p.add_argument('--wandb_project',type=str,   default='S3AD',
                   help='WandB project name')
    p.add_argument('--wandb_key',    type=str,   default=None,
                   help='WandB API key (or set WANDB_API_KEY env var)')

    return p.parse_args()


# ══════════════════════════════════════════════════════
# SECTION 2 — Dataset
# ══════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MVTecDataset(Dataset):
    def __init__(self, root, category, split='train',
                 img_size=256, max_samples=None):
        self.transform = make_transform(img_size)
        self.img_size  = img_size

        if split == 'train':
            pattern = os.path.join(root, category, 'train', 'good', '*.png')
            files   = sorted(glob(pattern))
            if not files:
                pattern = os.path.join(root, category, 'train', 'good', '*')
                files   = sorted(glob(pattern))
            self.files    = files[:max_samples] if max_samples else files
            self.labels   = [0] * len(self.files)
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
                    gt_path = None
                    if lbl == 1:
                        fname = os.path.splitext(os.path.basename(f))[0]
                        for ext in ['.png', '.bmp', '.jpg']:
                            for suf in ['_mask', '']:
                                p = os.path.join(gt_root, subfolder,
                                                 fname + suf + ext)
                                if os.path.exists(p):
                                    gt_path = p
                                    break
                            if gt_path:
                                break
                    self.gt_paths.append(gt_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), self.labels[idx], self.gt_paths[idx] or ''


def collate_train(batch):
    imgs   = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return imgs, labels


# ══════════════════════════════════════════════════════
# SECTION 3 — SNN Encoder (ANN2SNN ResNet-18, frozen)
# ══════════════════════════════════════════════════════

class ResNetEncoder(nn.Module):
    """
    ResNet-18 pretrained. Output: f2 [B,128,H/8,W/8], f3 [B,256,H/16,W/16].
    BatchNorm bounds activations → ANN2SNN calibration stable (max ~6-8).
    """
    def __init__(self):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


def build_snn_encoder(ann_encoder, calib_loader, device, mode='max'):
    """
    Convert ResNet-18 encoder to SNN via SpikingJelly ANN2SNN.
    Returns frozen SNN encoder.
    """
    ann_encoder.eval()
    m = mode if mode == 'max' else 0.99
    converter = ann2snn.Converter(
        dataloader=calib_loader,
        device=device,
        mode=m,
        momentum=0.1
    )
    snn = converter(ann_encoder)
    # Freeze — encoder is fixed, only scanner is trained
    for param in snn.parameters():
        param.requires_grad = False
    snn.eval()
    print("SNN encoder built and frozen.")
    return snn


def snn_encode(snn_encoder, imgs, timesteps, device):
    """
    Forward pass through SNN encoder for T timesteps.
    Returns firing rate maps f2 [B,128,H2,W2], f3 [B,256,H3,W3].
    These are spike-based features — no floating point activations.
    """
    imgs = imgs.to(device)
    functional.reset_net(snn_encoder)
    acc_f2 = acc_f3 = None
    with torch.no_grad():
        for _ in range(timesteps):
            f2, f3 = snn_encoder(imgs)
            acc_f2 = f2 if acc_f2 is None else acc_f2 + f2
            acc_f3 = f3 if acc_f3 is None else acc_f3 + f3
    return acc_f2 / timesteps, acc_f3 / timesteps  # firing rate [0,1]


# ══════════════════════════════════════════════════════
# SECTION 4 — SNN Spatial Scanner (trained)
# ══════════════════════════════════════════════════════

class SNNSpatialScanner(nn.Module):
    """
    SNN module that processes spatial tokens sequentially.
    Membrane state accumulates spatial context — not reset between tokens.
    Reset only once per image sequence (before scanning starts).
    """
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hc = hidden_channels or in_channels
        # Conv1x1 with bias - no normalization needed
        self.proj    = nn.Conv2d(in_channels, hc, 1, bias=True)
        self.if_node = neuron.IFNode(v_threshold=1.0, v_reset=None)  # soft reset

    def forward(self, x):
        """x: [B, C, 1, 1]"""
        h = self.proj(x)
        s = self.if_node(h)
        return s, self.if_node.v  # (spike, membrane)

    def reset(self):
        """Reset membrane potential - call once before scanning sequence"""
        self.if_node.reset()


# ══════════════════════════════════════════════════════
# SECTION 5 — Spatial Scan Functions
# ══════════════════════════════════════════════════════

def raster_scan(H, W):
    return [(r, c) for r in range(H) for c in range(W)]


def zigzag_scan(H, W):
    coords = []
    for r in range(H):
        row = [(r, c) for c in range(W)]
        coords += row if r % 2 == 0 else row[::-1]
    return coords


def hilbert_scan(H, W):
    def _h(n):
        if n == 1:
            return [(0, 0)]
        half = n // 2
        bl = _h(half)
        return ([(c, r) for r, c in bl] +
                [(r + half, c) for r, c in bl] +
                [(r + half, c + half) for r, c in bl] +
                [(half - 1 - c, half - 1 - r) for r, c in bl])

    size = min(H, W)
    p = 1
    while p * 2 <= size:
        p *= 2
    raw    = _h(p)
    scaled = [(int(r * H / p), int(c * W / p)) for r, c in raw]
    seen, out = set(), []
    for rc in scaled:
        if rc not in seen and rc[0] < H and rc[1] < W:
            seen.add(rc); out.append(rc)
    for r in range(H):
        for c in range(W):
            if (r, c) not in seen:
                out.append((r, c))
    return out


SCAN_FNS = {'raster': raster_scan, 'zigzag': zigzag_scan, 'hilbert': hilbert_scan}


# ══════════════════════════════════════════════════════
# SECTION 6 — Scan one feature map through scanner
# ══════════════════════════════════════════════════════

def scan_feature_map(feat, scanner, scan_dir):
    """
    feat: [B, C, H, W] — SNN firing rate feature map
    scanner: SNNSpatialScanner
    scan_dir: one of 'raster', 'zigzag', 'hilbert'

    Returns:
      spike_map: [B, H, W] — spike output per spatial position
      vmem_map:  [B, H, W] — membrane potential per spatial position
      recon_map: [B, C, H, W] — reconstructed feature map from spikes
                  (used for training loss)
    """
    B, C, H, W = feat.shape
    coords = SCAN_FNS[scan_dir](H, W)

    spike_map = torch.zeros(B, H, W, device=feat.device)
    vmem_map  = torch.zeros(B, H, W, device=feat.device)
    recon_map = torch.zeros(B, C, H, W, device=feat.device)

    scanner.reset()  # reset ONCE before sequence, not between tokens
    for (r, c) in coords:
        token       = feat[:, :, r:r+1, c:c+1]   # [B, C, 1, 1]
        spike, vmem = scanner(token)               # [B, C, 1, 1], [B, C, 1, 1]
        spike_map[:, r, c] = spike.mean(dim=1).squeeze(-1).squeeze(-1)
        vmem_map[:, r, c]  = vmem.abs().mean(dim=1).squeeze(-1).squeeze(-1)
        recon_map[:, :, r, c] = spike.squeeze(-1).squeeze(-1)

    return spike_map, vmem_map, recon_map


# ══════════════════════════════════════════════════════
# SECTION 7 — Loss
# ══════════════════════════════════════════════════════

class ScannerLoss(nn.Module):
    """
    Feature reconstruction loss for SNNSpatialScanner.
    Trains scanner to reproduce normal SNN firing rate features.
    Anomaly detection: normal → low loss, anomaly → high loss.

    Options:
      mse:         L2 between recon_map and feat
      cosine:      1 - cosine similarity per spatial position
      mse+cosine:  combination
    """
    def __init__(self, mode='mse'):
        super().__init__()
        self.mode = mode
        self.mse  = nn.MSELoss()

    def forward(self, recon_map, feat):
        if self.mode == 'mse':
            return self.mse(recon_map, feat)
        elif self.mode == 'cosine':
            # cosine per spatial position: [B, C, H, W] → mean over B,H,W
            r = recon_map.permute(0, 2, 3, 1)  # [B, H, W, C]
            f = feat.permute(0, 2, 3, 1)
            cos = F.cosine_similarity(r, f, dim=-1)  # [B, H, W]
            return (1 - cos).mean()
        else:  # mse+cosine
            r = recon_map.permute(0, 2, 3, 1)
            f = feat.permute(0, 2, 3, 1)
            cos = F.cosine_similarity(r, f, dim=-1)
            return self.mse(recon_map, feat) + (1 - cos).mean()


# ══════════════════════════════════════════════════════
# SECTION 8 — Training
# ══════════════════════════════════════════════════════

def train(args, snn_encoder, scanner_f2, scanner_f3,
          train_loader, device, wandb_run=None):
    """
    Train scanner_f2 and scanner_f3 on normal data.
    Objective: scanner learns to reconstruct normal SNN features.
    Normal → low reconstruction error.
    Anomaly (at test time) → high reconstruction error = anomaly signal.
    """
    criterion = ScannerLoss(mode=args.loss)
    optimizer = optim.Adam(
        list(scanner_f2.parameters()) + list(scanner_f3.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        scanner_f2.train()
        scanner_f3.train()
        total_loss = 0.0

        pbar = tqdm(train_loader,
                    desc=f'Epoch {epoch+1}/{args.epochs}',
                    disable=os.environ.get('DISABLE_TQDM', '0') == '1')

        for imgs, _ in pbar:
            # Step 1: get SNN firing rate features (no grad, encoder frozen)
            rate_f2, rate_f3 = snn_encode(snn_encoder, imgs, args.timesteps, device)

            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)

            # Step 2: scan f2 across all directions, compute reconstruction loss
            for scan_dir in args.scan_directions:
                _, _, recon_f2 = scan_feature_map(rate_f2, scanner_f2, scan_dir)
                _, _, recon_f3 = scan_feature_map(rate_f3, scanner_f3, scan_dir)
                loss = loss + criterion(recon_f2, rate_f2.detach())
                loss = loss + criterion(recon_f3, rate_f3.detach())

            loss = loss / (2 * len(args.scan_directions))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.6f}')

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.save_dir, f'{args.name}_scanner.pth')
        torch.save({
            'epoch':           epoch + 1,
            'scanner_f2':      scanner_f2.state_dict(),
            'scanner_f3':      scanner_f3.state_dict(),
            'optimizer':       optimizer.state_dict(),
            'train_loss':      avg_loss,
            'args':            vars(args),
        }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.save_dir, f'{args.name}_scanner_best.pth')
            torch.save({
                'epoch':      epoch + 1,
                'scanner_f2': scanner_f2.state_dict(),
                'scanner_f3': scanner_f3.state_dict(),
                'train_loss': avg_loss,
                'args':       vars(args),
            }, best_path)

        # Periodic eval
        if (epoch + 1) % args.eval_every == 0:
            print(f'\n[Eval at epoch {epoch+1}]')
            test_ds = MVTecDataset(args.data_path, args.name, 'test',
                                   img_size=args.img_size)
            img_auc, pix_auc = evaluate(
                args, snn_encoder, scanner_f2, scanner_f3, test_ds, device)
            print(f'  Image AUC: {img_auc:.4f}  Pixel AUC: {pix_auc:.4f}\n')

            if wandb_run:
                wandb_run.log({
                    'train/loss':    avg_loss,
                    'test/img_auc':  img_auc,
                    'test/pix_auc':  pix_auc,
                    'epoch':         epoch + 1,
                })
        else:
            if wandb_run:
                wandb_run.log({'train/loss': avg_loss, 'epoch': epoch + 1})

    print(f'\nTraining done. Best loss: {best_loss:.6f}')
    print(f'Checkpoint: {ckpt_path}')


# ══════════════════════════════════════════════════════
# SECTION 9 — Scoring
# ══════════════════════════════════════════════════════

def score_image(args, snn_encoder, scanner_f2, scanner_f3,
                img_tensor, device):
    """
    Compute anomaly score map for one image.

    Score per spatial position = firing rate deviation (reconstruction error)
                                 weighted by membrane residual.
    Aggregate across scan directions → final score map.

    Full spiking: snn_encoder produces spike-based features,
    scanner processes them token-by-token with membrane context.
    """
    snn_encoder.eval()
    scanner_f2.eval()
    scanner_f3.eval()

    rate_f2, rate_f3 = snn_encode(snn_encoder, img_tensor,
                                   args.timesteps, device)

    score_maps_f2, score_maps_f3 = [], []

    with torch.no_grad():
        for scan_dir in args.scan_directions:
            # f2
            B, C2, H2, W2 = rate_f2.shape
            spike_map_f2 = torch.zeros(H2, W2, device=device)
            vmem_map_f2  = torch.zeros(H2, W2, device=device)
            recon_map_f2 = torch.zeros(B, C2, H2, W2, device=device)
            scanner_f2.reset()
            for (r, c) in SCAN_FNS[scan_dir](H2, W2):
                token = rate_f2[:, :, r:r+1, c:c+1]
                spike, vmem = scanner_f2(token)
                spike_map_f2[r, c] = spike.mean()
                vmem_map_f2[r, c]  = vmem.abs().mean()
                recon_map_f2[:, :, r, c] = spike.squeeze(-1).squeeze(-1)

            # Reconstruction error per pixel (MSE in feature space)
            recon_err_f2 = (recon_map_f2 - rate_f2).pow(2).mean(dim=1).squeeze(0)
            v_norm_f2    = vmem_map_f2 / (vmem_map_f2.max() + 1e-8)
            score_maps_f2.append(recon_err_f2 * (1.0 + v_norm_f2))

            # f3
            B, C3, H3, W3 = rate_f3.shape
            spike_map_f3 = torch.zeros(H3, W3, device=device)
            vmem_map_f3  = torch.zeros(H3, W3, device=device)
            recon_map_f3 = torch.zeros(B, C3, H3, W3, device=device)
            scanner_f3.reset()
            for (r, c) in SCAN_FNS[scan_dir](H3, W3):
                token = rate_f3[:, :, r:r+1, c:c+1]
                spike, vmem = scanner_f3(token)
                spike_map_f3[r, c] = spike.mean()
                vmem_map_f3[r, c]  = vmem.abs().mean()
                recon_map_f3[:, :, r, c] = spike.squeeze(-1).squeeze(-1)

            recon_err_f3 = (recon_map_f3 - rate_f3).pow(2).mean(dim=1).squeeze(0)
            v_norm_f3    = vmem_map_f3 / (vmem_map_f3.max() + 1e-8)
            score_maps_f3.append(recon_err_f3 * (1.0 + v_norm_f3))

    # Aggregate across scan directions
    score_f2 = torch.stack(score_maps_f2).mean(0)  # [H2, W2]
    score_f3 = torch.stack(score_maps_f3).mean(0)  # [H3, W3]

    # Upsample f3 → f2 resolution, then average
    score_f3_up = F.interpolate(score_f3.unsqueeze(0).unsqueeze(0).float(),
                                 size=(H2, W2), mode='bilinear',
                                 align_corners=False).squeeze()
    combined = (score_f2 + score_f3_up) / 2.0  # [H2, W2]

    # Upsample to IMG_SIZE
    score_map = F.interpolate(combined.unsqueeze(0).unsqueeze(0).float(),
                               size=(args.img_size, args.img_size),
                               mode='bilinear', align_corners=False
                               ).squeeze().cpu().numpy()

    return score_map, float(np.max(score_map))


# ══════════════════════════════════════════════════════
# SECTION 10 — Evaluation
# ══════════════════════════════════════════════════════

def evaluate(args, snn_encoder, scanner_f2, scanner_f3, test_ds, device):
    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []

    for i in tqdm(range(len(test_ds)), desc='Evaluating', leave=False):
        img_t, lbl, gt_path = test_ds[i]
        img_t = img_t.unsqueeze(0)

        try:
            score_map, img_score = score_image(
                args, snn_encoder, scanner_f2, scanner_f3, img_t, device)
        except Exception as e:
            print(f'  SKIP {os.path.basename(test_ds.files[i])}: {e}')
            continue

        img_scores.append(img_score)
        img_labels.append(lbl)

        if lbl == 1 and gt_path:
            gt = cv2.imread(gt_path, 0)
            if gt is not None:
                gt = cv2.resize(gt, (args.img_size, args.img_size))
                pix_scores.extend(score_map.flatten().tolist())
                pix_labels.extend((gt > 127).astype(int).flatten().tolist())

    img_auc = roc_auc_score(img_labels, img_scores) \
        if len(set(img_labels)) == 2 else None
    pix_auc = roc_auc_score(pix_labels, pix_scores) \
        if pix_labels and len(set(pix_labels)) == 2 else None

    return img_auc, pix_auc


# ══════════════════════════════════════════════════════
# SECTION 11 — Main
# ══════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir,   exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    print('=' * 60)
    print(f'S3AD  |  category={args.name}  |  mode={args.mode}')
    print(f'Device: {device}  |  T={args.timesteps}  |  scan={args.scan_directions}')
    print('=' * 60)

    # ── WandB ──
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        if args.wandb_key:
            os.environ['WANDB_API_KEY'] = args.wandb_key
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f'{args.name}_T{args.timesteps}_{args.loss}',
                config=vars(args))
        except Exception as e:
            print(f'WandB init failed: {e}')

    # ── Build SNN Encoder ──
    print('\n[1/3] Building SNN encoder (ResNet-18 ANN2SNN)...')
    ann_encoder = ResNetEncoder().to(device)
    ann_encoder.eval()

    # Verify activation bounds
    dummy = torch.randn(2, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        f2d, f3d = ann_encoder(dummy)
    print(f'  ANN layer2 max: {f2d.max().item():.3f}  '
          f'layer3 max: {f3d.max().item():.3f}  (expected < 10)')

    train_ds = MVTecDataset(args.data_path, args.name, 'train',
                             img_size=args.img_size,
                             max_samples=args.calib_samples)
    calib_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=2,
                               collate_fn=collate_train)
    snn_encoder = build_snn_encoder(ann_encoder, calib_loader,
                                     device, mode=args.snn_mode)

    # ── Build Scanners ──
    print('\n[2/3] Building SNN spatial scanners...')
    scanner_f2 = SNNSpatialScanner(
        in_channels=128,
        hidden_channels=args.hidden_channels).to(device)
    scanner_f3 = SNNSpatialScanner(
        in_channels=256,
        hidden_channels=args.hidden_channels).to(device)

    # Resume if requested
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        scanner_f2.load_state_dict(ckpt['scanner_f2'])
        scanner_f3.load_state_dict(ckpt['scanner_f3'])
        print(f'  Resumed from: {args.resume}')

    # ── Train ──
    if args.mode in ('train', 'both'):
        print(f'\n[3a/3] Training scanners on {args.name} normal data...')
        full_train_ds = MVTecDataset(args.data_path, args.name, 'train',
                                      img_size=args.img_size)
        train_loader  = DataLoader(full_train_ds, batch_size=args.batch_size,
                                    shuffle=True, num_workers=2,
                                    collate_fn=collate_train)
        train(args, snn_encoder, scanner_f2, scanner_f3,
              train_loader, device, wandb_run)

    # ── Test ──
    if args.mode in ('test', 'both'):
        print(f'\n[3b/3] Testing on {args.name}...')

        # Load best checkpoint if available
        best_path = os.path.join(args.save_dir, f'{args.name}_scanner_best.pth')
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            scanner_f2.load_state_dict(ckpt['scanner_f2'])
            scanner_f3.load_state_dict(ckpt['scanner_f3'])
            print(f'  Loaded best checkpoint: {best_path}')

        test_ds = MVTecDataset(args.data_path, args.name, 'test',
                                img_size=args.img_size)
        print(f'  Test set: {len(test_ds)} images  '
              f'({sum(test_ds.labels)} anomaly, '
              f'{len(test_ds.labels) - sum(test_ds.labels)} normal)')

        img_auc, pix_auc = evaluate(
            args, snn_encoder, scanner_f2, scanner_f3, test_ds, device)

        ia = f'{img_auc:.4f}' if img_auc is not None else 'N/A'
        pa = f'{pix_auc:.4f}' if pix_auc is not None else 'N/A'
        print(f'\n  Image AUC: {ia}')
        print(f'  Pixel AUC: {pa}')

        # Save result
        out = os.path.join(args.result_dir, f'{args.name}_result.txt')
        with open(out, 'w') as f:
            f.write(f'S3AD Result — {args.name}\n')
            f.write(f'Timesteps: {args.timesteps}\n')
            f.write(f'Scan directions: {args.scan_directions}\n')
            f.write(f'Loss: {args.loss}\n')
            f.write(f'Image AUC: {ia}\n')
            f.write(f'Pixel AUC: {pa}\n')
        print(f'  Result saved: {out}')

        if wandb_run:
            wandb_run.summary['img_auc'] = img_auc
            wandb_run.summary['pix_auc'] = pix_auc

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()