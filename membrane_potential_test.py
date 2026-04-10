"""
membrane_potential_test.py  —  v4 Fixed Inf/NaN Version
------------------------------------------------
Anomaly score = ||v_final||² của SNN encoder sau T timesteps.
Fixes: Weight clamping, BN fusion, and safe normalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from glob import glob
import cv2
import os
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim_fn

from spikingjelly.activation_based import ann2snn, functional
from utils import read_img, get_patch, patch2img
from network import AutoEncoder
from options import Options

# ═══════════════════════════════════════════════
# SECTION 1 — Load & Convert
# ═══════════════════════════════════════════════

class CalibrationDataset(Dataset):
    def __init__(self, files, grayscale, patch_size, max_samples=100):
        self.files = files[:max_samples]
        self.grayscale = grayscale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = read_img(self.files[idx], self.grayscale)
        img = cv2.resize(img, (self.patch_size, self.patch_size))
        img = img / 255.0
        if self.grayscale:
            return torch.FloatTensor(img).unsqueeze(0), 0
        return torch.FloatTensor(img).permute(2, 0, 1), 0


def load_model(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(cfg).to(device)
    ckpt = os.path.join(cfg.chechpoint_dir, 'model.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Not found: {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print(f"Loaded: {ckpt}")
    return model, device

def debug_ann_activations(model_ann, loader, device):
    """
    Chạy thử một batch dữ liệu qua ANN và in thống kê từng lớp.
    Giúp phát hiện lớp nào có giá trị quá nhỏ hoặc quá lớn.
    """
    print("\n🔍 === DIAGNOSING ANN ACTIVATIONS ===")
    model_ann.eval()
    
    # Lấy 1 batch dữ liệu chuẩn
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    
    activations = {}
    
    # Hàm hook để bắt giá trị đầu ra của từng sub-module
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Đăng ký hook cho các lớp trong encoder/decoder (đặc biệt là sau ReLU)
    hooks = []
    for name, module in model_ann.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Chạy forward pass
    with torch.no_grad():
        _ = model_ann(imgs)

    # In thống kê
    print(f"{'Layer Name':<30} | {'Max':>10} | {'Mean':>10} | {'Min':>10} | {'Zeros %':>8}")
    print("-" * 80)
    
    for name, act in activations.items():
        a_max = act.max().item()
        a_mean = act.mean().item()
        a_min = act.min().item()
        zero_pct = (act == 0).float().mean().item() * 100
        
        # Đánh dấu cảnh báo nếu giá trị quá nhỏ (dễ gây bùng nổ SNN weight)
        warning = "⚠️ LOW" if 0 < a_max < 1e-4 else ""
        if a_max == 0: warning = "❌ ALL ZERO"
        if np.isinf(a_max) or np.isnan(a_max): warning = "🚨 INF/NAN"

        print(f"{name:<30} | {a_max:>10.5f} | {a_mean:>10.5f} | {a_min:>10.5f} | {zero_pct:>7.1f}% {warning}")

    # Gỡ bỏ hooks
    for h in hooks:
        h.remove()
    print("=" * 80 + "\n")

def fuse_bn_recursively(model):
    """
    Hàm hỗ trợ gộp BatchNorm. Với các phiên bản cũ, 
    việc gọi model.eval() là quan trọng nhất để cố định running_mean/var.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval() # Đảm bảo BN dùng thông số đã train, không update thêm
    return model

def convert_to_snn(model_ann, cfg, device):
    # 1. Fuse BatchNorm hoặc chuẩn bị chế độ Eval
    model_ann.eval()
    try:
        # Gọi chính cái hàm bạn định nghĩa ở trên (bỏ chữ model_parse.)
        fuse_bn_recursively(model_ann) 
        print("Set BatchNorm layers to Eval mode for fusion.")
    except Exception as e:
        print(f"Fusion hint: {e}")

    # 2. Chuẩn bị decoder
    dec = list(model_ann.decoder.children())
    if isinstance(dec[-1], nn.Sigmoid):
        model_ann.decoder = nn.Sequential(*dec[:-1])
        print("Removed Sigmoid from ANN.")

    # 3. Calibration
    files = glob(cfg.train_data_dir + '/*png')
    loader = DataLoader(
        CalibrationDataset(files, cfg.grayscale, cfg.patch_size),
        batch_size=64, shuffle=False, num_workers=0)

    debug_ann_activations(model_ann, loader, device)
    
    print(f"Calibrating on {min(len(files), 100)} normal samples...")
    # Dùng mode 0.99 cho bản SpikingJelly cũ
    converter = ann2snn.Converter(dataloader=loader, device=device, mode=0.9)
    snn = converter(model_ann)

    # 4. Cleanup weights: Chặn các trọng số bị bùng nổ sau convert
    for name, param in snn.named_parameters():
        if torch.isinf(param).any() or torch.isnan(param).any():
            param.data = torch.nan_to_num(param.data, nan=0.0, posinf=20.0, neginf=-20.0)
        param.data.clamp_(-100, 100) 

    print("ANN2SNN done.")
    return snn


# ═══════════════════════════════════════════════
# SECTION 2 — IFNode inspection
# ═══════════════════════════════════════════════

def get_ifnode_info(model_snn):
    nodes = [(name, mod) for name, mod in model_snn.named_modules()
             if 'IFNode' in str(type(mod))]
    half = len(nodes) // 2
    return nodes[:half], nodes[half:]


def debug_vfinal_stats(encoder_nodes, decoder_nodes):
    print("\n  [debug v_final distribution]")
    for name, mod in encoder_nodes:
        if hasattr(mod, 'v') and mod.v is not None:
            v = mod.v
            print(f"    ENC {name}: mean={v.abs().mean():.4f} "
                  f"max={v.abs().max():.4f} shape={tuple(v.shape)}")
    for name, mod in decoder_nodes:
        if hasattr(mod, 'v') and mod.v is not None:
            v = mod.v
            print(f"    DEC {name}: mean={v.abs().mean():.4f} "
                  f"max={v.abs().max():.4f} shape={tuple(v.shape)}")


# ═══════════════════════════════════════════════
# SECTION 3 — Score computation
# ═══════════════════════════════════════════════

def vfinal_to_scoremap(nodes, target_hw):
    maps = []
    for _, mod in nodes:
        if not hasattr(mod, 'v') or mod.v is None:
            continue
        
        # CHÈN: Giới hạn điện thế màng để tránh inf khi bình phương
        v_safe = torch.clamp(mod.v, min=-50.0, max=50.0)
        
        if v_safe.dim() == 4:                         
            s = v_safe.pow(2).mean(dim=1)             
        elif v_safe.dim() == 2:                       
            s = v_safe.pow(2).mean(dim=1)             
            s = s[:, None, None].expand(-1, *target_hw)
        else:
            continue

        if tuple(s.shape[-2:]) != tuple(target_hw):
            s = F.interpolate(s.unsqueeze(1).float(),
                              size=target_hw, mode='bilinear',
                              align_corners=False).squeeze(1)
        maps.append(s)

    if not maps:
        return None
    return torch.stack(maps, dim=0).mean(dim=0)


def run_snn_get_vfinal(batch, model_snn, encoder_nodes, decoder_nodes,
                       timesteps, use_encoder_only, patch_size):
    functional.reset_net(model_snn)
    recon = None
    with torch.no_grad():
        for t in range(timesteps):
            out = model_snn(batch)
            recon = out if t == 0 else recon + out

    nodes = encoder_nodes if use_encoder_only else encoder_nodes + decoder_nodes
    score = vfinal_to_scoremap(nodes, (patch_size, patch_size))

    if score is None:
        v_map = np.zeros((batch.shape[0], patch_size, patch_size))
    else:
        v_map = score.cpu().numpy()
        if v_map.ndim == 2:
            v_map = v_map[np.newaxis]

    recon = torch.sigmoid(recon / timesteps)
    return v_map, recon


def _recon_to_numpy(recon_t, grayscale):
    if grayscale:
        return recon_t.squeeze(1).cpu().numpy()   
    return recon_t.permute(0,2,3,1).cpu().numpy() 


def _recon_error(orig_np, rec_np, grayscale):
    rec_uint8 = (rec_np * 255).astype("uint8")
    orig_uint8 = orig_np if orig_np.dtype == np.uint8 else (orig_np * 255).astype("uint8")
    if grayscale:
        ssim_map = 1 - ssim_fn(orig_uint8, rec_uint8, win_size=11, full=True)[1]
        l1_map   = np.abs(orig_uint8/255. - rec_uint8/255.)
    else:
        min_dim  = min(orig_uint8.shape[:2])
        win_size = max(3, min(11, min_dim if min_dim%2==1 else min_dim-1))
        ssim_map = ssim_fn(orig_uint8, rec_uint8, win_size=win_size,
                           full=True, channel_axis=2)[1]
        ssim_map = 1 - np.mean(ssim_map, axis=2)
        l1_map   = np.mean(np.abs(orig_uint8/255. - rec_uint8/255.), axis=2)
    return ssim_map + l1_map


def get_score_map(img_path, cfg, model_snn, encoder_nodes, decoder_nodes,
                  timesteps, use_encoder_only, alpha=1.0):
    dev = next(model_snn.parameters()).device
    test_img = read_img(img_path, cfg.grayscale)
    if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
        test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
    if cfg.im_resize != cfg.mask_size:
        tmp = (cfg.im_resize - cfg.mask_size) // 2
        test_img = test_img[tmp:tmp+cfg.mask_size, tmp:tmp+cfg.mask_size]
    norm = test_img / 255.0

    if cfg.grayscale:
        t = torch.FloatTensor(norm).unsqueeze(0).unsqueeze(0).to(dev)
    else:
        t = torch.FloatTensor(norm).permute(2,0,1).unsqueeze(0).to(dev)

    # Patch processing
    if test_img.shape[:2] == (cfg.patch_size, cfg.patch_size):
        v_map, recon_t = run_snn_get_vfinal(t, model_snn, encoder_nodes,
                                            decoder_nodes, timesteps,
                                            use_encoder_only, cfg.patch_size)
        rec_np  = _recon_to_numpy(recon_t, cfg.grayscale)[0]
        err_map = _recon_error(norm, rec_np, cfg.grayscale)
        v_map2d = v_map[0]
    else:
        patches = get_patch(norm, cfg.patch_size, cfg.stride)
        pt = torch.FloatTensor(patches).unsqueeze(1).to(dev) if cfg.grayscale else \
             torch.FloatTensor(patches).permute(0,3,1,2).to(dev)

        all_v, all_rec = [], []
        for i in range(0, len(pt), 32):
            vm, rt = run_snn_get_vfinal(pt[i:i+32], model_snn, encoder_nodes, 
                                        decoder_nodes, timesteps, use_encoder_only, cfg.patch_size)
            all_v.append(vm)
            all_rec.append(_recon_to_numpy(rt, cfg.grayscale))

        all_v = np.concatenate(all_v, axis=0)
        all_rec = np.concatenate(all_rec, axis=0)
        err_patches = np.array([_recon_error(patches[i], all_rec[i], cfg.grayscale) for i in range(len(patches))])

        err_map = patch2img(err_patches[:,:,:,np.newaxis], cfg.im_resize, cfg.patch_size, cfg.stride)[:,:,0]
        v_map2d = patch2img(all_v[:,:,:,np.newaxis], cfg.im_resize, cfg.patch_size, cfg.stride)[:,:,0]

    # Hybrid Weighting (FIXED INF/NAN)
    if alpha > 0:
        v_map2d = np.nan_to_num(v_map2d, nan=0.0, posinf=50.0, neginf=-50.0)
        v_max = v_map2d.max()
        v_norm = v_map2d / v_max if v_max > 1e-6 else np.zeros_like(v_map2d)
        score_map = err_map * (1.0 + alpha * v_norm)
    else:
        score_map = err_map

    # Final cleanup before return
    score_map = np.nan_to_num(score_map, nan=0.0, posinf=1.0, neginf=0.0)
    
    if score_map.shape != (cfg.mask_size, cfg.mask_size):
        score_map = cv2.resize(score_map.astype(np.float32), (cfg.mask_size, cfg.mask_size))

    return test_img, score_map, float(np.max(score_map))

# ═══════════════════════════════════════════════
# SECTION 4 — Evaluation (Evaluation remains largely the same)
# ═══════════════════════════════════════════════

def evaluate(cfg, model_snn, encoder_nodes, decoder_nodes,
             timesteps, use_encoder_only, debug=False, alpha=1.0):
    label = "enc_only" if use_encoder_only else "all_layers"
    print(f"\n  [{label} T={timesteps} a={alpha}]", end=" ", flush=True)

    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []
    gt_dir = cfg.test_dir.replace('test', 'ground_truth')
    has_gt = os.path.exists(gt_dir)
    first_img = True

    file_list = [(p, 0, None) for p in glob(os.path.join(cfg.test_dir, 'good', '*'))]
    for folder in os.listdir(cfg.test_dir):
        if folder == 'good' or not os.path.isdir(os.path.join(cfg.test_dir, folder)): continue
        fpath = os.path.join(cfg.test_dir, folder)
        gt_folder = os.path.join(gt_dir, folder) if has_gt else None
        for p in glob(os.path.join(fpath, '*')):
            file_list.append((p, 1, gt_folder))

    for img_path, lbl, gt_folder in file_list:
        try:
            _, score_map, img_score = get_score_map(img_path, cfg, model_snn, encoder_nodes, 
                                                    decoder_nodes, timesteps, use_encoder_only, alpha=alpha)
            if debug and first_img:
                first_img = False
                debug_vfinal_stats(encoder_nodes, decoder_nodes)

            img_scores.append(img_score)
            img_labels.append(lbl)

            if lbl == 1 and gt_folder and os.path.exists(gt_folder):
                fname = os.path.splitext(os.path.basename(img_path))[0]
                gt_path = None
                for ext in ['.png', '.bmp', '.jpg']:
                    for suf in ['_mask', '']:
                        p = os.path.join(gt_folder, fname + suf + ext)
                        if os.path.exists(p): gt_path = p; break
                    if gt_path: break
                if gt_path:
                    gt = cv2.imread(gt_path, 0)
                    if gt is not None:
                        if gt.shape != score_map.shape:
                            gt = cv2.resize(gt, (score_map.shape[1], score_map.shape[0]))
                        pix_scores.extend(score_map.flatten().tolist())
                        pix_labels.extend((gt > 127).astype(int).flatten().tolist())
        except Exception as e:
            print(f"\n  SKIP {os.path.basename(img_path)}: {e}")

    img_auc = roc_auc_score(img_labels, img_scores) if len(set(img_labels)) == 2 else None
    pix_auc = roc_auc_score(pix_labels, pix_scores) if pix_labels and len(set(pix_labels)) == 2 else None

    ia_str = f"{img_auc:.4f}" if img_auc is not None else "0.0000"
    pa_str = f"{pix_auc:.4f}" if pix_auc is not None else "0.0000"
    print(f"Img={ia_str} Pix={pa_str}")
    return img_auc, pix_auc

def main():
    cfg = Options().parse()
    model_ann, device = load_model(cfg)
    model_snn = convert_to_snn(model_ann, cfg, device)
    encoder_nodes, decoder_nodes = get_ifnode_info(model_snn)

    timesteps_list = [10, 30, 50, 70, 100] # Giảm bớt T để test nhanh
    alpha_list     = [0.0, 1.0, 2.0, 3.0]

    results = []
    for T in timesteps_list:
        row = [T]
        for a in alpha_list:
            ia, pa = evaluate(cfg, model_snn, encoder_nodes, decoder_nodes, T, True, (T==1 and a==0.0), a)
            row.extend([ia, pa])
        results.append(row)

if __name__ == '__main__':
    main()