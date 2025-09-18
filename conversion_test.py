import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim
from glob import glob
import cv2
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from spikingjelly.activation_based import ann2snn, functional
from utils import read_img, get_patch, patch2img
from network import AutoEncoder
from options import Options

def debug_snn_structure(model_snn):
    """Debug SNN internal structure and state"""
    print("\n=== SNN STRUCTURE DEBUG ===")
    membrane_count = 0
    reset_count = 0
    
    for name, module in model_snn.named_modules():
        module_type = str(type(module))
        if hasattr(module, 'v'):  # Membrane potential
            print(f"{name}: has membrane potential (v)")
            membrane_count += 1
        if hasattr(module, 'reset'):  # Reset function  
            print(f"{name}: has reset function")
            reset_count += 1
        if 'IFNode' in module_type:
            print(f"{name}: IFNode - v_threshold={getattr(module, 'v_threshold', 'N/A')}")
    
    print(f"Total modules with membrane potential: {membrane_count}")
    print(f"Total modules with reset function: {reset_count}")
    
    if membrane_count == 0:
        print("⚠️  WARNING: No membrane potentials found - SNN may not have temporal dynamics")
    if reset_count == 0:
        print("⚠️  WARNING: No reset functions found - State may not reset between samples")

class CalibrationDataset(Dataset):
    """Dataset for SNN conversion calibration"""
    
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
            img = torch.FloatTensor(img).unsqueeze(0)
        else:
            img = torch.FloatTensor(img).permute(2, 0, 1)
        return img, torch.tensor(0)

def load_model(cfg):
    """Load trained ANN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(cfg).to(device)
    checkpoint_path = os.path.join(cfg.chechpoint_dir, 'model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded ANN checkpoint: {checkpoint_path}")
    return model, device


def create_calibration_dataloader(cfg):
    """Create dataloader for SNN conversion calibration"""
    good_files = glob(cfg.train_data_dir + '/*png')
    
    if not good_files:
        raise ValueError(f"No training files found in {cfg.train_data_dir}")
    
    dataset = CalibrationDataset(good_files, cfg.grayscale, cfg.patch_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Using {len(dataset)} training samples for calibration")
    return dataloader


def convert_to_snn(model_ann, cfg, device):
    """Pure SpikingJelly ANN to SNN conversion"""
    print("Converting ANN to SNN using pure SpikingJelly...")
    
    # Step 1: Remove Sigmoid layer (SpikingJelly can't handle it)
    decoder_layers = list(model_ann.decoder.children())
    if isinstance(decoder_layers[-1], nn.Sigmoid):
        model_ann.decoder = nn.Sequential(*decoder_layers[:-1])
        print("Removed Sigmoid layer before conversion")
    
    # Step 2: Create calibration dataloader
    calib_dataloader = create_calibration_dataloader(cfg)
    
    # Step 3: Use pure SpikingJelly converter
    converter = ann2snn.Converter(
        dataloader=calib_dataloader, 
        device=device, 
        mode= 0.5,
        momentum=0.1
    )
    
    # Step 4: Convert to SNN - this will replace ReLU with IFNode + VoltageScaler
    model_snn = converter(model_ann)
    print("SpikingJelly ANN to SNN conversion completed")
    
    return model_snn

def calculate_metrics_with_optimal_threshold(all_labels, all_scores):
    """Calculate AUC and optimal threshold from ROC curve"""
    from sklearn.metrics import roc_curve, confusion_matrix
    
    auc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    # Find optimal threshold using Youden's J statistic (maximizes TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    # Calculate TP/FP with optimal threshold
    predictions = (np.array(all_scores) >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    
    return auc, tp, fp, tn, fn, optimal_threshold, optimal_tpr, optimal_fpr


def get_snn_residual_map(img_path, cfg, model_snn, device, timesteps=50, debug_print=False):
    """Get residual map using pure SNN model with spike accumulation"""
    test_img = read_img(img_path, cfg.grayscale)
    functional.reset_net(model_snn)
    # Resize and crop image
    if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
        test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
    if cfg.im_resize != cfg.mask_size:
        tmp = (cfg.im_resize - cfg.mask_size) // 2
        test_img = test_img[tmp:tmp+cfg.mask_size, tmp:tmp+cfg.mask_size]
    
    test_img_norm = test_img / 255.0
    
    if test_img.shape[:2] == (cfg.patch_size, cfg.patch_size):
        # print("Processing single patch")
        decoded_img = process_single_patch_snn(test_img_norm, cfg, model_snn, device, timesteps, debug_print)
    else:
        # print("Processing multiple patches")
        decoded_img = process_multiple_patches_snn(test_img_norm, cfg, model_snn, device, timesteps, debug_print)
    
    # Calculate residual maps
    rec_img = np.reshape((decoded_img * 255.).astype('uint8'), test_img.shape)
    
    if cfg.grayscale:
        ssim_residual_map = 1 - ssim(test_img, rec_img, win_size=11, full=True)[1]
        l1_residual_map = np.abs(test_img / 255. - rec_img / 255.)
    else:
        min_dim = min(test_img.shape[:2])
        win_size = min(11, min_dim if min_dim % 2 == 1 else min_dim - 1)
        win_size = max(3, win_size)
        ssim_residual_map = ssim(test_img, rec_img, win_size=win_size, full=True, channel_axis=2)[1]
        ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
        l1_residual_map = np.mean(np.abs(test_img / 255. - rec_img / 255.), axis=2)
    
    return test_img, rec_img, ssim_residual_map, l1_residual_map


def process_single_patch_snn(test_img_norm, cfg, model_snn, device, timesteps, debug_print=False):
    """Process single patch through SNN with proper spike accumulation"""
    if cfg.grayscale:
        test_tensor = torch.FloatTensor(test_img_norm).unsqueeze(0).unsqueeze(0).to(device)
    else:
        test_tensor = torch.FloatTensor(test_img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Reset SNN state
    functional.reset_net(model_snn)
    with torch.no_grad():
        for t in range(timesteps):
            if t == 0:
                output = model_snn(test_tensor)  # Constant analog input
            else:
                output += model_snn(test_tensor)  # Tích lũy output
    
    # Chia cho timesteps để có firing rate
    decoded_tensor = output / timesteps
    decoded_tensor = torch.sigmoid(decoded_tensor)
    if cfg.grayscale:
        return decoded_tensor.squeeze().cpu().numpy()
    else:
        return decoded_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

def process_multiple_patches_snn(test_img_norm, cfg, model_snn, device, timesteps, debug_print=False):
    """Process multiple patches through SNN"""
    patches = get_patch(test_img_norm, cfg.patch_size, cfg.stride)
    
    if cfg.grayscale:
        patches_tensor = torch.FloatTensor(patches).unsqueeze(1).to(device)
    else:
        patches_tensor = torch.FloatTensor(patches).permute(0, 3, 1, 2).to(device)
    
    batch_size = 8  # Small batch size for SNN
    decoded_patches = []
    
    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i+batch_size]
            
            # Reset SNN state for each batch
            functional.reset_net(model_snn)
            for t in range(timesteps):
                if t == 0:
                    batch_output = model_snn(batch)  # Constant analog input
                else:
                    batch_output += model_snn(batch)  # Tích lũy
            
            # Convert to firing rate
            decoded_batch = batch_output / timesteps
            decoded_batch = torch.sigmoid(decoded_batch)
            
            if cfg.grayscale:
                decoded_batch = decoded_batch.squeeze(1).cpu().numpy()
            else:
                decoded_batch = decoded_batch.permute(0, 2, 3, 1).cpu().numpy()
            
            decoded_patches.append(decoded_batch)
    
    decoded_patches = np.concatenate(decoded_patches, axis=0)
    return patch2img(decoded_patches, cfg.im_resize, cfg.patch_size, cfg.stride)
def calculate_pixel_auc(cfg, model_snn, device, timesteps=80):
    """Calculate Pixel-level AUC using ground truth masks - Follow test.py logic"""
    print('Calculating SNN Pixel-level AUC...')
    
    # Check if ground truth folder exists  
    gt_dir = cfg.test_dir.replace('test', 'ground_truth')
    if not os.path.exists(gt_dir):
        print(f"Warning: Ground truth directory not found: {gt_dir}")
        print("Pixel-level AUC requires ground truth masks")
        return None
    
    all_pixel_scores = []
    all_pixel_labels = []
    
    # Process defective samples only (good samples don't have GT masks)
    defect_folders = [folder for folder in os.listdir(cfg.test_dir) 
                     if folder != 'good' and os.path.isdir(os.path.join(cfg.test_dir, folder))]
    
    processed_count = 0
    for folder in defect_folders:
        test_files = glob(os.path.join(cfg.test_dir, folder, '*'))
        gt_folder_path = os.path.join(gt_dir, folder)
        
        if not os.path.exists(gt_folder_path):
            continue
            
        for test_path in test_files:
            # Find corresponding ground truth mask - SAME AS test.py
            filename = os.path.splitext(os.path.basename(test_path))[0]
            possible_gt_extensions = ['.png', '.bmp', '.jpg', '.jpeg']
            gt_path = None
            
            for ext in possible_gt_extensions:
                potential_gt_path = os.path.join(gt_folder_path, filename + '_mask' + ext)
                if os.path.exists(potential_gt_path):
                    gt_path = potential_gt_path
                    break
                # Try without '_mask' suffix
                potential_gt_path = os.path.join(gt_folder_path, filename + ext)
                if os.path.exists(potential_gt_path):
                    gt_path = potential_gt_path
                    break
            
            if gt_path is None:
                continue
                
            try:
                # Get residual maps using SNN
                _, _, ssim_res, l1_res = get_snn_residual_map(test_path, cfg, model_snn, device, timesteps)
                combined_score = ssim_res + l1_res  # ✓ SAME AS test.py
                
                # Load and process ground truth - SAME AS test.py
                gt_mask = cv2.imread(gt_path, 0)
                if gt_mask is None:
                    continue
                    
                # Resize GT mask to match residual map size
                if gt_mask.shape != combined_score.shape:
                    gt_mask = cv2.resize(gt_mask, (combined_score.shape[1], combined_score.shape[0]))
                
                # Binarize ground truth (threshold at 127) - SAME AS test.py
                gt_binary = (gt_mask > 127).astype(int)
                
                # Flatten and add to arrays - SAME AS test.py
                all_pixel_scores.extend(combined_score.flatten())
                all_pixel_labels.extend(gt_binary.flatten())
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {test_path}: {e}")
    
    if processed_count == 0:
        print("Warning: No valid image-mask pairs found for pixel-level AUC")
        return None
    
    if len(set(all_pixel_labels)) < 2:
        print("Warning: Need both normal and defective pixels for AUC calculation")
        return None

    # Calculate metrics with TP/FP
    pixel_auc, tp, fp, tn, fn, threshold, tpr, fpr = calculate_metrics_with_optimal_threshold(all_pixel_labels, all_pixel_scores)
    print(f'SNN Pixel-level AUC: {pixel_auc:.4f}')
    print(f'Optimal threshold: {threshold:.6f}')
    print(f'At optimal threshold - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    print(f'TPR (Sensitivity): {tpr:.4f}, FPR: {fpr:.4f}')
    print(f'Processed {processed_count} images')
    
    return pixel_auc, tp, fp, tn, fn, tpr, fpr

def calculate_image_auc(cfg, model_snn, device, timesteps=50, debug_print=False):
    """Calculate Image-level AUC for SNN model"""
    print(f'Calculating SNN Image-level AUC with {timesteps} timesteps...')
    
    all_scores = []
    all_labels = []
    processed_count = 0
    
    # Process good samples
    good_files = glob(os.path.join(cfg.test_dir, 'good', '*'))
    for img_path in good_files:
        try:
            debug_this = debug_print and processed_count == 0  # Only debug first image
            _, _, ssim_res, l1_res = get_snn_residual_map(img_path, cfg, model_snn, device, timesteps, debug_this)
            score = np.max(ssim_res + l1_res)
            all_scores.append(score)
            all_labels.append(0)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process defective samples
    defect_folders = [folder for folder in os.listdir(cfg.test_dir) 
                     if folder != 'good' and os.path.isdir(os.path.join(cfg.test_dir, folder))]
    
    for folder in defect_folders:
        defect_files = glob(os.path.join(cfg.test_dir, folder, '*'))
        for img_path in defect_files:
            try:
                _, _, ssim_res, l1_res = get_snn_residual_map(img_path, cfg, model_snn, device, timesteps)
                score = np.max(ssim_res + l1_res)
                all_scores.append(score)
                all_labels.append(1)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    if len(set(all_labels)) < 2:
        print("Warning: Need both normal and defective samples for AUC calculation")
        return None, None, None, None, None
    
    # Calculate metrics with TP/FP
    image_auc, tp, fp, tn, fn, threshold, tpr, fpr = calculate_metrics_with_optimal_threshold(all_labels, all_scores)
    
    print(f'SNN Image-level AUC: {image_auc:.4f}')
    print(f'Optimal threshold: {threshold:.6f}')
    print(f'At optimal threshold - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    print(f'TPR (Sensitivity): {tpr:.4f}, FPR: {fpr:.4f}')
    print(f'Processed {len(all_scores)} samples (Good: {all_labels.count(0)}, Defect: {all_labels.count(1)})')
    
    return image_auc, tp, fp, tn, fn, tpr, fpr


def verify_snn_conversion(model_snn):
    """Verify if SNN conversion was successful"""
    print("\n=== SNN CONVERSION VERIFICATION ===")
    
    ifnode_count = 0
    voltage_scaler_count = 0
    
    for name, module in model_snn.named_modules():
        module_type = str(type(module))
        if 'IFNode' in module_type:
            print(f"Found IFNode: {name}")
            ifnode_count += 1
        elif 'VoltageScaler' in module_type:
            print(f"Found VoltageScaler: {name}")
            voltage_scaler_count += 1
    
    print(f"Total IFNodes: {ifnode_count}")
    print(f"Total VoltageScalers: {voltage_scaler_count}")
    
    success = ifnode_count > 0
    if success:
        print("✓ SNN conversion successful: Found spiking neurons")
    else:
        print("✗ SNN conversion failed: No spiking neurons found")
    
    return success


def test_timestep_effect():
    """Test how different timesteps affect SNN performance"""
    cfg = Options().parse()
    
    print("=" * 60)
    print("PURE SPIKINGJELLY ANN2SNN CONVERSION TEST")
    print("=" * 60)
    
    # Load and convert model
    model_ann, device = load_model(cfg)
    model_snn = convert_to_snn(model_ann, cfg, device)
    
    # Verify conversion
    if not verify_snn_conversion(model_snn):
        print("SNN conversion failed. Exiting.")
        return
    debug_snn_structure(model_snn)
    
    # Test different timesteps
    timesteps_to_test = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = []
    
    print(f"\n=== TESTING DIFFERENT TIMESTEPS ===")
    
    for T in timesteps_to_test:
        print(f"\n--- Testing T={T} timesteps ---")
        try:
            # Enable debug for first timestep only
            debug_print = True
            
            # Calculate both image and pixel AUC with TP/FP
            image_results = calculate_image_auc(cfg, model_snn, device, T, debug_print)
            pixel_results = calculate_pixel_auc(cfg, model_snn, device, T)
            
            if image_results[0] is not None:
                img_auc, img_tp, img_fp, img_tn, img_fn, img_tpr, img_fpr = image_results
                if pixel_results[0] is not None:
                    pix_auc, pix_tp, pix_fp, pix_tn, pix_fn, pix_tpr, pix_fpr = pixel_results
                    results.append((T, img_auc, pix_auc, img_tp, img_fp, img_tpr, img_fpr, pix_tp, pix_fp, pix_tpr, pix_fpr))
                else:
                    results.append((T, img_auc, None, img_tp, img_fp, img_tpr, img_fpr, None, None, None, None))
        except Exception as e:
            print(f"T={T}: Error - {e}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"TIMESTEP ANALYSIS RESULTS {cfg.name}")
    print("="*50)

    if results:
        print(f"{'T':>3} | {'Img AUC':>8} | {'TP/FP':>6} | {'TPR/FPR':>9} | {'Pix AUC':>8} | {'TP/FP':>10} | {'TPR/FPR':>9}")
        print("-" * 90)
        
        for result in results:
            if len(result) == 11:  # T, img_auc, pix_auc, img_tp, img_fp, img_tpr, img_fpr, pix_tp, pix_fp, pix_tpr, pix_fpr
                T, img_auc, pix_auc, img_tp, img_fp, img_tpr, img_fpr, pix_tp, pix_fp, pix_tpr, pix_fpr = result
                
                img_auc_str = f"{img_auc:.4f}" if img_auc is not None else "N/A"
                img_tp_fp = f"{img_tp}/{img_fp}" if img_tp is not None else "N/A"
                img_tpr_fpr = f"{img_tpr:.3f}/{img_fpr:.3f}" if img_tpr is not None else "N/A"
                
                if pix_auc is not None:
                    pix_auc_str = f"{pix_auc:.4f}"
                    # Format large numbers for pixel-level
                    pix_tp_str = f"{pix_tp//1000}k" if pix_tp > 1000 else str(pix_tp)
                    pix_fp_str = f"{pix_fp//1000}k" if pix_fp > 1000 else str(pix_fp)
                    pix_tp_fp = f"{pix_tp_str}/{pix_fp_str}"
                    pix_tpr_fpr = f"{pix_tpr:.3f}/{pix_fpr:.3f}"
                else:
                    pix_auc_str = "N/A"
                    pix_tp_fp = "N/A"
                    pix_tpr_fpr = "N/A"
                
                print(f"{T:3d} | {img_auc_str:>8} | {img_tp_fp:>6} | {img_tpr_fpr:>9} | {pix_auc_str:>8} | {pix_tp_fp:>10} | {pix_tpr_fpr:>9}")
            elif len(result) == 7:  # Backward compatibility
                T, img_auc, pix_auc, img_tp, img_fp, pix_tp, pix_fp = result
                print(f"{T:3d} | {img_auc:.4f} | {img_tp}/{img_fp} | N/A | {pix_auc:.4f} if pix_auc else 'N/A' | {pix_tp//1000}k/{pix_fp//1000}k | N/A")
            else:
                print(f"T={result[0]:3d} | Unexpected format ({len(result)} values)")
        
        # Find best results
        valid_results = [r for r in results if len(r) >= 2 and r[1] is not None]
        if valid_results:
            best_T, best_auc = max(valid_results, key=lambda x: x[1])[:2]
            print(f"\nBest Image AUC: T={best_T} with AUC={best_auc:.4f}")
            
            if len(valid_results[0]) > 2 and any(r[2] is not None for r in valid_results):
                best_pix_T, _, best_pix_auc = max([r for r in valid_results if r[2] is not None], key=lambda x: x[2])[:3]
                print(f"Best Pixel AUC: T={best_pix_T} with AUC={best_pix_auc:.4f}")
    else:
        print("No valid results obtained")


if __name__ == '__main__':
    test_timestep_effect()