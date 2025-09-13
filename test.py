import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage import morphology 
from glob import glob
import cv2
import os
import re
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from utils import read_img, get_patch, patch2img, set_img_color, bg_mask
from network import AutoEncoder
from options import Options


def load_model_for_testing(cfg):
    """Load trained model for testing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(cfg).to(device)
    
    if cfg.weight_file:
        checkpoint_path = os.path.join(cfg.chechpoint_dir, cfg.weight_file)
    else:
        # Find latest checkpoint
        file_list = os.listdir(cfg.chechpoint_dir)
        pth_files = [f for f in file_list if f.endswith('.pth')]
        if not pth_files:
            raise ValueError("No checkpoint files found")
        
        # Extract epochs from filenames (format: XX-loss.pth)
        latest_epoch = max([int(f.split('-')[0]) for f in pth_files])
        print('Loading latest weight file: ', latest_epoch)
        
        checkpoint_path = None
        for f in pth_files:
            if f.startswith(f"{latest_epoch:02d}-"):
                checkpoint_path = os.path.join(cfg.chechpoint_dir, f)
                break
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model, device


def get_residual_map(img_path, cfg, model, device):
    test_img = read_img(img_path, cfg.grayscale)

    if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
        test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
    if cfg.im_resize != cfg.mask_size:
        tmp = (cfg.im_resize - cfg.mask_size)//2
        test_img = test_img[tmp:tmp+cfg.mask_size, tmp:tmp+cfg.mask_size]

    test_img_ = test_img / 255.

    if test_img.shape[:2] == (cfg.patch_size, cfg.patch_size):
        # Single patch
        if cfg.grayscale:
            test_tensor = torch.FloatTensor(test_img_).unsqueeze(0).unsqueeze(0).to(device)
        else:
            test_tensor = torch.FloatTensor(test_img_).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            decoded_tensor = model(test_tensor)
        
        if cfg.grayscale:
            decoded_img = decoded_tensor.squeeze().cpu().numpy()
        else:
            decoded_img = decoded_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        # Multiple patches
        patches = get_patch(test_img_, cfg.patch_size, cfg.stride)
        
        # Convert patches to tensor
        if cfg.grayscale:
            patches_tensor = torch.FloatTensor(patches).unsqueeze(1).to(device)
        else:
            patches_tensor = torch.FloatTensor(patches).permute(0, 3, 1, 2).to(device)
        
        # Process in batches to avoid memory issues
        batch_size = 32
        decoded_patches = []
        with torch.no_grad():
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i:i+batch_size]
                decoded_batch = model(batch)
                
                if cfg.grayscale:
                    decoded_batch = decoded_batch.squeeze(1).cpu().numpy()
                else:
                    decoded_batch = decoded_batch.permute(0, 2, 3, 1).cpu().numpy()
                
                decoded_patches.append(decoded_batch)
        
        decoded_patches = np.concatenate(decoded_patches, axis=0)
        decoded_img = patch2img(decoded_patches, cfg.im_resize, cfg.patch_size, cfg.stride)

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


def calculate_image_auc(cfg, model, device):
    """Calculate Image-level AUC"""
    print('Calculating Image-level AUC...')
    
    all_scores = []
    all_labels = []
    
    # Process good samples (label = 0)
    good_files = glob(os.path.join(cfg.test_dir, 'good', '*'))
    if not good_files:
        print("Warning: No good samples found for AUC calculation")
    
    for img_path in good_files:
        try:
            _, _, ssim_res, l1_res = get_residual_map(img_path, cfg, model, device)
            # Image-level anomaly score: max of residual maps
            score = np.max(ssim_res + l1_res)
            all_scores.append(score)
            all_labels.append(0)  # Normal
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process defective samples (label = 1)
    defect_folders = [folder for folder in os.listdir(cfg.test_dir) 
                     if folder != 'good' and os.path.isdir(os.path.join(cfg.test_dir, folder))]
    
    for folder in defect_folders:
        defect_files = glob(os.path.join(cfg.test_dir, folder, '*'))
        for img_path in defect_files:
            try:
                _, _, ssim_res, l1_res = get_residual_map(img_path, cfg, model, device)
                score = np.max(ssim_res + l1_res)
                all_scores.append(score)
                all_labels.append(1)  # Defective
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    if len(set(all_labels)) < 2:
        print("Warning: Need both normal and defective samples for AUC calculation")
        return None, None, None, None
    
    # Calculate AUC
    image_auc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    print(f'Image-level AUC: {image_auc:.4f}')
    
    return image_auc, fpr, tpr, thresholds


def calculate_pixel_auc(cfg, model, device):
    """Calculate Pixel-level AUC using ground truth masks"""
    print('Calculating Pixel-level AUC...')
    
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
            # Find corresponding ground truth mask
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
                # Get residual maps
                _, _, ssim_res, l1_res = get_residual_map(test_path, cfg, model, device)
                combined_score = ssim_res + l1_res
                
                # Load and process ground truth
                gt_mask = cv2.imread(gt_path, 0)
                if gt_mask is None:
                    continue
                    
                # Resize GT mask to match residual map size
                if gt_mask.shape != combined_score.shape:
                    gt_mask = cv2.resize(gt_mask, (combined_score.shape[1], combined_score.shape[0]))
                
                # Binarize ground truth (threshold at 127)
                gt_binary = (gt_mask > 127).astype(int)
                
                # Flatten and add to arrays
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
    
    # Calculate pixel-level AUC
    pixel_auc = roc_auc_score(all_pixel_labels, all_pixel_scores)
    print(f'Pixel-level AUC: {pixel_auc:.4f} (processed {processed_count} images)')
    
    return pixel_auc

def calculate_threshold_accuracy(cfg, model, device):
    """Calculate classification accuracy với threshold"""
    correct = 0
    total = 0
    
    # Test good samples (should be classified as OK)
    good_files = glob(os.path.join(cfg.test_dir, 'good', '*'))
    for img_path in good_files:
        _, _, ssim_res, l1_res = get_residual_map(img_path, cfg, model, device)
        ssim_res *= cfg.depr_mask
        l1_res *= cfg.depr_mask
        
        # Check if ANY pixel exceeds threshold
        is_anomaly = (np.any(ssim_res > cfg.ssim_threshold) or 
                     np.any(l1_res > cfg.l1_threshold))
        
        if not is_anomaly:  # Correctly classified as OK
            correct += 1
        total += 1
    
    good_accuracy = correct / len(good_files) * 100
    
    # Test defective samples (should be classified as NOK)
    defect_count = 0
    defect_correct = 0
    defect_folders = ['color', 'cut', 'fold', 'glue', 'poke']  # leather defects
    
    for folder in defect_folders:
        if os.path.exists(os.path.join(cfg.test_dir, folder)):
            defect_files = glob(os.path.join(cfg.test_dir, folder, '*'))
            for img_path in defect_files:
                _, _, ssim_res, l1_res = get_residual_map(img_path, cfg, model, device)
                ssim_res *= cfg.depr_mask
                l1_res *= cfg.depr_mask
                
                is_anomaly = (np.any(ssim_res > cfg.ssim_threshold) or 
                             np.any(l1_res > cfg.l1_threshold))
                
                if is_anomaly:  # Correctly classified as NOK
                    defect_correct += 1
                defect_count += 1
    
    defect_accuracy = defect_correct / defect_count * 100 if defect_count > 0 else 0
    overall_accuracy = (correct + defect_correct) / (total + defect_count) * 100
    
    print(f'OK Accuracy: {good_accuracy:.1f}% ({correct}/{len(good_files)})')
    print(f'NOK Accuracy: {defect_accuracy:.1f}% ({defect_correct}/{defect_count})')
    print(f'Overall Accuracy: {overall_accuracy:.1f}%')
    
    return good_accuracy, defect_accuracy, overall_accuracy

def plot_roc_curves(cfg, image_auc_data, save_dir):
    """Plot and save ROC curves"""
    if image_auc_data[0] is None:
        return
        
    image_auc, fpr, tpr, thresholds = image_auc_data
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Image AUC = {image_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {cfg.name.upper()}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'ROC curve saved to: {roc_path}')


def get_threshold(cfg, model, device):
    print('Estimating threshold...')
    valid_good_list = glob(cfg.train_data_dir + '/*png')
    num_valid_data = int(np.ceil(len(valid_good_list) * 0.2))
    total_rec_ssim, total_rec_l1 = [], []
    
    for img_path in valid_good_list[-num_valid_data:]:
        _, _, ssim_residual_map, l1_residual_map = get_residual_map(img_path, cfg, model, device)
        total_rec_ssim.append(ssim_residual_map)
        total_rec_l1.append(l1_residual_map)
    
    total_rec_ssim = np.array(total_rec_ssim)
    total_rec_l1 = np.array(total_rec_l1)
    ssim_threshold = float(np.percentile(total_rec_ssim, [cfg.percent]))
    l1_threshold = float(np.percentile(total_rec_l1, [cfg.percent]))
    print('ssim_threshold: %f, l1_threshold: %f' %(ssim_threshold, l1_threshold))
    
    if not cfg.ssim_threshold:
        cfg.ssim_threshold = ssim_threshold
    if not cfg.l1_threshold:
        cfg.l1_threshold = l1_threshold


def get_depressing_mask(cfg):
    depr_mask = np.ones((cfg.mask_size, cfg.mask_size)) * 0.2
    depr_mask[5:cfg.mask_size-5, 5:cfg.mask_size-5] = 1
    cfg.depr_mask = depr_mask


def get_results(file_list, cfg, model, device):
    for img_path in file_list:
        img_name = img_path.split(os.sep)[-1][:-4]
        c = '' if not cfg.sub_folder else k
        test_img, rec_img, ssim_residual_map, l1_residual_map = get_residual_map(img_path, cfg, model, device)

        ssim_residual_map *= cfg.depr_mask
        if 'ssim' in cfg.loss:
            l1_residual_map *= cfg.depr_mask

        mask = np.zeros((cfg.mask_size, cfg.mask_size))
        mask[ssim_residual_map > cfg.ssim_threshold] = 1
        mask[l1_residual_map > cfg.l1_threshold] = 1
        
        if cfg.bg_mask == 'B':
            bg_m = bg_mask(test_img.copy(), 50, cv2.THRESH_BINARY, cfg.grayscale)
            mask *= bg_m
        elif cfg.bg_mask == 'W':
            bg_m = bg_mask(test_img.copy(), 200, cv2.THRESH_BINARY_INV, cfg.grayscale)
            mask *= bg_m
            
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255

        vis_img = set_img_color(test_img.copy(), mask, weight_foreground=0.3, grayscale=cfg.grayscale)

        cv2.imwrite(cfg.save_dir+'/'+c+'_'+img_name+'_residual.png', mask)
        cv2.imwrite(cfg.save_dir+'/'+c+'_'+img_name+'_origin.png', test_img)
        cv2.imwrite(cfg.save_dir+'/'+c+'_'+img_name+'_rec.png', rec_img)
        cv2.imwrite(cfg.save_dir+'/'+c+'_'+img_name+'_visual.png', vis_img)


if __name__ == '__main__':
    cfg = Options().parse()
    
    # Load model
    model, device = load_model_for_testing(cfg)
    print(f"Model loaded on device: {device}")
    
    # Get threshold if not provided
    if not cfg.ssim_threshold or not cfg.l1_threshold:
        get_threshold(cfg, model, device)

    # Get depressing mask
    get_depressing_mask(cfg)

    # ==================== AUC EVALUATION ====================
    print("\n" + "="*50)
    print("AUC EVALUATION")
    print("="*50)
    
    # Calculate Image-level AUC
    image_auc_data = calculate_image_auc(cfg, model, device)
    
    # Calculate Pixel-level AUC
    pixel_auc = calculate_pixel_auc(cfg, model, device)
    
    # Plot ROC curve
    if image_auc_data[0] is not None:
        plot_roc_curves(cfg, image_auc_data, cfg.save_dir)
    
    # Save AUC results to file
    auc_results_path = os.path.join(cfg.save_dir, 'auc_results.txt')
    with open(auc_results_path, 'w') as f:
        f.write(f"Dataset: {cfg.name}\n")
        f.write(f"Model: {cfg.loss}\n")
        f.write(f"Patch size: {cfg.patch_size}\n")
        f.write(f"Z dimension: {cfg.z_dim}\n")
        f.write(f"Grayscale: {cfg.grayscale}\n")
        f.write("-" * 30 + "\n")
        if image_auc_data[0] is not None:
            f.write(f"Image-level AUC: {image_auc_data[0]:.4f}\n")
        else:
            f.write("Image-level AUC: N/A\n")
        if pixel_auc is not None:
            f.write(f"Pixel-level AUC: {pixel_auc:.4f}\n")
        else:
            f.write("Pixel-level AUC: N/A (requires ground truth masks)\n")
    
    print(f"AUC results saved to: {auc_results_path}")
    
    # ==================== ORIGINAL TESTING ====================
    print("\n" + "="*50)
    print("THRESHOLD-BASED EVALUATION")
    print("="*50)
    accuracy_results = calculate_threshold_accuracy(cfg, model, device)

    # Process test images (original functionality)
    if cfg.sub_folder:
        for k in cfg.sub_folder:
            test_list = glob(cfg.test_dir+'/'+k+'/*')
            print(f"Processing {k}: {len(test_list)} images")
            get_results(test_list, cfg, model, device)
    else:
        test_list = glob(cfg.test_dir+'/*')
        print(f"Processing: {len(test_list)} images")
        get_results(test_list, cfg, model, device)
    
    print("Testing completed!")
    print(f"Results saved to: {cfg.save_dir}")