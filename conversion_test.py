import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim
from skimage import morphology 
from glob import glob
import cv2
import os
import re
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from spikingjelly.activation_based import ann2snn, neuron, functional
from utils import read_img, get_patch, patch2img, set_img_color, bg_mask
from network import AutoEncoder
from options import Options


class MembraneOutputLayer(nn.Module):
    """Custom layer that returns membrane potential instead of spikes"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, tau=2.0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.neuron = neuron.LIFNode(tau=tau, detach_reset=False)
        
    def forward(self, x):
        x = self.conv(x)
        # Debug input to neuron
        print(f"DEBUG: Conv input to neuron range: [{x.min():.4f}, {x.max():.4f}]")
        
        # Check if neuron state changes between calls
        v_before = self.neuron.v.clone() if hasattr(self.neuron, 'v') and self.neuron.v is not None else torch.zeros_like(x)
        print(f"DEBUG: Membrane before update: [{v_before.min():.4f}, {v_before.max():.4f}]")
        
        # Run LIF dynamics but return membrane potential
        spike_output = self.neuron(x)  # This should update v and return spikes
        print(f"DEBUG: Spike output range: [{spike_output.min():.4f}, {spike_output.max():.4f}]")
        print(f"DEBUG: Spike count: {spike_output.sum().item()}")
        
        v_output = self.neuron.v
        print(f"DEBUG: Membrane after update: [{v_output.min():.4f}, {v_output.max():.4f}]")
        
        # Check if membrane potential actually changed
        v_diff = torch.abs(v_output - v_before).max()
        print(f"DEBUG: Membrane change: {v_diff:.6f}")
        
        return v_output


class CalibrationDataset(Dataset):
    """Dataset for calibrating SNN conversion"""
    def __init__(self, files, grayscale, patch_size):
        self.files = files
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
        return img, torch.tensor(0)  # Dummy label


def load_model_for_conversion(cfg):
    """Load trained ANN model for conversion"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ann = AutoEncoder(cfg).to(device)
    
    # Load model checkpoint
    checkpoint_path = os.path.join(cfg.chechpoint_dir, 'model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_ann.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded ANN checkpoint: {checkpoint_path}")
    else:
        raise ValueError(f"Model file not found: {checkpoint_path}")
    
    model_ann.eval()
    return model_ann, device


def create_calibration_dataloader(cfg):
    """Create dataloader for SNN conversion calibration"""
    # Use training good samples for calibration
    good_files = glob(cfg.train_data_dir + '/*png')
    
    if len(good_files) == 0:
        raise ValueError(f"No training files found in {cfg.train_data_dir}")
    
    # Limit to reasonable number for calibration
    if len(good_files) > 100:
        good_files = good_files[:100]
    
    print(f"Using {len(good_files)} training samples for calibration")
    
    dataset = CalibrationDataset(good_files, cfg.grayscale, cfg.patch_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    return dataloader


def convert_ann_to_snn(cfg, model_ann, device):
    """Convert ANN model to SNN using SpikingJelly"""
    print("Converting ANN to SNN...")
    
    # Remove Sigmoid layer before conversion
    original_decoder = list(model_ann.decoder.children())
    if isinstance(original_decoder[-1], nn.Sigmoid):
        model_ann.decoder = nn.Sequential(*original_decoder[:-1])
        print("✓ Removed Sigmoid layer before conversion")
    
    # Create calibration dataloader
    calib_dataloader = create_calibration_dataloader(cfg)
    
    # Convert to SNN using correct SpikingJelly syntax
    converter = ann2snn.Converter(
        dataloader=calib_dataloader, 
        device=device, 
        mode='99.9%',
        momentum=0.1
    )
    
    model_snn = converter(model_ann)
    print("✓ Basic ANN to SNN conversion completed")
    
    # Replace final layer with custom membrane output layer
    decoder_module = model_snn.decoder
    decoder_layers = list(decoder_module.children())
    
    # Find and replace the last layer
    last_layer = decoder_layers[-1]
    if hasattr(last_layer, 'conv'):  # SpikingJelly converted layer
        orig_conv = last_layer.conv
        in_channels = orig_conv.in_channels
        out_channels = orig_conv.out_channels
        kernel_size = orig_conv.kernel_size
        stride = orig_conv.stride
        padding = orig_conv.padding
        
        # Create custom membrane output layer
        membrane_layer = MembraneOutputLayer(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            tau=2.0
        )
        
        # Copy weights from original layer
        membrane_layer.conv.load_state_dict(orig_conv.state_dict())
        
        # Replace final layer
        decoder_layers[-1] = membrane_layer
        model_snn.decoder = nn.Sequential(*decoder_layers)
        
        print("✓ Replaced final layer with membrane output layer")
    
    return model_snn


def get_snn_residual_map(img_path, cfg, model_snn, device, timesteps=50):
    """Get residual map using SNN model"""
    test_img = read_img(img_path, cfg.grayscale)

    if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
        test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
    if cfg.im_resize != cfg.mask_size:
        tmp = (cfg.im_resize - cfg.mask_size)//2
        test_img = test_img[tmp:tmp+cfg.mask_size, tmp:tmp+cfg.mask_size]

    test_img_ = test_img / 255.

    if test_img.shape[:2] == (cfg.patch_size, cfg.patch_size):
        # Single patch processing
        if cfg.grayscale:
            test_tensor = torch.FloatTensor(test_img_).unsqueeze(0).unsqueeze(0).to(device)
        else:
            test_tensor = torch.FloatTensor(test_img_).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Reset SNN state
        functional.reset_net(model_snn)
        
        # Run SNN for multiple timesteps
        outputs = []
        print(f"DEBUG: Running SNN with {timesteps} timesteps")
        with torch.no_grad():
            for t in range(timesteps):
                output = model_snn(test_tensor)
                outputs.append(output)
                if t < 3:  # Print first 3 timesteps
                    print(f"DEBUG: Timestep {t} output range: [{output.min():.4f}, {output.max():.4f}]")

        # Average outputs over time (rate coding)
        decoded_tensor = torch.stack(outputs).mean(dim=0)
        print(f"DEBUG: Final averaged output range: [{decoded_tensor.min():.4f}, {decoded_tensor.max():.4f}]")

        # Check if outputs are actually different
        output_variance = torch.stack(outputs).var(dim=0).mean()
        print(f"DEBUG: Output variance across timesteps: {output_variance:.6f}")
        
        # Apply sigmoid to get [0,1] range
        decoded_tensor = torch.sigmoid(decoded_tensor)
        
        if cfg.grayscale:
            decoded_img = decoded_tensor.squeeze().cpu().numpy()
        else:
            decoded_img = decoded_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        # Multiple patches processing
        patches = get_patch(test_img_, cfg.patch_size, cfg.stride)
        
        if cfg.grayscale:
            patches_tensor = torch.FloatTensor(patches).unsqueeze(1).to(device)
        else:
            patches_tensor = torch.FloatTensor(patches).permute(0, 3, 1, 2).to(device)
        
        # Process in batches
        batch_size = 8  # Smaller batch size for SNN
        decoded_patches = []
        
        with torch.no_grad():
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i:i+batch_size]
                
                # Reset SNN state for each batch
                functional.reset_net(model_snn)
                
                # Run SNN for multiple timesteps
                batch_outputs = []
                for t in range(timesteps):
                    output = model_snn(batch)
                    batch_outputs.append(output)
                
                # Average over time
                decoded_batch = torch.stack(batch_outputs).mean(dim=0)
                decoded_batch = torch.sigmoid(decoded_batch)
                
                if cfg.grayscale:
                    decoded_batch = decoded_batch.squeeze(1).cpu().numpy()
                else:
                    decoded_batch = decoded_batch.permute(0, 2, 3, 1).cpu().numpy()
                
                decoded_patches.append(decoded_batch)
        
        decoded_patches = np.concatenate(decoded_patches, axis=0)
        decoded_img = patch2img(decoded_patches, cfg.im_resize, cfg.patch_size, cfg.stride)

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


def calculate_snn_image_auc(cfg, model_snn, device, timesteps=4):
    """Calculate Image-level AUC for SNN model"""
    print('Calculating SNN Image-level AUC...')
    
    all_scores = []
    all_labels = []
    
    # Process good samples (label = 0)
    good_files = glob(os.path.join(cfg.test_dir, 'good', '*'))
    if not good_files:
        print("Warning: No good samples found for AUC calculation")
    
    for img_path in good_files:
        try:
            _, _, ssim_res, l1_res = get_snn_residual_map(img_path, cfg, model_snn, device, timesteps)
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
                _, _, ssim_res, l1_res = get_snn_residual_map(img_path, cfg, model_snn, device, timesteps)
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
    
    print(f'SNN Image-level AUC: {image_auc:.4f}')
    print(f"DEBUG: Processing with T={timesteps}")  # ADD THIS
    print(f"DEBUG: Total samples processed: {len(all_scores)}")  # ADD THIS
    print(f"DEBUG: Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")  # ADD THIS
    return image_auc, fpr, tpr, thresholds

def calculate_snn_pixel_auc(cfg, model_snn, device, timesteps=80):
    """Calculate Pixel-level AUC for SNN model using ground truth masks"""
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
                _, _, ssim_res, l1_res = get_snn_residual_map(test_path, cfg, model_snn, device, timesteps)
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
    print(f'SNN Pixel-level AUC: {pixel_auc:.4f} (processed {processed_count} images)')
    
    return pixel_auc

def get_snn_threshold(cfg, model_snn, device, timesteps=50):
    """Estimate threshold for SNN model"""
    print('Estimating SNN threshold...')
    valid_good_list = glob(cfg.train_data_dir + '/*png')
    num_valid_data = int(np.ceil(len(valid_good_list) * 0.2))
    total_rec_ssim, total_rec_l1 = [], []
    
    for img_path in valid_good_list[-num_valid_data:]:
        _, _, ssim_residual_map, l1_residual_map = get_snn_residual_map(
            img_path, cfg, model_snn, device, timesteps)
        total_rec_ssim.append(ssim_residual_map)
        total_rec_l1.append(l1_residual_map)
    
    total_rec_ssim = np.array(total_rec_ssim)
    total_rec_l1 = np.array(total_rec_l1)
    ssim_threshold = float(np.percentile(total_rec_ssim, [cfg.percent]))
    l1_threshold = float(np.percentile(total_rec_l1, [cfg.percent]))
    print('SNN ssim_threshold: %f, l1_threshold: %f' %(ssim_threshold, l1_threshold))
    
    return ssim_threshold, l1_threshold


def get_depressing_mask(cfg):
    """Create depressing mask"""
    depr_mask = np.ones((cfg.mask_size, cfg.mask_size)) * 0.2
    depr_mask[5:cfg.mask_size-5, 5:cfg.mask_size-5] = 1
    cfg.depr_mask = depr_mask


def save_snn_results(cfg, image_auc_data, pixel_auc, save_dir, timesteps):
    """Save SNN evaluation results"""
    results_path = os.path.join(save_dir, 'snn_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"SNN Model Results\n")
        f.write(f"Dataset: {cfg.name}\n")
        f.write(f"Original Loss: {cfg.loss}\n")
        f.write(f"Patch size: {cfg.patch_size}\n")
        f.write(f"Z dimension: {cfg.z_dim}\n")
        f.write(f"Grayscale: {cfg.grayscale}\n")
        f.write(f"SNN Timesteps: {timesteps}\n")
        f.write("-" * 30 + "\n")
        if image_auc_data[0] is not None:
            f.write(f"SNN Image-level AUC: {image_auc_data[0]:.4f}\n")
        else:
            f.write("SNN Image-level AUC: N/A\n")
        if pixel_auc is not None:
            f.write(f"SNN Pixel-level AUC: {pixel_auc:.4f}\n")
        else:
            f.write("SNN Pixel-level AUC: N/A (requires ground truth masks)\n")
    
    print(f"SNN results saved to: {results_path}")

def verify_conversion(model_ann, model_snn):
    print("=== CONVERSION VERIFICATION ===")
    
    # Check if models are different objects
    print(f"Same object? {model_ann is model_snn}")
    
    # Check model types
    print(f"ANN type: {type(model_ann)}")
    print(f"SNN type: {type(model_snn)}")
    
    # Check if SNN has spiking neurons
    snn_modules = list(model_snn.named_modules())
    spiking_found = False
    for name, module in snn_modules:
        if 'neuron' in str(type(module)).lower() or 'if' in str(type(module)).lower():
            print(f"Found spiking module: {name} -> {type(module)}")
            spiking_found = True
    
    if not spiking_found:
        print("WARNING: No spiking neurons found in converted model!")
    
    return spiking_found

if __name__ == '__main__':
    cfg = Options().parse()
    timesteps = 10  # Number of timesteps for SNN inference
    
    print("=" * 60)
    print("ANN TO SNN CONVERSION AND TESTING")
    print("=" * 60)
    
    # Step 1: Load ANN model
    model_ann, device = load_model_for_conversion(cfg)
    print(f"Model loaded on device: {device}")
    
    # Step 2: Convert ANN to SNN
    model_snn = convert_ann_to_snn(cfg, model_ann, device)
    print("✓ SNN conversion completed!")

    verify_conversion(model_ann, model_snn)
    
    # Step 3: Create save directory for SNN results
    snn_save_dir = cfg.save_dir.replace('reconst', 'snn_reconst')
    if not os.path.exists(snn_save_dir):
        os.makedirs(snn_save_dir)
    
    # Step 4: Get depressing mask
    get_depressing_mask(cfg)
    
    # Step 5: Get SNN thresholds
    snn_ssim_threshold, snn_l1_threshold = get_snn_threshold(cfg, model_snn, device, timesteps)
    cfg.snn_ssim_threshold = snn_ssim_threshold
    cfg.snn_l1_threshold = snn_l1_threshold
    
    # Step 6: Evaluate SNN performance
    print("\n" + "="*50)
    print("SNN MODEL EVALUATION")
    print("="*50)
    
    # Calculate SNN AUC
    snn_image_auc_data = calculate_snn_image_auc(cfg, model_snn, device, timesteps)
    snn_pixel_auc = calculate_snn_pixel_auc(cfg, model_snn, device, timesteps)

    # Step 7: Save results
    save_snn_results(cfg, snn_image_auc_data, snn_pixel_auc, snn_save_dir, timesteps)
    
    print("\n" + "="*60)
    print("CONVERSION AND TESTING COMPLETED!")
    print("="*60)
    print(f"Original model: ANN AutoEncoder")
    print(f"Converted model: SNN AutoEncoder (T={timesteps})")
    print(f"Results saved to: {snn_save_dir}")
    
    if snn_image_auc_data[0] is not None:
        print(f"SNN Image-level AUC: {snn_image_auc_data[0]:.4f}")
    if snn_pixel_auc is not None:
        print(f"SNN Pixel-level AUC: {snn_pixel_auc:.4f}")
    
    print("Conversion and testing completed!")