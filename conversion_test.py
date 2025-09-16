import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim
from glob import glob
import cv2
import os
from sklearn.metrics import roc_auc_score, roc_curve

from spikingjelly.activation_based import ann2snn, neuron, functional
from utils import read_img, get_patch, patch2img
from network import AutoEncoder
from options import Options


class MembraneOutputLayer(nn.Module):
    """Custom layer that returns membrane potential instead of spikes for reconstruction"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, tau=2.0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.neuron = neuron.LIFNode(tau=tau, detach_reset=False)
        
    def forward(self, x):
        x = self.conv(x)
        self.neuron(x)  # Update neuron state
        return self.neuron.v  # Return membrane potential for reconstruction


class CalibrationDataset(Dataset):
    """Dataset for SNN conversion calibration"""
    
    def __init__(self, files, grayscale, patch_size, max_samples=1000):
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


class ANNToSNNConverter:
    """Handles ANN to SNN conversion and evaluation"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load trained ANN model"""
        model = AutoEncoder(self.cfg).to(self.device)
        checkpoint_path = os.path.join(self.cfg.chechpoint_dir, 'model.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded ANN checkpoint: {checkpoint_path}")
        return model
    
    def create_calibration_dataloader(self):
        """Create dataloader for SNN conversion calibration"""
        good_files = glob(self.cfg.train_data_dir + '/*png')
        
        if not good_files:
            raise ValueError(f"No training files found in {self.cfg.train_data_dir}")
        
        dataset = CalibrationDataset(good_files, self.cfg.grayscale, self.cfg.patch_size)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        
        print(f"Using {len(dataset)} training samples for calibration")
        return dataloader
    
    def convert_to_snn(self, model_ann):
        """Convert ANN model to SNN using SpikingJelly"""
        print("Converting ANN to SNN...")
        
        # Remove Sigmoid layer before conversion
        decoder_layers = list(model_ann.decoder.children())
        if isinstance(decoder_layers[-1], nn.Sigmoid):
            model_ann.decoder = nn.Sequential(*decoder_layers[:-1])
            print("Removed Sigmoid layer before conversion")
        
        # Create calibration dataloader and converter
        calib_dataloader = self.create_calibration_dataloader()
        converter = ann2snn.Converter(
            dataloader=calib_dataloader, 
            device=self.device, 
            mode='max',
            momentum=0.1
        )
        
        # Convert to SNN
        model_snn = converter(model_ann)
        print("Basic ANN to SNN conversion completed")
        
        # Replace final layer with custom membrane output layer
        self._replace_final_layer(model_snn)
        return model_snn
    
    def _replace_final_layer(self, model_snn):
        """Replace final layer with membrane output layer"""
        decoder_layers = list(model_snn.decoder.children())
        last_layer = decoder_layers[-1]
        
        if hasattr(last_layer, 'conv'):
            orig_conv = last_layer.conv
            membrane_layer = MembraneOutputLayer(
                in_channels=orig_conv.in_channels,
                out_channels=orig_conv.out_channels, 
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                tau=2.0
            )
            
            # Copy weights from original layer
            membrane_layer.conv.load_state_dict(orig_conv.state_dict())
            decoder_layers[-1] = membrane_layer
            model_snn.decoder = nn.Sequential(*decoder_layers)
            
            print("Replaced final layer with membrane output layer")
    
    def get_residual_map(self, img_path, model_snn, timesteps=50):
        """Get residual map using SNN model"""
        test_img = read_img(img_path, self.cfg.grayscale)
        
        # Resize and crop image
        if test_img.shape[:2] != (self.cfg.im_resize, self.cfg.im_resize):
            test_img = cv2.resize(test_img, (self.cfg.im_resize, self.cfg.im_resize))
        if self.cfg.im_resize != self.cfg.mask_size:
            tmp = (self.cfg.im_resize - self.cfg.mask_size) // 2
            test_img = test_img[tmp:tmp+self.cfg.mask_size, tmp:tmp+self.cfg.mask_size]
        
        test_img_norm = test_img / 255.0
        
        if test_img.shape[:2] == (self.cfg.patch_size, self.cfg.patch_size):
            decoded_img = self._process_single_patch(test_img_norm, model_snn, timesteps)
        else:
            decoded_img = self._process_multiple_patches(test_img_norm, model_snn, timesteps)
        
        # Calculate residual maps
        rec_img = np.reshape((decoded_img * 255.).astype('uint8'), test_img.shape)
        
        if self.cfg.grayscale:
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
    
    def _process_single_patch(self, test_img_norm, model_snn, timesteps):
        """Process single patch through SNN"""
        if self.cfg.grayscale:
            test_tensor = torch.FloatTensor(test_img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            test_tensor = torch.FloatTensor(test_img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        functional.reset_net(model_snn)
        
        outputs = []
        with torch.no_grad():
            for _ in range(timesteps):
                output = model_snn(test_tensor)
                outputs.append(output)
        
        decoded_tensor = torch.stack(outputs).mean(dim=0)
        decoded_tensor = torch.sigmoid(decoded_tensor)
        
        if self.cfg.grayscale:
            return decoded_tensor.squeeze().cpu().numpy()
        else:
            return decoded_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    def _process_multiple_patches(self, test_img_norm, model_snn, timesteps):
        """Process multiple patches through SNN"""
        patches = get_patch(test_img_norm, self.cfg.patch_size, self.cfg.stride)
        
        if self.cfg.grayscale:
            patches_tensor = torch.FloatTensor(patches).unsqueeze(1).to(self.device)
        else:
            patches_tensor = torch.FloatTensor(patches).permute(0, 3, 1, 2).to(self.device)
        
        batch_size = 8
        decoded_patches = []
        
        with torch.no_grad():
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i:i+batch_size]
                functional.reset_net(model_snn)
                
                batch_outputs = []
                for _ in range(timesteps):
                    output = model_snn(batch)
                    batch_outputs.append(output)
                
                decoded_batch = torch.stack(batch_outputs).mean(dim=0)
                decoded_batch = torch.sigmoid(decoded_batch)
                
                if self.cfg.grayscale:
                    decoded_batch = decoded_batch.squeeze(1).cpu().numpy()
                else:
                    decoded_batch = decoded_batch.permute(0, 2, 3, 1).cpu().numpy()
                
                decoded_patches.append(decoded_batch)
        
        decoded_patches = np.concatenate(decoded_patches, axis=0)
        return patch2img(decoded_patches, self.cfg.im_resize, self.cfg.patch_size, self.cfg.stride)


class SNNEvaluator:
    """Evaluates SNN model performance"""
    
    def __init__(self, cfg, converter):
        self.cfg = cfg
        self.converter = converter
        
    def calculate_image_auc(self, model_snn, timesteps=50):
        """Calculate Image-level AUC for SNN model"""
        print('Calculating SNN Image-level AUC...')
        
        all_scores = []
        all_labels = []
        
        # Process good samples
        good_files = glob(os.path.join(self.cfg.test_dir, 'good', '*'))
        for img_path in good_files:
            try:
                _, _, ssim_res, l1_res = self.converter.get_residual_map(img_path, model_snn, timesteps)
                score = np.max(ssim_res + l1_res)
                all_scores.append(score)
                all_labels.append(0)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Process defective samples
        defect_folders = [folder for folder in os.listdir(self.cfg.test_dir) 
                         if folder != 'good' and os.path.isdir(os.path.join(self.cfg.test_dir, folder))]
        
        for folder in defect_folders:
            defect_files = glob(os.path.join(self.cfg.test_dir, folder, '*'))
            for img_path in defect_files:
                try:
                    _, _, ssim_res, l1_res = self.converter.get_residual_map(img_path, model_snn, timesteps)
                    score = np.max(ssim_res + l1_res)
                    all_scores.append(score)
                    all_labels.append(1)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if len(set(all_labels)) < 2:
            print("Warning: Need both normal and defective samples for AUC calculation")
            return None, None, None, None
        
        image_auc = roc_auc_score(all_labels, all_scores)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        
        print(f'SNN Image-level AUC: {image_auc:.4f}')
        return image_auc, fpr, tpr, thresholds
    
    def calculate_pixel_auc(self, model_snn, timesteps=80):
        """Calculate Pixel-level AUC using ground truth masks"""
        print('Calculating SNN Pixel-level AUC...')
        
        gt_dir = self.cfg.test_dir.replace('test', 'ground_truth')
        if not os.path.exists(gt_dir):
            print(f"Warning: Ground truth directory not found: {gt_dir}")
            return None
        
        all_pixel_scores = []
        all_pixel_labels = []
        
        defect_folders = [folder for folder in os.listdir(self.cfg.test_dir) 
                         if folder != 'good' and os.path.isdir(os.path.join(self.cfg.test_dir, folder))]
        
        processed_count = 0
        for folder in defect_folders:
            test_files = glob(os.path.join(self.cfg.test_dir, folder, '*'))
            gt_folder_path = os.path.join(gt_dir, folder)
            
            if not os.path.exists(gt_folder_path):
                continue
                
            for test_path in test_files:
                gt_path = self._find_ground_truth_mask(test_path, gt_folder_path)
                if gt_path is None:
                    continue
                    
                try:
                    _, _, ssim_res, l1_res = self.converter.get_residual_map(test_path, model_snn, timesteps)
                    combined_score = ssim_res + l1_res
                    
                    gt_mask = cv2.imread(gt_path, 0)
                    if gt_mask is None:
                        continue
                        
                    if gt_mask.shape != combined_score.shape:
                        gt_mask = cv2.resize(gt_mask, (combined_score.shape[1], combined_score.shape[0]))
                    
                    gt_binary = (gt_mask > 127).astype(int)
                    
                    all_pixel_scores.extend(combined_score.flatten())
                    all_pixel_labels.extend(gt_binary.flatten())
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {test_path}: {e}")
        
        if processed_count == 0 or len(set(all_pixel_labels)) < 2:
            print("Warning: No valid samples found for pixel-level AUC")
            return None
        
        pixel_auc = roc_auc_score(all_pixel_labels, all_pixel_scores)
        print(f'SNN Pixel-level AUC: {pixel_auc:.4f} (processed {processed_count} images)')
        return pixel_auc
    
    def _find_ground_truth_mask(self, test_path, gt_folder_path):
        """Find corresponding ground truth mask"""
        filename = os.path.splitext(os.path.basename(test_path))[0]
        possible_extensions = ['.png', '.bmp', '.jpg', '.jpeg']
        
        for ext in possible_extensions:
            # Try with '_mask' suffix first
            gt_path = os.path.join(gt_folder_path, filename + '_mask' + ext)
            if os.path.exists(gt_path):
                return gt_path
            # Try without '_mask' suffix
            gt_path = os.path.join(gt_folder_path, filename + ext)
            if os.path.exists(gt_path):
                return gt_path
        return None
    
    def estimate_thresholds(self, model_snn, timesteps=50):
        """Estimate thresholds for SNN model"""
        print('Estimating SNN thresholds...')
        
        valid_good_list = glob(self.cfg.train_data_dir + '/*png')
        num_valid_data = int(np.ceil(len(valid_good_list) * 0.2))
        
        total_rec_ssim, total_rec_l1 = [], []
        
        for img_path in valid_good_list[-num_valid_data:]:
            _, _, ssim_residual_map, l1_residual_map = self.converter.get_residual_map(
                img_path, model_snn, timesteps)
            total_rec_ssim.append(ssim_residual_map)
            total_rec_l1.append(l1_residual_map)
        
        total_rec_ssim = np.array(total_rec_ssim)
        total_rec_l1 = np.array(total_rec_l1)
        
        ssim_threshold = float(np.percentile(total_rec_ssim, [self.cfg.percent]))
        l1_threshold = float(np.percentile(total_rec_l1, [self.cfg.percent]))
        
        print(f'SNN ssim_threshold: {ssim_threshold:.6f}, l1_threshold: {l1_threshold:.6f}')
        return ssim_threshold, l1_threshold


def save_results(cfg, image_auc_data, pixel_auc, save_dir, timesteps):
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
    
    print(f"Results saved to: {results_path}")


def create_depressing_mask(cfg):
    """Create depressing mask for evaluation"""
    depr_mask = np.ones((cfg.mask_size, cfg.mask_size)) * 0.2
    depr_mask[5:cfg.mask_size-5, 5:cfg.mask_size-5] = 1
    cfg.depr_mask = depr_mask


def verify_conversion(model_ann, model_snn):
    """Verify if conversion was successful"""
    print("=== CONVERSION VERIFICATION ===")
    
    # Check if models are different objects
    print(f"Same object? {model_ann is model_snn}")
    
    # Check if SNN has spiking neurons
    snn_modules = list(model_snn.named_modules())
    spiking_count = 0
    
    for name, module in snn_modules:
        module_type = str(type(module)).lower()
        if 'neuron' in module_type or 'lif' in module_type:
            print(f"Found spiking module: {name} -> {type(module)}")
            spiking_count += 1
    
    success = spiking_count > 0
    if not success:
        print("WARNING: No spiking neurons found in converted model!")
    else:
        print(f"Conversion successful: {spiking_count} spiking modules found")
    
    return success


def main():
    """Main execution function"""
    cfg = Options().parse()
    timesteps = 10
    
    print("=" * 60)
    print("ANN TO SNN CONVERSION AND TESTING")
    print("=" * 60)
    
    # Initialize converter and evaluator
    converter = ANNToSNNConverter(cfg)
    evaluator = SNNEvaluator(cfg, converter)
    
    # Load and convert model
    model_ann = converter.load_model()
    model_snn = converter.convert_to_snn(model_ann)
    
    # Verify conversion
    conversion_success = verify_conversion(model_ann, model_snn)
    if not conversion_success:
        print("Conversion failed. Exiting.")
        return
    
    # Create save directory
    snn_save_dir = cfg.save_dir.replace('reconst', 'snn_reconst')
    os.makedirs(snn_save_dir, exist_ok=True)
    
    # Setup evaluation
    create_depressing_mask(cfg)
    
    # Estimate thresholds
    ssim_threshold, l1_threshold = evaluator.estimate_thresholds(model_snn, timesteps)
    cfg.snn_ssim_threshold = ssim_threshold
    cfg.snn_l1_threshold = l1_threshold
    
    # Evaluate performance
    print("\n" + "="*50)
    print("SNN MODEL EVALUATION")
    print("="*50)
    
    image_auc_data = evaluator.calculate_image_auc(model_snn, timesteps)
    pixel_auc = evaluator.calculate_pixel_auc(model_snn, timesteps)
    
    # Save results
    save_results(cfg, image_auc_data, pixel_auc, snn_save_dir, timesteps)
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION AND TESTING COMPLETED!")
    print("="*60)
    print(f"Original model: ANN AutoEncoder")
    print(f"Converted model: SNN AutoEncoder (T={timesteps})")
    print(f"Results saved to: {snn_save_dir}")
    
    if image_auc_data[0] is not None:
        print(f"SNN Image-level AUC: {image_auc_data[0]:.4f}")
    if pixel_auc is not None:
        print(f"SNN Pixel-level AUC: {pixel_auc:.4f}")


if __name__ == '__main__':
    main()