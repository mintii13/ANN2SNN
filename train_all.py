import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm
import subprocess
import sys
import time

from network import AutoEncoder
from utils import generate_image_list, augment_images, read_img
from options import Options
from ssim import SSIM

# Define all MVTec AD classes
MVTEC_CLASSES = [
    # Objects
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
    # Textures  
    'carpet', 'grid', 'leather', 'tile', 'wood'
]

# Class-specific configurations (from README)
CLASS_CONFIGS = {
    'bottle': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': 'W', 'p_rotate': 0.0},
    'cable': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': None, 'p_rotate': 0.0, 'p_horizonal_flip': 0.0, 'p_vertical_flip': 0.0},
    'capsule': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': 'W', 'p_rotate': 0.0, 'p_horizonal_flip': 0.0, 'p_vertical_flip': 0.0},
    'carpet': {'im_resize': 512, 'patch_size': 128, 'z_dim': 100, 'bg_mask': None, 'rotate_angle_vari': 10},
    'grid': {'im_resize': 256, 'patch_size': 128, 'z_dim': 100, 'bg_mask': None, 'grayscale': True},
    'hazelnut': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': 'B', 'p_rotate_crop': 0.0},
    'leather': {'im_resize': 256, 'patch_size': 128, 'z_dim': 100, 'bg_mask': None},
    'metal_nut': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': 'B', 'p_rotate_crop': 0.0, 'p_horizonal_flip': 0.0, 'p_vertical_flip': 0.0},
    'pill': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': 'B', 'p_rotate': 0.0, 'p_horizonal_flip': 0.0, 'p_vertical_flip': 0.0},
    'screw': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': 'W', 'grayscale': True, 'p_rotate': 0.0},
    'tile': {'im_resize': 256, 'patch_size': 128, 'z_dim': 100, 'bg_mask': None},
    'toothbrush': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': None, 'p_rotate': 0.0, 'p_vertical_flip': 0.0},
    'transistor': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': None, 'p_rotate': 0.0, 'p_vertical_flip': 0.0},
    'wood': {'im_resize': 256, 'patch_size': 128, 'z_dim': 100, 'bg_mask': None, 'rotate_angle_vari': 15},
    'zipper': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'bg_mask': None, 'grayscale': True, 'p_rotate': 0.0}
}

class ImageDataset(Dataset):
    def __init__(self, filenames, grayscale):
        self.filenames = filenames
        self.grayscale = grayscale
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img = read_img(self.filenames[idx], self.grayscale)
        img = img.astype(np.float32) / 255.0
        
        if self.grayscale:
            img = torch.FloatTensor(img).unsqueeze(0)
        else:
            img = torch.FloatTensor(img).permute(2, 0, 1)
        
        return img, img

# Custom Loss Functions
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(window_size=window_size)
        
    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)

class SSIMPlusL1Loss(nn.Module):
    def __init__(self, window_size=11, alpha=1.0):
        super(SSIMPlusL1Loss, self).__init__()
        self.ssim = SSIM(window_size=window_size)
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        
    def forward(self, img1, img2):
        ssim_loss = 1 - self.ssim(img1, img2)
        l1_loss = self.l1(img1, img2)
        return ssim_loss + self.alpha * l1_loss


def train_single_class(class_name, base_cfg):
    """Train a single class with its specific configuration"""
    print(f"\n{'='*60}")
    print(f"TRAINING CLASS: {class_name.upper()}")
    print(f"{'='*60}")
    
    # Create class-specific config
    cfg = base_cfg
    cfg.name = class_name
    
    # Apply class-specific parameters
    class_config = CLASS_CONFIGS.get(class_name, {})
    for key, value in class_config.items():
        setattr(cfg, key, value)
    
    # Update paths
    cfg.train_data_dir = os.path.join(base_cfg.train_data_dir.replace('/leather/', f'/{class_name}/'))
    cfg.test_dir = os.path.join(base_cfg.test_dir.replace('/leather/', f'/{class_name}/'))
    cfg.aug_dir = f'./train_patches/{class_name}'
    cfg.chechpoint_dir = f'./results/{class_name}/chechpoints/{cfg.loss}'
    cfg.save_dir = f'./results/{class_name}/reconst/ssim_l1_metric_{cfg.loss}'
    
    # Create directories
    for dir_path in [cfg.chechpoint_dir, cfg.aug_dir, cfg.save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Update derived parameters
    cfg.input_channel = 1 if getattr(cfg, 'grayscale', False) else 3
    cfg.p_crop = 1 if cfg.patch_size != cfg.im_resize else 0
    cfg.mask_size = cfg.patch_size if cfg.im_resize - cfg.patch_size < cfg.stride else cfg.im_resize
    
    print(f"Config: patch_size={cfg.patch_size}, z_dim={cfg.z_dim}, grayscale={getattr(cfg, 'grayscale', False)}")
    
    # Check if dataset exists
    if not os.path.exists(cfg.train_data_dir):
        print(f"ERROR: Training data not found: {cfg.train_data_dir}")
        return False
    
    try:
        # Data preparation
        if cfg.aug_dir and cfg.do_aug:
            img_list = generate_image_list(cfg)
            augment_images(img_list, cfg)

        dataset_dir = cfg.aug_dir if cfg.aug_dir else cfg.train_data_dir
        file_list = glob(dataset_dir + '/*')
        
        if len(file_list) == 0:
            print(f"ERROR: No training images found in {dataset_dir}")
            return False
            
        print(f"Found {len(file_list)} training images")
        
        num_valid_data = int(np.ceil(len(file_list) * 0.2))
        train_dataset = ImageDataset(file_list[:-num_valid_data], getattr(cfg, 'grayscale', False))
        valid_dataset = ImageDataset(file_list[-num_valid_data:], getattr(cfg, 'grayscale', False))

        # Adjust batch size for memory constraints
        effective_batch_size = min(cfg.batch_size, 64 if cfg.patch_size == 256 else 128)
        
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=effective_batch_size, shuffle=False)

        # Model setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoEncoder(cfg).to(device)

        # Loss function
        if cfg.loss == 'ssim_loss':
            criterion = SSIMLoss()
        elif cfg.loss == 'ssim_l1_loss':
            criterion = SSIMPlusL1Loss(alpha=cfg.weight)
        else:
            criterion = nn.MSELoss()

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training on {device} with batch_size={effective_batch_size}")

        for epoch in range(cfg.epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Clear cache periodically
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(valid_loader)
            
            print(f'Epoch {epoch+1}/{cfg.epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, os.path.join(cfg.chechpoint_dir, f'{epoch:02d}-{avg_val_loss:.5f}.pth'))
                print(f'New best model saved')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        # Generate sample reconstructions
        model.eval()
        with torch.no_grad():
            sample_data, _ = next(iter(valid_loader))
            sample_data = sample_data.to(device)
            reconstructed = model(sample_data)
            
            save_snapshot_dir = cfg.chechpoint_dir + '/snapshot/'
            if not os.path.exists(save_snapshot_dir):
                os.makedirs(save_snapshot_dir)
            
            for i in range(min(len(reconstructed), 5)):
                if getattr(cfg, 'grayscale', False):
                    recon_img = (reconstructed[i].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                else:
                    recon_img = (reconstructed[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f'{save_snapshot_dir}{i}_rec_valid.png', recon_img)

        print(f'Training completed for {class_name}!')
        return True
        
    except Exception as e:
        print(f"ERROR during training {class_name}: {e}")
        return False


def run_test_for_class(class_name, base_cfg):
    """Run test.py for a specific class"""
    print(f"\n{'='*60}")
    print(f"TESTING CLASS: {class_name.upper()}")
    print(f"{'='*60}")
    
    # Build test command
    class_config = CLASS_CONFIGS.get(class_name, {})
    
    cmd = [
        'python', 'test.py',
        '--name', class_name,
        '--loss', base_cfg.loss,
        '--im_resize', str(class_config.get('im_resize', 256)),
        '--patch_size', str(class_config.get('patch_size', 128)),
        '--z_dim', str(class_config.get('z_dim', 100))
    ]
    
    # Add optional parameters
    if class_config.get('grayscale', False):
        cmd.append('--grayscale')
    
    if class_config.get('bg_mask'):
        cmd.extend(['--bg_mask', class_config['bg_mask']])
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ Test completed successfully for {class_name}")
            # Print key results
            lines = result.stdout.split('\n')
            for line in lines:
                if 'AUC:' in line or 'Accuracy:' in line:
                    print(f"   {line.strip()}")
        else:
            print(f"❌ Test failed for {class_name}")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Test timeout for {class_name}")
    except Exception as e:
        print(f"ERROR during testing {class_name}: {e}")


def main():
    # Parse base configuration
    base_cfg = Options().parse()
    
    # Training summary
    results_summary = {}
    start_time = time.time()
    
    print(f"{'='*80}")
    print(f"SEQUENTIAL TRAINING OF ALL MVTEC CLASSES")
    print(f"Total classes: {len(MVTEC_CLASSES)}")
    print(f"Epochs per class: {base_cfg.epochs}")
    print(f"{'='*80}")
    
    for i, class_name in enumerate(MVTEC_CLASSES):
        class_start_time = time.time()
        
        print(f"\n[{i+1}/{len(MVTEC_CLASSES)}] Processing: {class_name}")
        
        # Train the class
        success = train_single_class(class_name, base_cfg)
        
        if success:
            # Run test after training
            run_test_for_class(class_name, base_cfg)
            
            class_time = time.time() - class_start_time
            results_summary[class_name] = {
                'status': 'completed',
                'time': class_time
            }
            print(f"✅ {class_name} completed in {class_time/60:.1f} minutes")
        else:
            results_summary[class_name] = {
                'status': 'failed',
                'time': 0
            }
            print(f"❌ {class_name} failed")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final summary
    total_time = time.time() - start_time
    completed = sum(1 for r in results_summary.values() if r['status'] == 'completed')
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Completed: {completed}/{len(MVTEC_CLASSES)} classes")
    print(f"Success rate: {completed/len(MVTEC_CLASSES)*100:.1f}%")
    
    print(f"\nPer-class results:")
    for class_name, result in results_summary.items():
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        time_str = f"{result['time']/60:.1f}min" if result['time'] > 0 else "N/A"
        print(f"  {status_icon} {class_name:12} - {result['status']:9} - {time_str}")


if __name__ == '__main__':
    main()