import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from network import AutoEncoder
from utils import generate_image_list, augment_images, read_img, get_patch, patch2img, bg_mask, set_img_color
from options import Options
from ssim import SSIM
import subprocess, sys
import setproctitle
setproctitle.setproctitle("Minh Tri Nguyen is training...") 
# WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
    # Set API key
    os.environ['WANDB_API_KEY'] = '997d5b0dec14260a6aa6e91d178a836d82a483d9'
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb'")

cfg = Options().parse()

class ImageDataset(Dataset):
    def __init__(self, filenames, grayscale):
        self.filenames = filenames
        self.grayscale = grayscale
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img = read_img(self.filenames[idx], self.grayscale)
        img = img / 255.0
        
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
  
def call_test_script(cfg, epoch):
    """Call test.py as subprocess and parse results"""
    # Import CLASS_CONFIGS from train_all.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_all", "train_all.py")
    train_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_all)

    # Build test command với đúng config
    test_cmd = [
        sys.executable, 'test.py',
        '--name', cfg.name,
        '--loss', cfg.loss,
        '--im_resize', str(cfg.im_resize),
        '--patch_size', str(cfg.patch_size),
        '--z_dim', str(cfg.z_dim)
    ]

    # Add grayscale nếu có trong cfg hiện tại
    if cfg.grayscale:
        test_cmd.append('--grayscale')

    # Import và add class-specific test configs (chỉ các tham số đặc biệt)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_all", "train_all.py")
        train_all = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_all)
        
        if cfg.name in train_all.CLASS_CONFIGS:
            test_config = train_all.CLASS_CONFIGS[cfg.name].get('test', {})
            for key, value in test_config.items():
                if key == 'bg_mask' and value:
                    test_cmd.extend(['--bg_mask', value])
                # Không cần check grayscale ở đây vì đã add ở trên
    except:
        # Fallback nếu không import được train_all.py
        pass
    
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=600)
        
        # Parse output để extract AUC scores
        image_auc = None
        pixel_auc = None
        
        for line in result.stdout.split('\n'):
            if 'Image-level AUC:' in line:
                try:
                    image_auc = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Pixel-level AUC:' in line:
                try:
                    # Extract chỉ số AUC, loại bỏ phần text sau
                    auc_text = line.split(':')[1].strip()
                    pixel_auc = float(auc_text.split()[0])  # Lấy số đầu tiên trước space
                except:
                    pass
        
        return image_auc, pixel_auc, True
        
    except subprocess.TimeoutExpired:
        print(f"Test subprocess timed out at epoch {epoch}")
        return None, None, False
    except Exception as e:
        print(f"Test subprocess failed at epoch {epoch}: {e}")
        return None, None, False

def init_wandb(cfg):
    """Initialize WandB"""
    if not WANDB_AVAILABLE:
        return None
    
    config = {
        "learning_rate": cfg.lr,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "patch_size": cfg.patch_size,
        "z_dim": cfg.z_dim,
        "loss_function": cfg.loss,
        "grayscale": cfg.grayscale,
        "weight_decay": cfg.decay,
        "dataset": cfg.name,
        "im_resize": cfg.im_resize,
        "mask_size": cfg.mask_size,
        "stride": cfg.stride
    }
    
    try:
        run = wandb.init(
            project="ANN2SNN",
            name=f"{cfg.name}_{cfg.loss}_{cfg.patch_size}px_sv_ReLU_nonStop",
            config=config,
            tags=[cfg.name, cfg.loss, "autoencoder"],
            notes=f"Autoencoder training on {cfg.name} dataset with {cfg.loss} loss"
        )
        return run
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None

# Data preparation
if cfg.aug_dir and cfg.do_aug:
    img_list = generate_image_list(cfg)
    augment_images(img_list, cfg)

dataset_dir = cfg.aug_dir if cfg.aug_dir else cfg.train_data_dir
file_list = glob(dataset_dir + '/*')
# Sử dụng toàn bộ train set
train_dataset = ImageDataset(file_list, cfg.grayscale)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

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

# Initialize WandB
wandb_run = init_wandb(cfg)

################### Training loop ###################
best_image_auc = 0.0
best_pixel_auc = 0.0

print(f"Training on device: {device}")
print(f"Dataset: {cfg.name}, Loss: {cfg.loss}")
print(f"Train samples: {len(train_dataset)}")

DISABLE_PROGRESS = os.environ.get('DISABLE_TQDM', '0') == '1'
for epoch in range(cfg.epochs):
    # Training phase only
    model.train()
    train_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.epochs} [Train]', disable=DISABLE_PROGRESS)
    for batch_idx, (data, target) in enumerate(train_pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Base metrics to log
    metrics = {
        "train/loss": avg_train_loss,
        "epoch": (epoch + 1)
    }
    
    test_image_auc = None
    test_pixel_auc = None
    # Save current model mỗi epoch
    current_model_path = os.path.join(cfg.chechpoint_dir, 'model.pth')
    checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_image_auc': test_image_auc if test_image_auc is not None else 0.0,
            'test_pixel_auc': test_pixel_auc if test_pixel_auc is not None else 0.0,
            'config': cfg.__dict__
        }
    torch.save(checkpoint, current_model_path)
    print(f'Current model saved: {current_model_path}')

    # Test every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Call test.py subprocess
        test_image_auc, test_pixel_auc, test_success = call_test_script(cfg, epoch + 1)
        # Add validation and test metrics
        
        print(f'\nTest at epoch {epoch + 1}...')
        print('test_image_auc:', test_image_auc)
        print('test_pixel_auc:', test_pixel_auc)
        if test_success and test_image_auc is not None:
            metrics["test/image_auc"] = test_image_auc
            if test_image_auc > best_image_auc:
                best_image_auc = test_image_auc
                metrics["test/best_image_auc"] = best_image_auc
        
        if test_success and test_pixel_auc is not None:
            metrics["test/pixel_auc"] = test_pixel_auc
            if test_pixel_auc > best_pixel_auc:
                best_pixel_auc = test_pixel_auc
                metrics["test/best_pixel_auc"] = best_pixel_auc
    
    # Log to WandB
    if wandb_run:
        wandb_run.log(metrics)
    
    # Enhanced printing
    print(f'Epoch {epoch+1}/{cfg.epochs}: Train Loss: {avg_train_loss:.6f}', end='')
    if test_image_auc is not None:
        print(f', Image AUC: {test_image_auc:.4f}', end='')
    if test_pixel_auc is not None:
        print(f', Pixel AUC: {test_pixel_auc:.4f}', end='')
    print()

final_model_path = os.path.join(cfg.chechpoint_dir, 'best_model.pth')
final_checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'test_image_auc': best_image_auc,
    'test_pixel_auc': best_pixel_auc,
    'config': cfg.__dict__
}
torch.save(final_checkpoint, final_model_path)
print(f'Final model saved as best_model.pth: {final_model_path}')

# Enhanced training summary
print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Dataset: {cfg.name}")
print(f"Loss function: {cfg.loss}")
print(f"Patch size: {cfg.patch_size}")
print(f"Z dimension: {cfg.z_dim}")
print(f"Total epochs: {epoch + 1}")
print(f"Best image AUC: {best_image_auc:.4f}")
print(f"Best pixel AUC: {best_pixel_auc:.4f}")
print(f"Models saved:")
print(f"  - Final model: {os.path.join(cfg.chechpoint_dir, 'model.pth')}")
print(f"Results saved to: {cfg.chechpoint_dir}")

# Enhanced WandB summary
if wandb_run:
    wandb_run.summary["best_image_auc"] = best_image_auc
    wandb_run.summary["best_pixel_auc"] = best_pixel_auc
    wandb_run.summary["final_epoch"] = epoch + 1
    
    # if final_image_auc is not None:
    #     wandb_run.summary["final_image_auc"] = final_image_auc
    # if final_pixel_auc is not None:
    #     wandb_run.summary["final_pixel_auc"] = final_pixel_auc
    
    print(f"WandB run: {wandb_run.url}")
    wandb_run.finish()

print("Training completed!")