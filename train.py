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

def calculate_val_loss(cfg, model, device, valid_loader, criterion, epoch):
    """Only calculate validation loss"""
    model.eval()
    val_loss = 0.0
    val_pbar = tqdm(valid_loader, desc=f'Epoch {epoch}/{cfg.epochs} [Val]', disable=DISABLE_PROGRESS)
    with torch.no_grad():
        for data, target in val_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_val_loss = val_loss / len(valid_loader)
    return avg_val_loss
    
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
        '--z_dim', str(cfg.z_dim),
        '--weight_file', f'current_epoch_{epoch}.pth'
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
                    pixel_auc = float(line.split(':')[1].strip())
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
            name=f"{cfg.name}_{cfg.loss}_{cfg.patch_size}px",
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
num_valid_data = int(np.ceil(len(file_list) * 0.2))

train_dataset = ImageDataset(file_list[:-num_valid_data], cfg.grayscale)
valid_dataset = ImageDataset(file_list[-num_valid_data:], cfg.grayscale)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

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
best_val_loss = float('inf')
best_image_auc = 0.0
best_pixel_auc = 0.0
patience_counter = 0
PATIENCE_LIMIT = 5  # Changed from 20 to 5

print(f"Training on device: {device}")
print(f"Dataset: {cfg.name}, Loss: {cfg.loss}")
print(f"Train samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}")
print(f"Testing every 10 epochs. Early stopping patience: {PATIENCE_LIMIT} tests")

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
    
    # Test every 10 epochs (validation + AUCs)
    avg_val_loss = None
    test_image_auc = None
    test_pixel_auc = None
    
    if (epoch + 1) % 10 == 1:
        print(f'\nTesting at epoch {epoch + 1}...')
        avg_val_loss = calculate_val_loss(cfg, model, device, valid_loader, criterion, epoch + 1)
        
        # Call test.py subprocess
        test_image_auc, test_pixel_auc, test_success = call_test_script(cfg, epoch + 1)
        
        # Add validation and test metrics
        metrics["val/loss"] = avg_val_loss
        print('test_success:', test_success)
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
        
        # Early stopping and model saving based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'test_image_auc': test_image_auc if test_image_auc is not None else 0.0,
                'test_pixel_auc': test_pixel_auc if test_pixel_auc is not None else 0.0,
                'best_val_loss': best_val_loss,
                'best_image_auc': best_image_auc,
                'best_pixel_auc': best_pixel_auc,
                'config': cfg.__dict__
            }
            
            # Save best model
            best_model_path = os.path.join(cfg.chechpoint_dir, f'{epoch+1:02d}-{avg_val_loss:.5f}.pth')
            torch.save(checkpoint, best_model_path)
            
            best_generic_path = os.path.join(cfg.chechpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_generic_path)
            
            print(f'New best model saved: {best_model_path}')
            
            # Log model artifact to WandB
            if wandb_run:
                artifact = wandb.Artifact(
                    name=f"model-epoch-{epoch+1}",
                    type="model",
                    description=f"Best model at epoch {epoch+1} with val_loss {avg_val_loss:.5f}"
                )
                artifact.add_file(best_model_path)
                wandb_run.log_artifact(artifact)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_LIMIT:
                print(f'Early stopping at epoch {epoch+1} (after {patience_counter} tests without improvement)')
                break
    
    # Log to WandB
    if wandb_run:
        wandb_run.log(metrics)
    
    # Enhanced printing
    print(f'Epoch {epoch+1}/{cfg.epochs}: Train Loss: {avg_train_loss:.6f}', end='')
    if avg_val_loss is not None:
        print(f', Val Loss: {avg_val_loss:.6f}', end='')
    if test_image_auc is not None:
        print(f', Image AUC: {test_image_auc:.4f}', end='')
    if test_pixel_auc is not None:
        print(f', Pixel AUC: {test_pixel_auc:.4f}', end='')
    print()

# # Final test
# print("\nPerforming final test...")
# final_image_auc, final_pixel_auc, test_success = call_test_script(cfg, epoch + 1)
# final_val_loss = calculate_val_loss(cfg, model, device, valid_loader, criterion, epoch + 1)

# if wandb_run:
#     if final_val_loss is not None:
#         wandb_run.log({"final_test/val_loss": final_val_loss})
#     if final_image_auc is not None:
#         wandb_run.log({"final_test/image_auc": final_image_auc})
#     if final_pixel_auc is not None:
#         wandb_run.log({"final_test/pixel_auc": final_pixel_auc})

# Enhanced training summary
print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Dataset: {cfg.name}")
print(f"Loss function: {cfg.loss}")
print(f"Patch size: {cfg.patch_size}")
print(f"Z dimension: {cfg.z_dim}")
print(f"Total epochs: {epoch + 1}")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Best image AUC: {best_image_auc:.4f}")
print(f"Best pixel AUC: {best_pixel_auc:.4f}")
print(f"Early stopped: {'Yes' if patience_counter >= PATIENCE_LIMIT else 'No'}")
print(f"Models saved:")
print(f"  - Best model: {os.path.join(cfg.chechpoint_dir, 'best_model.pth')}")
print(f"Results saved to: {cfg.chechpoint_dir}")

# Enhanced WandB summary
if wandb_run:
    wandb_run.summary["best_val_loss"] = best_val_loss
    wandb_run.summary["best_image_auc"] = best_image_auc
    wandb_run.summary["best_pixel_auc"] = best_pixel_auc
    wandb_run.summary["final_epoch"] = epoch + 1
    wandb_run.summary["early_stopped"] = patience_counter >= PATIENCE_LIMIT
    
    # if final_image_auc is not None:
    #     wandb_run.summary["final_image_auc"] = final_image_auc
    # if final_pixel_auc is not None:
    #     wandb_run.summary["final_pixel_auc"] = final_pixel_auc
    
    print(f"WandB run: {wandb_run.url}")
    wandb_run.finish()

print("Training completed!")