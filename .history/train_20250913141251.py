import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm

from network import AutoEncoder
from utils import generate_image_list, augment_images, read_img
from options import Options
from pytorch_ssim import SSIM

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

# Data preparation (keep same logic as original)
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

# Training loop
best_val_loss = float('inf')
patience_counter = 0

print(f"Training on device: {device}")

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
    
    # Save best model (similar to ModelCheckpoint)
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
        if patience_counter >= 20:  # Early stopping patience
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
    
    for i in range(min(len(reconstructed), 10)):
        if cfg.grayscale:
            recon_img = (reconstructed[i].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        else:
            recon_img = (reconstructed[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f'{save_snapshot_dir}{i}_rec_valid.png', recon_img)

print('Training completed!')