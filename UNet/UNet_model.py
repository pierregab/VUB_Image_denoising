import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import psutil
import subprocess
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def print_memory_stats(prefix=""):
    if torch.cuda.is_available():
        print(f"{prefix} CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"{prefix} CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
    elif torch.backends.mps.is_available():
        memory_info = psutil.virtual_memory()
        print(f"{prefix} System Memory Used: {memory_info.used / 1024 ** 3:.2f} GB")
        print(f"{prefix} System Memory Available: {memory_info.available / 1024 ** 3:.2f} GB")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.upconv2 = self.upconv(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        dec2 = self.dec2(torch.cat((self.upconv2(enc3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))
        return self.final(dec1)

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

# Define the model and optimizer
unet = UNet().to(device)
optimizer = optim.Adam(unet.parameters(), lr=2e-4, betas=(0.9, 0.999))

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Sample training loop
def train_step(model, clean_images, noisy_images, optimizer):
    model.train()
    optimizer.zero_grad()
    
    clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
    
    denoised_images = model(noisy_images)
    loss = charbonnier_loss(denoised_images, clean_images)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_model(model, train_loader, optimizer, writer, num_epochs=10):
    for epoch in range(num_epochs):
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            loss = train_step(model, clean_images, noisy_images, optimizer)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss:.4f}")
            
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
                clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
                denoised_images = model(noisy_images)
                
                clean_images = denormalize(clean_images.cpu())
                noisy_images = denormalize(noisy_images.cpu())
                denoised_images = denormalize(denoised_images.cpu())
                
                grid_clean = make_grid(clean_images, nrow=4, normalize=True)
                grid_noisy = make_grid(noisy_images, nrow=4, normalize=True)
                grid_denoised = make_grid(denoised_images, nrow=4, normalize=True)
                
                writer.add_image(f'Epoch_{epoch + 1}/Clean Images', grid_clean, epoch + 1)
                writer.add_image(f'Epoch_{epoch + 1}/Noisy Images', grid_noisy, epoch + 1)
                writer.add_image(f'Epoch_{epoch + 1}/Denoised Images', grid_denoised, epoch + 1)
             
                if batch_idx >= 0:  # Change this if you want more batches
                    break
        
        writer.flush()

    # Save the model checkpoint
    checkpoint_path = os.path.join("checkpoints", "unet_denoising.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

def start_tensorboard(log_dir):
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    log_dir = os.path.join("runs", "unet_denoising")
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)
    
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=64, augment=False, dataset_percentage=0.1)
    train_model(unet, train_loader, optimizer, writer, num_epochs=20)
    writer.close()
