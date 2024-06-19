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
import random

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def print_memory_stats(prefix=""):
    if torch.cuda.is_available():
        print(f"{prefix} CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"{prefix} CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
    elif torch.backends.mps.is_available():
        memory_info = psutil.virtual_memory()
        print(f"{prefix} System Memory Used: {memory_info.used / 1024 ** 3:.2f} GB")
        print(f"{prefix} System Memory Available: {memory_info.available / 1024 ** 3:.2f} GB")

class UNet_S(nn.Module):
    def __init__(self):
        super(UNet_S, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = self.upconv(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = self.conv_block(64, 1, final_layer=True)

    def conv_block(self, in_channels, out_channels, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        dec4 = self.dec4(torch.cat((self.upconv4(enc4), enc3), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc2), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc1), dim=1))
        dec1 = self.dec1(dec2)
        return dec1

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=50):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps

    def forward_diffusion(self, clean_image, noisy_image, t):
        alpha = t / self.timesteps
        interpolated_image = alpha * noisy_image + (1 - alpha) * clean_image
        return interpolated_image

    def improved_sampling(self, noisy_image):
        x_t = noisy_image
        for t in reversed(range(1, self.timesteps + 1)):
            alpha_t = t / self.timesteps
            alpha_t_prev = (t - 1) / self.timesteps
            #print_memory_stats(f"Before UNet at timestep {t}")
            x_t_unet = self.unet(x_t)
            #print_memory_stats(f"After UNet at timestep {t}")
            #print(f"Size of x_t_unet at timestep {t}: {x_t_unet.size()}")
            x_tilde = (1 - alpha_t) * x_t_unet + alpha_t * noisy_image
            x_tilde_prev_unet = self.unet(x_t)
            x_tilde_prev = (1 - alpha_t_prev) * x_tilde_prev_unet + alpha_t_prev * noisy_image
            x_t = x_t - x_tilde + x_tilde_prev
            #print_memory_stats(f"After update x_t at timestep {t}")
        return x_t

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.improved_sampling(noisy_step_image)
        return denoised_image

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

# Define the model and optimizer
unet = UNet_S().to(device)
model = DiffusionModel(unet).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Sample training loop
def train_step(model, clean_images, noisy_images, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    timesteps = model.timesteps
    clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
    
    with autocast():
        denoised_images = model(clean_images, noisy_images, timesteps)
        loss = charbonnier_loss(denoised_images, clean_images)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

def train_model(model, train_loader, optimizer, writer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            loss = train_step(model, clean_images, noisy_images, optimizer)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss:.4f}")
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)

        model.eval()
        with torch.no_grad():
            # Log a random subset of validation images
            val_batches = random.sample(list(train_loader), min(5, len(train_loader)))
            for batch_idx, (noisy_images, clean_images) in enumerate(val_batches):
                clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
                denoised_images = model(clean_images, noisy_images, model.timesteps)

                clean_images = denormalize(clean_images.cpu())
                noisy_images = denormalize(noisy_images.cpu())
                denoised_images = denormalize(denoised_images.cpu())

                grid_clean = make_grid(clean_images, nrow=4, normalize=True)
                grid_noisy = make_grid(noisy_images, nrow=4, normalize=True)
                grid_denoised = make_grid(denoised_images, nrow=4, normalize=True)

                writer.add_image(f'Epoch_{epoch + 1}/Clean Images', grid_clean, epoch + 1)
                writer.add_image(f'Epoch_{epoch + 1}/Noisy Images', grid_noisy, epoch + 1)
                writer.add_image(f'Epoch_{epoch + 1}/Denoised Images', grid_denoised, epoch + 1)
                
        writer.flush()


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

    log_dir = os.path.join("runs", "diffusion")
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)
    
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=1, augment=False, dataset_percentage=0.001)
    train_model(model, train_loader, optimizer, writer, num_epochs=10)
    writer.close()