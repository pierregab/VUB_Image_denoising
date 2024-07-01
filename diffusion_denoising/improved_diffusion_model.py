import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import psutil
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time
import torch.utils.checkpoint as cp

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def print_memory_stats(prefix=""):
    memory_info = psutil.virtual_memory()
    print(f"{prefix} System Memory Used: {memory_info.used / 1024 ** 3:.2f} GB")
    print(f"{prefix} System Memory Available: {memory_info.available / 1024 ** 3:.2f} GB")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = x
        x = nn.functional.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return nn.functional.silu(x + self.shortcut(residual))

class UNet_S_Checkpointed(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()
        self.inc = ResidualBlock(in_channels, base_channels)
        self.down1 = ResidualBlock(base_channels, base_channels * 2)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.down3 = ResidualBlock(base_channels * 4, base_channels * 8)
        
        self.up3 = ResidualBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up2 = ResidualBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = ResidualBlock(base_channels * 2 + base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        # Reshape t to ensure it has the correct dimensions
        if t.dim() == 0:
            t = t.view(1, 1, 1, 1)
        else:
            t = t.view(-1, 1, 1, 1)
        t = t.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        
        x = torch.cat([x, t], dim=1)
        
        x1 = cp.checkpoint(self.inc, x, use_reentrant=False)
        x2 = cp.checkpoint(self.down1, nn.functional.avg_pool2d(x1, 2), use_reentrant=False)
        x3 = cp.checkpoint(self.down2, nn.functional.avg_pool2d(x2, 2), use_reentrant=False)
        x4 = cp.checkpoint(self.down3, nn.functional.avg_pool2d(x3, 2), use_reentrant=False)
        
        x = cp.checkpoint(self.up3, torch.cat([nn.functional.interpolate(x4, scale_factor=2, mode='nearest'), x3], dim=1), use_reentrant=False)
        x = cp.checkpoint(self.up2, torch.cat([nn.functional.interpolate(x, scale_factor=2, mode='nearest'), x2], dim=1), use_reentrant=False)
        x = cp.checkpoint(self.up1, torch.cat([nn.functional.interpolate(x, scale_factor=2, mode='nearest'), x1], dim=1), use_reentrant=False)
        return self.outc(x)

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=20):
        super().__init__()
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
            t_tensor = torch.tensor([t / self.timesteps], device=noisy_image.device)
            x_t_unet = self.unet(x_t, t_tensor)
            x_tilde = (1 - alpha_t) * x_t_unet + alpha_t * noisy_image
            t_tensor_prev = torch.tensor([(t - 1) / self.timesteps], device=noisy_image.device)
            x_tilde_prev_unet = self.unet(x_t, t_tensor_prev)
            x_tilde_prev = (1 - alpha_t_prev) * x_tilde_prev_unet + alpha_t_prev * noisy_image
            x_t = x_t - x_tilde + x_tilde_prev
        return x_t

    def forward(self, clean_image, noisy_image, t):
        if isinstance(t, int):
            t = torch.tensor([t], device=clean_image.device)
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.improved_sampling(noisy_step_image)
        return denoised_image

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

# Define the checkpointed model and optimizer
unet_checkpointed = UNet_S_Checkpointed().to(device)
model_checkpointed = DiffusionModel(unet_checkpointed).to(device)
optimizer = optim.Adam(model_checkpointed.parameters(), lr=2e-4, betas=(0.9, 0.999))

def denormalize(tensor):
    return tensor * 0.5 + 0.5

def train_step_checkpointed(model, clean_images, noisy_images, optimizer):
    model.train()
    optimizer.zero_grad()
    
    timesteps = model.timesteps
    clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
    
    timesteps_tensor = torch.tensor([timesteps], device=device)
    
    denoised_images = model(clean_images, noisy_images, timesteps_tensor)
    loss = charbonnier_loss(denoised_images, clean_images)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_model_checkpointed(model, train_loader, optimizer, writer, num_epochs=10):
    for epoch in range(num_epochs):
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            loss = train_step_checkpointed(model, clean_images, noisy_images, optimizer)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss:.4f}")
            
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
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
             
                if batch_idx >= 0:  # Change this if you want more batches
                    break
        
        writer.flush()

    # Save the model checkpoint
    checkpoint_path = os.path.join("checkpoints", "diffusion_model_checkpointed.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    log_dir = os.path.join("runs", "diffusion_checkpointed")
    writer = SummaryWriter(log_dir=log_dir)
    
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=2, augment=False, dataset_percentage=0.01)
    train_model_checkpointed(model_checkpointed, train_loader, optimizer, writer, num_epochs=10)
    writer.close()