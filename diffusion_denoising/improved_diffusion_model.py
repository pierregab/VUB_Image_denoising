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
import torch.utils.checkpoint as cp

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class UNet_Optimized(nn.Module):
    def __init__(self):
        super(UNet_Optimized, self).__init__()
        self.enc1 = self.conv_block(2, 16)  # Reduced number of channels
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.upconv3 = self.upconv(64, 32)
        self.upconv2 = self.upconv(32, 16)
        self.dec3 = self.conv_block(64, 32)
        self.dec2 = self.conv_block(32, 16)
        self.dec1 = self.conv_block(16, 1, final_layer=True)

    def conv_block(self, in_channels, out_channels, final_layer=False):
        if final_layer:
            return nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, t):
        # Concatenate the time step with the input image
        t = t.expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat((x, t), dim=1)

        enc1 = cp.checkpoint(self.enc1, x, use_reentrant=False)
        enc2 = cp.checkpoint(self.enc2, self.pool(enc1), use_reentrant=False)
        enc3 = cp.checkpoint(self.enc3, self.pool(enc2), use_reentrant=False)
        dec3 = cp.checkpoint(self.dec3, torch.cat((self.upconv3(enc3), enc2), dim=1), use_reentrant=False)
        dec2 = cp.checkpoint(self.dec2, torch.cat((self.upconv2(dec3), enc1), dim=1), use_reentrant=False)
        dec1 = self.dec1(dec2)
        return dec1

class DiffusionModelOptimized(nn.Module):
    def __init__(self, unet, timesteps=20):
        super(DiffusionModelOptimized, self).__init__()
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
            t_tensor = torch.tensor([t / self.timesteps], device=noisy_image.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x_t_unet = self.unet(x_t, t_tensor)
            x_tilde = (1 - alpha_t) * x_t_unet + alpha_t * noisy_image
            t_tensor_prev = torch.tensor([(t - 1) / self.timesteps], device=noisy_image.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x_tilde_prev_unet = self.unet(x_t, t_tensor_prev)
            x_tilde_prev = (1 - alpha_t_prev) * x_tilde_prev_unet + alpha_t_prev * noisy_image
            x_t = x_t - x_tilde + x_tilde_prev
        return x_t

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.improved_sampling(noisy_step_image)
        return denoised_image

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Initialize the optimized model and optimizer
unet_optimized = UNet_Optimized().to(device)
model_optimized = DiffusionModelOptimized(unet_optimized).to(device)
optimizer = optim.Adam(model_optimized.parameters(), lr=2e-4, betas=(0.9, 0.999))

def train_step_checkpointed(model, clean_images, noisy_images, optimizer):
    model.train()
    optimizer.zero_grad()
    
    timesteps = model.timesteps
    clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
    
    denoised_images = model(clean_images, noisy_images, timesteps)
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
    checkpoint_path = os.path.join("checkpoints", "diffusion_model_optimized.pth")
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

    log_dir = os.path.join("runs", "diffusion_optimized")
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)
    
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=8, augment=False, dataset_percentage=0.01)
    train_model_checkpointed(model_optimized, train_loader, optimizer, writer, num_epochs=10)
    writer.close()
