import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import psutil
import subprocess
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from datetime import datetime

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

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Reduce channels to 16
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Reduce channels to 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),  # Adjusted for concatenated channels
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Change output channels to 1 for grayscale
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x2 = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        if x2.size() != x1.size():
            x2 = nn.functional.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = self.decoder(torch.cat([x2, x1], dim=1))
        return x3

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=70):
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
            x_tilde = (1 - alpha_t) * self.unet(x_t)
            if x_tilde.size() != noisy_image.size():
                x_tilde = nn.functional.interpolate(x_tilde, size=noisy_image.size()[2:], mode='bilinear', align_corners=True)
            x_tilde += alpha_t * noisy_image
            
            x_tilde_prev = (1 - alpha_t_prev) * self.unet(x_t)
            if x_tilde_prev.size() != noisy_image.size():
                x_tilde_prev = nn.functional.interpolate(x_tilde_prev, size=noisy_image.size()[2:], mode='bilinear', align_corners=True)
            x_tilde_prev += alpha_t_prev * noisy_image
            
            x_t = x_t - x_tilde + x_tilde_prev
        return x_t

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.improved_sampling(noisy_step_image)
        return denoised_image

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

# Define the model and optimizer
unet = UNet().to(device)
model = DiffusionModel(unet).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))

# TensorBoard writer - unique log directory
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("runs", "diffusion", current_time)
writer = SummaryWriter(log_dir=log_dir)

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Sample training loop
def train_step(model, clean_images, noisy_images, optimizer):
    model.train()
    optimizer.zero_grad()
    
    timesteps = model.timesteps
    total_loss = 0
    for t in range(timesteps):
        with torch.no_grad():  # Temporarily use no_grad to avoid storing unnecessary gradients
            clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
        denoised_images = model(clean_images, noisy_images, t)
        loss = charbonnier_loss(denoised_images, clean_images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / timesteps

def train_model(model, train_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for batch_idx, (clean_images, noisy_images) in enumerate(train_loader):
            print_memory_stats("Before train_step")
            loss = train_step(model, clean_images, noisy_images, optimizer)
            print_memory_stats("After train_step")
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss:.4f}")
            
            # Log loss to TensorBoard
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
            
            # Save and visualize images in TensorBoard
            if batch_idx % 1 == 0:  # Adjust the frequency as needed
                model.eval()
                with torch.no_grad():
                    clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
                    denoised_images = model(clean_images, noisy_images, model.timesteps)
                    
                    clean_images = denormalize(clean_images.cpu())
                    noisy_images = denormalize(noisy_images.cpu())
                    denoised_images = denormalize(denoised_images.cpu())
                    
                    grid_clean = make_grid(clean_images, nrow=4, normalize=True)
                    grid_noisy = make_grid(noisy_images, nrow=4, normalize=True)
                    grid_denoised = make_grid(denoised_images, nrow=4, normalize=True)
                    
                    writer.add_image(f'Epoch_{epoch + 1}/Clean Images', grid_clean, epoch * len(train_loader) + batch_idx)
                    writer.add_image(f'Epoch_{epoch + 1}/Noisy Images', grid_noisy, epoch * len(train_loader) + batch_idx)
                    writer.add_image(f'Epoch_{epoch + 1}/Denoised Images', grid_denoised, epoch * len(train_loader) + batch_idx)

def start_tensorboard(log_dir):
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

if __name__ == "__main__":
    # Clear any cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Start TensorBoard
    start_tensorboard(log_dir)
    
    # Load data
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=1, augment=True, dataset_percentage=0.01)
    
    # Train the model
    train_model(model, train_loader, optimizer, num_epochs=10)

    # Close the TensorBoard writer
    writer.close()
