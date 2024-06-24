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
import math

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

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        #print(f"Block input x: {x.shape}")
        #print(f"Block input t: {t.shape}")
        
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input)")

        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        # Reshape time embedding to match batch size and channel dimensions of h
        time_emb = time_emb.view(h.shape[0], -1, 1, 1)
        h = h + time_emb.expand(-1, -1, h.size(2), h.size(3))
        h = self.bnorm2(self.relu(self.conv2(h)))
        #print(f"Block output h: {h.shape}")
        return self.transform(h)

class UNet_S_Improved(nn.Module):
    def __init__(self, time_emb_dim=100):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv2d(1, 64, 3, padding=1)
        
        self.downs = nn.ModuleList([
            Block(64, 128, time_emb_dim),
            Block(128, 256, time_emb_dim),
            Block(256, 512, time_emb_dim),
            Block(512, 1024, time_emb_dim)
        ])
        
        self.ups = nn.ModuleList([
            Block(1024, 512, time_emb_dim, up=True),
            Block(512, 256, time_emb_dim, up=True),
            Block(256, 128, time_emb_dim, up=True),
            Block(128, 64, time_emb_dim, up=True)
        ])
        
        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        #print(f"UNet input x: {x.shape}")
        #print(f"UNet input t: {t.shape}")

        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input)")

        t = self.time_mlp(t)
        x = self.conv0(x)
        residuals = []
        
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
        
        #print(f"UNet output x: {x.shape}")
        return self.output(x)

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=20):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps

    def forward_diffusion(self, clean_image, noisy_image, t):
        alpha = t.view(-1, 1, 1, 1) / self.timesteps  # Reshape alpha to match the image dimensions
        interpolated_image = alpha * noisy_image + (1 - alpha) * clean_image
        return interpolated_image
    
    def improved_sampling(self, noisy_image):
        x_t = noisy_image
        for t in reversed(range(1, self.timesteps + 1)):
            alpha_t = t / self.timesteps
            alpha_t_prev = (t - 1) / self.timesteps
            t_tensor = torch.full((noisy_image.size(0),), t / self.timesteps, device=noisy_image.device)
            x_t_unet = self.unet(x_t, t_tensor)
            x_tilde = (1 - alpha_t) * x_t_unet + alpha_t * noisy_image
            t_tensor_prev = torch.full((noisy_image.size(0),), (t - 1) / self.timesteps, device=noisy_image.device)
            x_tilde_prev_unet = self.unet(x_t, t_tensor_prev)
            x_tilde_prev = (1 - alpha_t_prev) * x_tilde_prev_unet + alpha_t_prev * noisy_image
            x_t = x_t - x_tilde + x_tilde_prev
        return x_t

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        print(f"noisy_step_image: {noisy_step_image.shape}")
        denoised_image = self.improved_sampling(noisy_step_image)
        return denoised_image

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

# Training function
def train_diffusion_model(model, train_loader, optimizer, num_epochs, device, writer):
    for epoch in range(num_epochs):
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
            optimizer.zero_grad()
            
            t = torch.randint(1, model.timesteps + 1, (clean_images.size(0),), device=device).float()
            denoised_images = model(clean_images, noisy_images, t)
            loss = charbonnier_loss(denoised_images, clean_images)
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Visualization
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                noisy_sample, clean_sample = next(iter(train_loader))
                noisy_sample, clean_sample = noisy_sample.to(device), clean_sample.to(device)
                denoised_sample = model(clean_sample, noisy_sample, torch.tensor([model.timesteps], device=device).float())
                
                grid = make_grid(torch.cat([clean_sample, noisy_sample, denoised_sample], dim=0), 
                                 nrow=clean_sample.size(0), normalize=True, value_range=(-1, 1))
                writer.add_image(f'Samples Epoch {epoch+1}', grid, epoch+1)
                model.train()

        writer.add_scalar('Loss/train', loss.item(), epoch)

def start_tensorboard(log_dir):
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

if __name__ == "__main__":
    # Initialize model, optimizer, and other components
    unet = UNet_S_Improved().to(device)
    diffusion_model = DiffusionModel(unet).to(device)
    optimizer = optim.AdamW(diffusion_model.parameters(), lr=1e-4)

    # Load data and start training
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=1, augment=True, dataset_percentage=0.1)

    log_dir = "runs/diffusion_model_denoising"
    writer = SummaryWriter(log_dir=log_dir)

    train_diffusion_model(diffusion_model, train_loader, optimizer, num_epochs=100, device=device, writer=writer)

    writer.close()
