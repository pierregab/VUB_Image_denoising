import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import subprocess
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_msssim

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

import torch
import torch.nn as nn

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.relu(self.bn(self.conv(x)))
        return torch.cat([x, skip], 1)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=32):
        super(SimpleUNet, self).__init__()
        
        self.inc = SimpleBlock(in_channels, base_filters)
        self.down1 = DownsampleBlock(base_filters, base_filters*2)
        self.down2 = DownsampleBlock(base_filters*2, base_filters*4)
        
        self.middle = SimpleBlock(base_filters*4, base_filters*4)
        
        self.up2 = UpsampleBlock(base_filters*4, base_filters*2)
        self.up1 = UpsampleBlock(base_filters*4, base_filters)
        
        self.outc = nn.Conv2d(base_filters*2, out_channels, 1)

    def forward(self, x, t):
        t = t.expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat((x, t), dim=1)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x3 = self.middle(x3)
        
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        
        return self.outc(x)

class SimpleDiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=50):
        super(SimpleDiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps

    def forward_diffusion(self, clean_image, noisy_image, t):
        alpha = t / self.timesteps
        return alpha * noisy_image + (1 - alpha) * clean_image

    def sampling(self, noisy_image):
        x_t = noisy_image
        for t in reversed(range(1, self.timesteps + 1)):
            t_tensor = torch.tensor([t / self.timesteps], device=noisy_image.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x_t = self.unet(x_t, t_tensor)
        return x_t

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.sampling(noisy_step_image)
        return denoised_image

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

def combined_loss(pred, target, mse_weight=0, charbonnier_weight=1, ssim_weight=0, epsilon=1e-3):
    mse_loss = nn.MSELoss()(pred, target)
    charbonnier = charbonnier_loss(pred, target, epsilon)
    ssim_loss = 1 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)

    return mse_weight * mse_loss + charbonnier_weight * charbonnier + ssim_weight * ssim_loss

# Define the checkpointed model and optimizer
unet_checkpointed = SimpleUNet(base_filters=64).to(device)
model_checkpointed = SimpleDiffusionModel(unet_checkpointed).to(device)
optimizer = optim.Adam(model_checkpointed.parameters(), lr=2e-4, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=10)

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Sample training loop
def train_step_checkpointed(model, clean_images, noisy_images, optimizer, accumulation_steps, clip_value=0.1):
    model.train()
    optimizer.zero_grad()
    
    batch_size = clean_images.size(0)
    timesteps = model.timesteps
    
    # Sample a random timestep uniformly for each image in the batch
    t = torch.randint(0, timesteps + 1, (batch_size,), device=clean_images.device).float()
    
    # Normalize the timestep
    t_normalized = t / timesteps
    
    # Expand the timestep tensor to match the image dimensions
    t_tensor = t_normalized.view(batch_size, 1, 1, 1).expand(-1, 1, clean_images.size(2), clean_images.size(3))
    
    # Move t_tensor to the same device as the model
    t_tensor = t_tensor.to(device)
    
    # Interpolate the noisy image
    alpha = t_tensor
    interpolated_images = alpha * noisy_images + (1 - alpha) * clean_images
    
    # Move interpolated_images to the same device as the model
    interpolated_images = interpolated_images.to(device)
    
    # Denoise the interpolated image
    denoised_images = model.unet(interpolated_images, t_tensor)
    
    # Calculate the combined loss
    loss = combined_loss(denoised_images, clean_images.to(device))
    loss.backward()
    
    # Clip the gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    
    return loss.item()

def train_model_checkpointed(model, train_loader, val_loader, optimizer, scheduler, writer, num_epochs=10, start_epoch=0, accumulation_steps=4, clip_value=1.0):
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            loss = train_step_checkpointed(model, clean_images, noisy_images, optimizer, accumulation_steps, clip_value)
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss:.4f}")
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
        
        # Validation phase (on a single batch)
        model.eval()
        with torch.no_grad():
            # Get a single batch from the validation loader
            val_noisy_images, val_clean_images = next(iter(val_loader))
            val_noisy_images, val_clean_images = val_noisy_images.to(device), val_clean_images.to(device)

            # Generate denoised images using only the noisy images
            denoised_images = model.sampling(val_noisy_images)  # Updated method call
            
            # Calculate validation loss
            validation_loss = combined_loss(denoised_images, val_clean_images)

            # Denormalize images for visualization
            val_clean_images = denormalize(val_clean_images.cpu())
            val_noisy_images = denormalize(val_noisy_images.cpu())
            denoised_images = denormalize(denoised_images.cpu())
            
            # Create image grids
            grid_clean = make_grid(val_clean_images[:10], nrow=4, normalize=True)  # Only show 10 images
            grid_noisy = make_grid(val_noisy_images[:10], nrow=4, normalize=True)
            grid_denoised = make_grid(denoised_images[:10], nrow=4, normalize=True)
            
            # Add images to TensorBoard
            writer.add_image(f'Epoch_{epoch + 1}/Clean Images', grid_clean, epoch + 1)
            writer.add_image(f'Epoch_{epoch + 1}/Noisy Images', grid_noisy, epoch + 1)
            writer.add_image(f'Epoch_{epoch + 1}/Denoised Images', grid_denoised, epoch + 1)

        # Log validation loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {validation_loss:.4f}")
        writer.add_scalar('Loss/validation', validation_loss, epoch + 1)
        
        writer.flush()

        scheduler.step()

        # Save the model checkpoint after each epoch
        checkpoint_path = os.path.join("checkpoints", f"diffusion_SimplifiedUNet_model_checkpointed_epoch_{epoch + 1}.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        # load on cuda or mps device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0

def start_tensorboard(log_dir):
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        log_dir = os.path.join("runs", "diffusion_checkpointed")
        writer = SummaryWriter(log_dir=log_dir)
        start_tensorboard(log_dir)
        
        image_folder = 'DIV2K_train_HR.nosync'
        train_loader, val_loader = load_data(image_folder, batch_size=16, augment=False, dataset_percentage=0.1, validation_split=0.1, use_rgb=True, num_workers=8)
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join("checkpoints", "diffusion_SimplifiedUNet_model_checkpointed_epoch_7.pth")
        start_epoch = load_checkpoint(model_checkpointed, optimizer, scheduler, checkpoint_path)
        
        train_model_checkpointed(model_checkpointed, train_loader, val_loader, optimizer, scheduler, writer, num_epochs=300, start_epoch=start_epoch)
        writer.close()
    except Exception as e:
        print(f"An error occurred: {e}")
