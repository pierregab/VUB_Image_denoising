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
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_msssim

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class UNet_S_Checkpointed(nn.Module):
    def __init__(self):
        super(UNet_S_Checkpointed, self).__init__()
        self.enc1 = self.conv_block(4, 32)  # Change input channels to 4 (3 for RGB image + 1 for time step)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.upconv3 = self.upconv(128, 64)
        self.upconv2 = self.upconv(64, 32)
        self.dec3 = self.conv_block(128, 64)
        self.dec2 = self.conv_block(64, 32)
        self.dec1 = self.conv_block(32, 3, final_layer=True)  # Change output channels to 3 for RGB

    def conv_block(self, in_channels, out_channels, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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

def combined_loss(pred, target, mse_weight=0.5, charbonnier_weight=0.3, ssim_weight=0.2, epsilon=1e-3):
    mse_loss = nn.MSELoss()(pred, target)
    charbonnier = charbonnier_loss(pred, target, epsilon)
    ssim_loss = 1 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)

    return mse_weight * mse_loss + charbonnier_weight * charbonnier + ssim_weight * ssim_loss

# Define the checkpointed model and optimizer
unet_checkpointed = UNet_S_Checkpointed().to(device)
model_checkpointed = DiffusionModel(unet_checkpointed).to(device)
optimizer = optim.Adam(model_checkpointed.parameters(), lr=2e-4, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=10)

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Sample training loop
def train_step_checkpointed(model, clean_images, noisy_images, optimizer):
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
    optimizer.step()

    
    return loss.item()


def train_model_checkpointed(model, train_loader, optimizer, writer, scheduler, start_epoch=0, num_epochs=10):
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            loss = train_step_checkpointed(model, clean_images, noisy_images, optimizer)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss:.4f}")
            
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
        
        # Validation phase
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch_idx, (noisy_images, clean_images) in enumerate(val_loader):
                noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

                # Generate denoised images using only the noisy images
                denoised_images = model.improved_sampling(noisy_images)
                
                # Calculate validation loss
                loss = combined_loss(denoised_images, clean_images)
                validation_loss += loss.item()

                # Denormalize images for visualization
                clean_images = denormalize(clean_images.cpu())
                noisy_images = denormalize(noisy_images.cpu())
                denoised_images = denormalize(denoised_images.cpu())
                
                # Create image grids
                grid_clean = make_grid(clean_images[:10], nrow=4, normalize=True)  # Only show 10 images
                grid_noisy = make_grid(noisy_images[:10], nrow=4, normalize=True)
                grid_denoised = make_grid(denoised_images[:10], nrow=4, normalize=True)
                
                # Add images to TensorBoard for the first batch
                if batch_idx == 0:
                    writer.add_image(f'Epoch_{epoch + 1}/Clean Images', grid_clean, epoch + 1)
                    writer.add_image(f'Epoch_{epoch + 1}/Noisy Images', grid_noisy, epoch + 1)
                    writer.add_image(f'Epoch_{epoch + 1}/Denoised Images', grid_denoised, epoch + 1)
             

        # Log validation loss
        validation_loss /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {validation_loss:.4f}")
        writer.add_scalar('Loss/validation', validation_loss, epoch + 1)
        
        writer.flush()

        scheduler.step()

        # Save the model checkpoint after each epoch
        checkpoint_path = os.path.join("checkpoints", f"diffusion_model_checkpointed_epoch_{epoch + 1}.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint.get('epoch', 0)})")
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    log_dir = os.path.join("runs", "diffusion_checkpointed")
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)
    
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=64, augment=False, dataset_percentage=0.1, validation_split=0.1, use_rgb=True)
    
    checkpoint_path = "checkpoints/diffusion_model_checkpointed_epoch_50.pth"  # Adjust the path as needed
    start_epoch = load_checkpoint(model_checkpointed, optimizer, checkpoint_path)
    
    num_epochs = 50  # Total number of epochs you want to train for
    train_model_checkpointed(model_checkpointed, train_loader, optimizer, writer, scheduler, start_epoch, num_epochs)
    writer.close()
