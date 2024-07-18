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

@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))

class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))

class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))

class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)
        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))
        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))
        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))
        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))
        return out_3 + x

# Modified RDUNet to handle timestep input
class RDUNet_T(nn.Module):
    def __init__(self, channels=4, base_filters=64):  # channels=4 to include the timestep channel
        super(RDUNet_T, self).__init__()
        filters_0 = base_filters
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        self.input_block = InputBlock(channels, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, 3)

        self.apply(init_weights())  # Apply the weight initialization

    def forward(self, inputs, t):
        # Concatenate the time step with the input image
        t = t.expand(inputs.size(0), 1, inputs.size(2), inputs.size(3))
        x = torch.cat((inputs, t), dim=1)

        out_0 = self.input_block(x)
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)

        out_4 = self.up_2([out_3, out_2])
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)

        return self.output_block(out_6) + inputs

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=20):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps

    def forward_diffusion(self, clean_image, noisy_image, t):
        alpha = t / self.timesteps
        interpolated_image = alpha * noisy_image + (1 - alpha) * clean_image
        return interpolated_image

    def direct_sampling(self, noisy_image):
        t = torch.tensor([1.0], device=noisy_image.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        denoised_image = self.unet(noisy_image, t)
        return denoised_image

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.direct_sampling(noisy_step_image)
        return denoised_image

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

def combined_loss(pred, target, mse_weight=0, charbonnier_weight=1, ssim_weight=0, epsilon=1e-3):
    mse_loss = nn.MSELoss()(pred, target)
    charbonnier = charbonnier_loss(pred, target, epsilon)
    ssim_loss = 1 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)

    return mse_weight * mse_loss + charbonnier_weight * charbonnier + ssim_weight * ssim_loss

# Define the checkpointed model and optimizer
unet_checkpointed = RDUNet_T(base_filters=32).to(device)
model_checkpointed = DiffusionModel(unet_checkpointed).to(device)
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
            denoised_images = model.improved_sampling(val_noisy_images)
            
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
        checkpoint_path = os.path.join("checkpoints", f"diffusion_RDUnet_model_checkpointed_epoch_{epoch + 1}.pth")
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
        train_loader, val_loader = load_data(image_folder, batch_size=4, augment=False, dataset_percentage=0.01, validation_split=0.1, use_rgb=True, num_workers=8)
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join("checkpoints", "diffusion_RDUnet_model_checkpointed_epoch_89.pth")
        start_epoch = load_checkpoint(model_checkpointed, optimizer, scheduler, checkpoint_path)
        
        train_model_checkpointed(model_checkpointed, train_loader, val_loader, optimizer, scheduler, writer, num_epochs=300, start_epoch=start_epoch)
        writer.close()
    except Exception as e:
        print(f"An error occurred: {e}")
