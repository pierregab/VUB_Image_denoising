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
import torchvision.models as models
import torch.nn.functional as F

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
    
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:29].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)
        return self.mse(vgg_input, vgg_target)

perceptual_loss = VGGPerceptualLoss().to(device)

def total_variation(image):
    # Compute differences
    diff_h = image[:, :, :, 1:] - image[:, :, :, :-1]
    diff_v = image[:, :, 1:, :] - image[:, :, :-1, :]
    
    # Sum of absolute differences
    tv_h = torch.abs(diff_h).sum()
    tv_v = torch.abs(diff_v).sum()
    
    # Normalize by the number of elements
    num_elements = image.numel()
    return (tv_h + tv_v) / num_elements

def total_variation_loss(pred, target):
    # Compute total variation for both predicted and target images
    tv_pred = total_variation(pred)
    tv_target = total_variation(target)
    
    # Compute the absolute difference between the two total variations
    return torch.abs(tv_pred - tv_target)

def histogram_matching_loss(pred, target, num_bins=256):
    pred_hist = torch.histc(pred, bins=num_bins, min=0, max=1)
    target_hist = torch.histc(target, bins=num_bins, min=0, max=1)
    
    pred_hist = pred_hist / torch.sum(pred_hist)
    target_hist = target_hist / torch.sum(target_hist)
    
    return torch.mean((pred_hist - target_hist) ** 2)

def local_contrast_norm(x, kernel_size=3):
    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)
    mean_sq = F.avg_pool2d(x**2, kernel_size, stride=1, padding=kernel_size//2)
    var = mean_sq - mean**2
    std = torch.sqrt(var + 1e-6)
    return (x - mean) / std

def local_contrast_loss(pred, target):
    pred_norm = local_contrast_norm(pred)
    target_norm = local_contrast_norm(target)
    return F.mse_loss(pred_norm, target_norm)

def dynamic_range_loss(pred, target):
    pred_range = torch.max(pred) - torch.min(pred)
    target_range = torch.max(target) - torch.min(target)
    return torch.abs(pred_range - target_range)

def gradient_distribution_loss(pred, target):
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    pred_grad_dist = torch.cat([pred_grad_x.flatten(), pred_grad_y.flatten()])
    target_grad_dist = torch.cat([target_grad_x.flatten(), target_grad_y.flatten()])
    
    return F.kl_div(F.log_softmax(pred_grad_dist, dim=0), F.softmax(target_grad_dist, dim=0))

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

def contrast_difference_loss(pred, target):
    pred_contrast = torch.mean(torch.abs(pred - torch.mean(pred)))
    target_contrast = torch.mean(torch.abs(target - torch.mean(target)))
    return torch.abs(pred_contrast - target_contrast)

def combined_loss(pred, target):
    mse_loss = nn.MSELoss()(pred, target)
    ssim_loss = 1 - pytorch_msssim.ssim(pred, target)
    tv_loss = total_variation(pred)
    perc_loss = perceptual_loss(pred, target)
    
    # New contrast-related losses
    hist_match_loss = histogram_matching_loss(pred, target)
    contrast_diff_loss = contrast_difference_loss(pred, target)
    local_contrast_loss_term = local_contrast_loss(pred, target)
    dynamic_range_loss_term = dynamic_range_loss(pred, target)
    grad_dist_loss = gradient_distribution_loss(pred, target)
    
    combined = (1e2 * mse_loss + 1e3 * ssim_loss + 5e1 * tv_loss + 0.1 * perc_loss +
                1e4 * hist_match_loss + 1e1 * contrast_diff_loss + 
                0.1 * local_contrast_loss_term + 1e2 * dynamic_range_loss_term +
                1e9 * grad_dist_loss)
    
    return combined, mse_loss, ssim_loss, tv_loss, perc_loss, hist_match_loss, contrast_diff_loss, local_contrast_loss_term, dynamic_range_loss_term, grad_dist_loss

# Define the checkpointed model and optimizer
unet_checkpointed = UNet_S_Checkpointed().to(device)
model_checkpointed = DiffusionModel(unet_checkpointed).to(device)
optimizer = optim.Adam(model_checkpointed.parameters(), lr=2e-4, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=10)


def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Sample training loop
def train_step_checkpointed(model, clean_images, noisy_images, optimizer, writer, epoch, batch_idx, train_loader_length):
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
    losses = combined_loss(denoised_images, clean_images.to(device))
    loss = losses[0]
    loss.backward()
    optimizer.step()

    # Log all individual losses
    loss_names = ['combined_loss', 'mse_loss', 'ssim_loss', 'tv_loss', 'perc_loss', 'hist_match_loss', 
                  'contrast_diff_loss', 'local_contrast_loss_term', 'dynamic_range_loss_term', 'grad_dist_loss']
    for i, loss_name in enumerate(loss_names):
        writer.add_scalar(f'Loss/train_{loss_name}', losses[i].item(), epoch * train_loader_length + batch_idx)
    
    return loss.item()


def train_model_checkpointed(model, train_loader, val_loader, optimizer, scheduler, writer, num_epochs=10, start_epoch=0):
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            loss = train_step_checkpointed(model, clean_images, noisy_images, optimizer, writer, epoch, batch_idx, len(train_loader))
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
            losses = combined_loss(denoised_images, val_clean_images)
            validation_loss = losses[0]

            # Log validation losses
            loss_names = ['combined_loss', 'mse_loss', 'ssim_loss', 'tv_loss', 'perc_loss', 'hist_match_loss', 
                          'contrast_diff_loss', 'local_contrast_loss_term', 'dynamic_range_loss_term', 'grad_dist_loss']
            for i, loss_name in enumerate(loss_names):
                writer.add_scalar(f'Loss/validation_{loss_name}', losses[i].item(), epoch + 1)

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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {validation_loss:.4f}")
        
        writer.flush()

        scheduler.step()

        # Save the model checkpoint after each epoch
        checkpoint_path = os.path.join("checkpoints", f"diffusion_tv_model_checkpointed_epoch_{epoch + 1}.pth")
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
        train_loader, val_loader = load_data(image_folder, batch_size=32, augment=False, dataset_percentage=0.1, validation_split=0.1, use_rgb=True, num_workers=8)
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join("checkpoints", "diffusion_tv_model_checkpointed_epoch_2000.pth")
        start_epoch = load_checkpoint(model_checkpointed, optimizer, scheduler, checkpoint_path)
        
        train_model_checkpointed(model_checkpointed, train_loader, val_loader, optimizer, scheduler, writer, num_epochs=200, start_epoch=start_epoch)
        writer.close()
    except Exception as e:
        print(f"An error occurred: {e}")
 