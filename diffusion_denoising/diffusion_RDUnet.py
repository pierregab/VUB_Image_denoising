import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import subprocess
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.utils.checkpoint as cp
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_msssim
from Unet.Unet_model import RDUNet_T, init_weights

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data as load_div2k_data
from dataset_creation.SIDD_dataset import load_data as load_sidd_data  # Assuming your new SIDD data loader script is named SIDD_dataset.py

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=20):
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

def combined_loss(pred, target, mse_weight=0, charbonnier_weight=1, ssim_weight=0, epsilon=1e-3):
    mse_loss = nn.MSELoss()(pred, target)
    charbonnier = charbonnier_loss(pred, target, epsilon)
    ssim_loss = 1 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)

    return mse_weight * mse_loss + charbonnier_weight * charbonnier + ssim_weight * ssim_loss

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
    if (checkpoint_path is not None) and os.path.isfile(checkpoint_path):
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
        # Argument parser for command line arguments
        parser = argparse.ArgumentParser(description="Train a diffusion model with optional optimizer and scheduler choice.")
        parser.add_argument('--dataset_choice', type=str, default='SIDD', choices=['DIV2K', 'SIDD'], help="Choice of dataset (DIV2K or SIDD)")
        parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to model checkpoint")
        parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs to train")
        parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
        parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading")
        parser.add_argument('--validation_split', type=float, default=0.2, help="Validation split percentage")
        parser.add_argument('--augment', action='store_false', help="Use data augmentation")
        parser.add_argument('--dataset_percentage', type=float, default=0.1, help="Percentage of dataset to use for training")
        parser.add_argument('--base_filters', type=int, default=32, help="Base number of filters for the model")
        parser.add_argument('--timesteps', type=int, default=20, help="Number of timesteps for the diffusion model")
        parser.add_argument('--optimizer_choice', type=str, default='adamw', choices=['adam', 'adamw'], help="Choice of optimizer (adam or adamw)")
        parser.add_argument('--scheduler_choice', type=str, default='step', choices=['cosine', 'step'], help="Choice of scheduler (cosine or step)")
        
        args = parser.parse_args()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        log_dir = os.path.join("runs", "diffusion_checkpointed")
        writer = SummaryWriter(log_dir=log_dir)
        start_tensorboard(log_dir)

        image_folder = 'DIV2K_train_HR.nosync' if args.dataset_choice == 'DIV2K' else 'SIDD_dataset.nosync/SIDD_Medium_Srgb'
        
        # Load data based on the selected dataset
        if args.dataset_choice == 'DIV2K':
            train_loader, val_loader = load_div2k_data(image_folder, batch_size=args.batch_size, augment=args.augment, dataset_percentage=args.dataset_percentage, validation_split=args.validation_split, use_rgb=True, num_workers=args.num_workers)
        elif args.dataset_choice == 'SIDD':
            train_loader, val_loader = load_sidd_data(image_folder, batch_size=args.batch_size, augment=args.augment, dataset_percentage=args.dataset_percentage, validation_split=args.validation_split, use_rgb=True, num_workers=args.num_workers)
        
        # Initialize the model with the configured parameters
        unet_checkpointed = RDUNet_T(base_filters=args.base_filters).to(device)
        model_checkpointed = DiffusionModel(unet_checkpointed, timesteps=args.timesteps).to(device)
        
        # Apply weight initialization
        unet_checkpointed.apply(init_weights())
        
        # Define the optimizer
        if args.optimizer_choice == 'adam':
            optimizer = optim.Adam(model_checkpointed.parameters(), lr=2e-4, betas=(0.9, 0.999))
            scheduler = CosineAnnealingLR(optimizer, T_max=10)  # Default scheduler
        elif args.optimizer_choice == 'adamw':
            optimizer = optim.AdamW(model_checkpointed.parameters(), lr=1e-4, weight_decay=1e-5)
            scheduler_step = 3
            scheduler_gamma = 0.5
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        # Load checkpoint if provided
        start_epoch = load_checkpoint(model_checkpointed, optimizer, scheduler, args.checkpoint_path)
        
        train_model_checkpointed(model_checkpointed, train_loader, val_loader, optimizer, scheduler, writer, num_epochs=args.num_epochs, start_epoch=start_epoch)
        writer.close()
    except Exception as e:
        print(f"An error occurred: {e}")
