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

class RDUNet(nn.Module):
    def __init__(self, channels=3, base_filters=64):
        super(RDUNet, self).__init__()
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

        self.output_block = OutputBlock(filters_0, channels)

        self.apply(init_weights())  # Apply the weight initialization

    def forward(self, inputs):
        out_0 = self.input_block(inputs)
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

# Define the model and optimizer
unet = RDUNet(base_filters=128).to(device)  # Using base filters 128 as per the paper
optimizer = optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-5)  # Adjusting optimizer parameters

# Scheduler
scheduler_step = 3  # Number of epochs before each scheduler step as per the paper
scheduler_gamma = 0.5  # Decay factor
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Sample training loop with gradient accumulation
def train_step(model, clean_images, noisy_images, optimizer, criterion, batch_idx, accumulation_steps):
    model.train()
    
    clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
    
    denoised_images = model(noisy_images)
    loss = criterion(denoised_images, clean_images)
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping here
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item()

def train_model(model, train_loader, optimizer, scheduler, writer, num_epochs=10, accumulation_steps=4):
    criterion = nn.L1Loss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            loss = train_step(model, clean_images, noisy_images, optimizer, criterion, batch_idx, accumulation_steps)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss:.4f}")
            
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
                clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
                denoised_images = model(noisy_images)
                
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
        scheduler.step()  # Update the learning rate

        if (epoch + 1) % 5 == 0:
    
            # Save the model checkpoint
            checkpoint_path = os.path.join("checkpoints", "rdunet_denoising.pth")
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

    log_dir = os.path.join("runs", "rdunet_denoising")
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)
    
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=4, augment=False, dataset_percentage=0.1, use_rgb=True, num_workers=10)  # Adjusting batch size
    train_model(unet, train_loader, optimizer, scheduler, writer, num_epochs=21, accumulation_steps=4)  # Adjusting number of epochs
    writer.close()
