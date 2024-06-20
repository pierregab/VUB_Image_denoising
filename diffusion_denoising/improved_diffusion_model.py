import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import optuna
import sys
import os
import psutil
import subprocess
from torchvision.utils import make_grid
import time
import torch.utils.checkpoint as cp

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Define the ChannelAttention class
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))  # Initialize to zeros
        self.beta = nn.Parameter(torch.ones(1))  # Initialize to zeros
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bn_out = self.bn(x)
        mu = torch.mean(bn_out, dim=[0, 2, 3], keepdim=True)
        var = torch.var(bn_out, dim=[0, 2, 3], keepdim=True)
        weights = self.gamma / torch.sqrt(var + 1e-5)
        normalized_bn_out = (bn_out - mu) / torch.sqrt(var + 1e-5)
        mc = self.sigmoid(weights * normalized_bn_out + self.beta)
        return mc * x

# Define the SpatialAttention class
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        ms = self.sigmoid(self.conv1(combined))
        return ms * x

# Define the CooperativeAttention class
class CooperativeAttention(nn.Module):
    def __init__(self, in_channels):
        super(CooperativeAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Define the ConvBlock class
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

# Define the ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.output_norm = nn.BatchNorm2d(in_channels)  # Add a BatchNorm layer for normalization

        # Initialize weights and biases
        self.initialize_weights()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.lrelu(out)
        return out

    def initialize_weights(self):
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)

# Define the DeconvBlock class
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeconvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

# Define the MultiScaleConv class
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        mid_channels = out_channels // 4  # Each branch will produce mid_channels
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=1, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, stride=1, padding=3)
        self.final_conv = nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)
        
        self.bn1x1 = nn.BatchNorm2d(mid_channels)
        self.bn3x3 = nn.BatchNorm2d(mid_channels)
        self.bn5x5 = nn.BatchNorm2d(mid_channels)
        self.bn7x7 = nn.BatchNorm2d(mid_channels)
        self.bn_final = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1x1 = self.bn1x1(self.conv1x1(x))
        out3x3 = self.bn3x3(self.conv3x3(x))
        out5x5 = self.bn5x5(self.conv5x5(x))
        out7x7 = self.bn7x7(self.conv7x7(x))
        concatenated = torch.cat([out1x1, out3x3, out5x5, out7x7], dim=1)
        return self.bn_final(self.final_conv(concatenated))

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        # Initial Conv Block
        self.initial_conv = MultiScaleConv(in_channels + 1, 64)  # Adapt input channels to include time step

        # Feature Domain Denoising Part
        self.denoising_blocks = nn.Sequential(*[ConvBlock(64, 64) for _ in range(8)])

        # One Convolution Block
        self.one_conv_block = ConvBlock(64, 64)

        # Cooperative Attention
        self.cooperative_attention = CooperativeAttention(64)

        # Residual Blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(9)])

        # Convolution Blocks leading to a single-channel output
        self.conv_blocks = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),  # reduce to 32 channels, kernel size 3
            ConvBlock(32, 16, kernel_size=3, stride=1, padding=1),  # reduce to 16 channels, kernel size 5
            ConvBlock(16, 8, kernel_size=3, stride=1, padding=1),   # reduce to 8 channels, kernel size 7
            ConvBlock(8, 4, kernel_size=3, stride=1, padding=1),    # reduce to 4 channels, kernel size 5
            ConvBlock(4, out_channels, kernel_size=1, stride=1, padding=0),  # finally reduce to out_channels, kernel size 1
        )

        # Final tanh activation for output
        self.final_activation = nn.Tanh()

    def forward(self, x, t):
        # Concatenate the time step with the input image
        t = t.expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat((x, t), dim=1)

        intermediate_outputs = {}

        # Initial Conv Block
        initial_conv_output = self.initial_conv(x)
        intermediate_outputs['initial_conv_output'] = initial_conv_output

        # Feature Domain Denoising
        denoising_output = self.denoising_blocks(initial_conv_output)
        intermediate_outputs['denoising_output'] = denoising_output

        # Subtract initial conv result from denoising output
        denoising_output = denoising_output - initial_conv_output

        # One Convolution Block
        one_conv_output = self.one_conv_block(denoising_output)
        intermediate_outputs['one_conv_output'] = one_conv_output

        # Cooperative Attention
        attention_output = self.cooperative_attention(one_conv_output)
        intermediate_outputs['attention_output'] = attention_output

        # Residual Blocks
        residual_output = self.residual_blocks(attention_output)
        intermediate_outputs['residual_output'] = residual_output

        # Add residual output to attention_output
        combined_output = residual_output + one_conv_output
        intermediate_outputs['combined_output'] = combined_output

        # Convolution Blocks leading to a single-channel output
        conv_output = self.conv_blocks(combined_output)
        intermediate_outputs['conv_output'] = conv_output

        # Add global cross-layer connection from input to the final output
        final_output = conv_output + x[:, :-1, :, :]  # exclude time step
        intermediate_outputs['final_output'] = final_output

        # Apply final tanh activation to map to pixel value range
        output = self.final_activation(final_output)
        intermediate_outputs['output'] = output

        return output, intermediate_outputs

# Define the DiffusionModel class
class DiffusionModel(nn.Module):
    def __init__(self, generator, timesteps=5):
        super(DiffusionModel, self).__init__()
        self.generator = generator
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
            x_t_gen, _ = self.generator(x_t, t_tensor)
            x_tilde = (1 - alpha_t) * x_t_gen + alpha_t * noisy_image
            t_tensor_prev = torch.tensor([(t - 1) / self.timesteps], device=noisy_image.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x_tilde_prev_gen, _ = self.generator(x_t, t_tensor_prev)
            x_tilde_prev = (1 - alpha_t_prev) * x_tilde_prev_gen + alpha_t_prev * noisy_image
            x_t = x_t - x_tilde + x_tilde_prev
        return x_t

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.improved_sampling(noisy_step_image)
        return denoised_image

# Define the charbonnier loss function
def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

# Define the denormalize function
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Define the training step function
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

# Define the training loop
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

# Define the function to start TensorBoard
def start_tensorboard(log_dir):
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

# Main function
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    log_dir = os.path.join("runs", "diffusion_checkpointed")
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)
    
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=1, augment=False, dataset_percentage=0.001)
    
    # Initialize the checkpointed model and optimizer with the adapted generator
    generator = Generator(in_channels=1, out_channels=1).to(device)
    model_checkpointed = DiffusionModel(generator).to(device)
    optimizer = optim.Adam(model_checkpointed.parameters(), lr=2e-4, betas=(0.9, 0.999))
    
    train_model_checkpointed(model_checkpointed, train_loader, optimizer, writer, num_epochs=10)
    writer.close()
