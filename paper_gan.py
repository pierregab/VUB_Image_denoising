import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Define the Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# Define the Spatial Attention Block
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

# Define the Cooperative Attention Block
class CooperativeAttention(nn.Module):
    def __init__(self, in_channels):
        super(CooperativeAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

# Define the Generator with updated architecture
class RCAGANGenerator(nn.Module):
    def __init__(self):
        super(RCAGANGenerator, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=7, padding=3)
        )
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(9)]
        )
        self.coop_attention = CooperativeAttention(64)
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(64, 1, kernel_size=1)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        features = self.feature_extraction(x)
        x = self.conv1(features)
        x = self.conv_blocks(x)
        x = self.coop_attention(x)
        x = self.deconv_blocks(x)
        x = self.tanh(x)
        return x

# Define the PatchGAN Discriminator with updated architecture
class PatchGAN(nn.Module):
    def __init__(self, in_channels=2):
        super(PatchGAN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 16 * 16, 1024)  # Corrected input size
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)


# Define the VGG Perceptual Loss (modified for single-channel)
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = nn.Sequential(*list(vgg.children())[:16])
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = x.expand(-1, 3, -1, -1)
        y = y.expand(-1, 3, -1, -1)
        x_vgg = self.layers(x)
        y_vgg = self.layers(y)
        return nn.functional.l1_loss(x_vgg, y_vgg)

# Define the Multimodal Loss Function
def multimodal_loss(gen_clean, clean, disc_output, real_labels, lambda_pixel, lambda_perceptual, lambda_texture, vgg_loss):
    # Adversarial Loss
    loss_adv = nn.BCELoss()(disc_output, real_labels)

    # Pixel-wise Loss
    loss_pixel = nn.L1Loss()(gen_clean, clean)

    # Perceptual Loss
    loss_perceptual = vgg_loss(gen_clean, clean)

    # Texture Loss
    loss_texture = edge_loss(gen_clean, clean)

    # Total Loss
    total_loss = lambda_pixel * loss_pixel + lambda_perceptual * loss_perceptual + lambda_texture * loss_texture + loss_adv
    return total_loss

# Define Edge Loss for Texture Preservation
def edge_loss(gen, clean):
    sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gen.device)
    sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gen.device)
    
    gen_x = F.conv2d(gen, sobel_kernel_x, padding=1)
    gen_y = F.conv2d(gen, sobel_kernel_y, padding=1)
    gen_edges = torch.sqrt(gen_x**2 + gen_y**2 + 1e-6)

    clean_x = F.conv2d(clean, sobel_kernel_x, padding=1)
    clean_y = F.conv2d(clean, sobel_kernel_y, padding=1)
    clean_edges = torch.sqrt(clean_x**2 + clean_y**2 + 1e-6)
    
    return nn.functional.l1_loss(gen_edges, clean_edges)

# Denormalize function
def denormalize(tensor):
    return tensor * 0.5 + 0.5

def train_rca_gan(train_loader, val_loader, num_epochs=200, lambda_pixel=100, lambda_perceptual=0.1, lambda_texture=1.0,
                  lr=0.0001, betas=(0.5, 0.999), device=torch.device("cuda" if torch.cuda.is_available() else "mps")):
    # Initialize the model
    generator = RCAGANGenerator().to(device)
    discriminator = PatchGAN().to(device)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Loss functions
    vgg_loss = VGGPerceptualLoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

    global_step = 0

    for epoch in range(num_epochs):
        for i, (noisy_images, clean_images) in enumerate(train_loader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Generate clean images
            gen_clean = generator(noisy_images)
            
            # Prepare real and fake data for discriminator
            real_data = torch.cat((noisy_images, clean_images), 1)
            fake_data = torch.cat((noisy_images, gen_clean.detach()), 1)
            
            # Get discriminator outputs
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data)
            
            # Match label size with discriminator output size
            real_labels = torch.ones_like(real_output) * 0.9
            fake_labels = torch.zeros_like(real_output) * 0.1

            # Calculate discriminator loss
            d_loss_real = nn.BCELoss()(real_output, real_labels)
            d_loss_fake = nn.BCELoss()(fake_output, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_data = torch.cat((noisy_images, gen_clean), 1)
            disc_output = discriminator(fake_data)
            
            # Match label size with discriminator output size for generator
            real_labels = torch.ones_like(disc_output) * 0.9
            
            g_loss = multimodal_loss(gen_clean, clean_images, disc_output, real_labels, lambda_pixel, lambda_perceptual, lambda_texture, vgg_loss)
            g_loss.backward()
            optimizer_G.step()

            # Print training progress
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            # Log batch losses to TensorBoard
            writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
            writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
            global_step += 1
            
        # Log example images to TensorBoard at the end of each epoch
        with torch.no_grad():
            example_noisy = noisy_images[:4].cpu()  # Take first 4 examples from the last batch
            example_clean = clean_images[:4].cpu()
            example_gen = generator(example_noisy.to(device)).cpu()
            
            # Denormalize images to [0, 1] for better visualization
            example_noisy = denormalize(example_noisy)
            example_clean = denormalize(example_clean)
            example_gen = denormalize(example_gen)

            # Log images
            writer.add_images('Noisy Images', example_noisy, epoch)
            writer.add_images('Generated Images', example_gen, epoch)
            writer.add_images('Clean Images', example_clean, epoch)

        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

    print("Training finished.")
    writer.close()
