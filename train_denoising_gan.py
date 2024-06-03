import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import torch.nn.utils.spectral_norm as spectral_norm
from tqdm import tqdm
import torch.nn.functional as F

# Attention Block for the U-Net
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# U-Net Model with Attention Blocks
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = self.upconv(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = self.conv_block(64, 1, final_layer=True)
        self.att4 = AttentionBlock(256, 256, 128)
        self.att3 = AttentionBlock(128, 128, 64)
        self.att2 = AttentionBlock(64, 64, 32)

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

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        upconv4 = self.upconv4(enc4)
        att4 = self.att4(enc3, upconv4)
        dec4 = self.dec4(torch.cat((upconv4, att4), dim=1))
        upconv3 = self.upconv3(dec4)
        att3 = self.att3(enc2, upconv3)
        dec3 = self.dec3(torch.cat((upconv3, att3), dim=1))
        upconv2 = self.upconv2(dec3)
        att2 = self.att2(enc1, upconv2)
        dec2 = self.dec2(torch.cat((upconv2, att2), dim=1))
        dec1 = self.dec1(dec2)
        return dec1

# PatchGAN Discriminator
class PatchGAN(nn.Module):
    def __init__(self, in_channels=2):
        super(PatchGAN, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Perceptual Loss using VGG19
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = nn.Sequential(*list(vgg.children())[:16])
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.layers(x)
        y_vgg = self.layers(y)
        return nn.functional.l1_loss(x_vgg, y_vgg)

# Gradient Penalty for WGAN
def compute_gradient_penalty(D, noisy_samples, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(noisy_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = torch.cat((noisy_samples, interpolates), 1)  # Concatenate along channel dimension
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(noisy_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# Denormalize function
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Edge Loss function
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

def train_denoising_gan(train_loader, val_loader, num_epochs=200, lambda_pixel=100, lambda_perceptual=0.1, lambda_edge=1.0, lambda_gp=10,
                        lr=0.0001, betas=(0.5, 0.999), device=torch.device("cpu"), log_dir='runs/denoising_gan',
                        checkpoint_dir='checkpoints', checkpoint_prefix='denoising_gan'):
    # Initialize the model
    generator = UNet().to(device)
    discriminator = PatchGAN().to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal) 

    # Loss functions
    criterion_GAN = nn.BCELoss()
    criterion_pixelwise = nn.L1Loss()
    vgg_loss = VGGPerceptualLoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    for epoch in range(num_epochs):
        for i, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Generate a batch of images
            gen_clean = generator(noisy)
            
            # Convert single channel images to three channels for perceptual loss
            gen_clean_3ch = gen_clean.repeat(1, 3, 1, 1)
            clean_3ch = clean.repeat(1, 3, 1, 1)
            
            # Determine the size of the PatchGAN output
            pred_fake = discriminator(torch.cat((noisy, gen_clean), 1))
            patch_size = pred_fake.size()
            
            # Adversarial ground truths with label smoothing
            valid = torch.ones(patch_size, requires_grad=False).to(device) * 0.9
            fake = torch.zeros(patch_size, requires_grad=False).to(device) * 0.1
            
            # ------------------
            # Train Discriminator
            # ------------------
            
            optimizer_D.zero_grad()
            
            # Real loss
            pred_real = discriminator(torch.cat((noisy, clean), 1))
            loss_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = discriminator(torch.cat((noisy, gen_clean.detach()), 1))
            loss_fake = criterion_GAN(pred_fake, fake)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, noisy.data, clean.data, gen_clean.data)
            
            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake) + lambda_gp * gradient_penalty
            
            loss_D.backward()
            # Gradient clipping for discriminator
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            
            # ------------------
            # Train Generators
            # ------------------
            
            optimizer_G.zero_grad()
            
            # Adversarial loss
            pred_fake = discriminator(torch.cat((noisy, gen_clean), 1))
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(gen_clean, clean)
            
            # Perceptual loss
            loss_perceptual = vgg_loss(gen_clean_3ch, clean_3ch)
            
            # Edge loss
            loss_edge = edge_loss(gen_clean, clean)
            
            # Total generator loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel + lambda_perceptual * loss_perceptual + lambda_edge * loss_edge
            
            loss_G.backward()
            # Gradient clipping for generator
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            # Print training progress
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

            # Log batch losses to TensorBoard
            writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
            writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)
            global_step += 1
            
        # Log example images to TensorBoard at the end of each epoch
        writer.add_images('Noisy Images', denormalize(noisy), epoch)
        writer.add_images('Generated Clean Images', denormalize(gen_clean), epoch)
        writer.add_images('Ground Truth Clean Images', denormalize(clean), epoch)
        
        # Save model checkpoints
        torch.save(generator.state_dict(), f"{checkpoint_dir}/{checkpoint_prefix}_generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"{checkpoint_dir}/{checkpoint_prefix}_discriminator_{epoch}.pth")

        # Update learning rate schedulers
        scheduler_G.step()
        scheduler_D.step()

    # Close the TensorBoard writer
    writer.close()
