import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

###################### DIFFUSION MODEL ######################

# Gaussian Diffusion Model

class GaussianDiffusion(nn.Module):
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_timesteps=1000):
        super(GaussianDiffusion, self).__init__()
        self.num_timesteps = num_timesteps
        self.beta_schedule = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.posterior_variance = self.beta_schedule * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.sqrt_alpha_cumprod[t].to(x_start.device) * x_start +
            self.sqrt_one_minus_alpha_cumprod[t].to(x_start.device) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            (x_t - self.sqrt_one_minus_alpha_cumprod[t].to(x_t.device) * noise) /
            self.sqrt_alpha_cumprod[t].to(x_t.device)
        )

    def p_losses(self, model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)
        return F.mse_loss(noise, predicted_noise)

    def get_noised_tensor(self, model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        return model(x_noisy, t)

    def forward(self, model, x_start, t, noise=None):
        return self.p_losses(model, x_start, t, noise=noise)
    
# UNet Model

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, t):
        print("UNet input shape:", x.shape)
        x = self.encoder(x)
        print("After UNet encoder:", x.shape)
        x = self.middle(x)
        print("After UNet middle:", x.shape)
        x = self.decoder(x)
        print("After UNet decoder:", x.shape)
        return x
    
###################### ATTENTION STRUCTURES ######################

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bn_out = self.bn(x)
        mu = torch.mean(bn_out, dim=[0, 2, 3], keepdim=True)
        var = torch.var(bn_out, dim=[0, 2, 3], keepdim=True)
        weights = self.gamma / torch.sqrt(var + 1e-5)
        normalized_bn_out = (bn_out - mu) / torch.sqrt(var + 1e-5)
        mc = self.sigmoid(weights * normalized_bn_out + self.beta)
        return mc * x

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

class CooperativeAttention(nn.Module):
    def __init__(self, in_channels):
        super(CooperativeAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    
###################### BLOCKS FOR GAN STRUCTURE ######################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeconvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))
    
###################### GAN STRUCTURE ######################

# Generator

class GeneratorWithDiffusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GeneratorWithDiffusion, self).__init__()

        # Feature Extraction Part
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

        # Feature Domain Denoising Part
        self.denoising_blocks = nn.Sequential(*[ConvBlock(64, 64) for _ in range(8)])

        # One Convolution Block
        self.one_conv_block = ConvBlock(64, 64)

        # Cooperative Attention
        self.cooperative_attention = CooperativeAttention(64)

        # Diffusion Model Integration
        self.diffusion_model = GaussianDiffusion()
        self.unet = UNet(64, 64)

        # Residual Blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(9)])

        # Deconvolution Blocks
        deconv_blocks = [DeconvBlock(64, 64, kernel_size=3) for _ in range(4)]
        final_deconv_block = DeconvBlock(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.deconv_blocks = nn.Sequential(*deconv_blocks, final_deconv_block)

        # Final tanh activation for output
        self.final_activation = nn.Tanh()

    def forward(self, x):
        print("Input shape:", x.shape)
        
        # Feature Extraction
        feature_extraction_output = self.feature_extraction(x)
        print("After feature extraction:", feature_extraction_output.shape)
        
        # Feature Domain Denoising
        denoising_output = self.denoising_blocks(feature_extraction_output)
        print("After denoising blocks:", denoising_output.shape)
        
        # Subtract feature extraction result from denoising output
        denoising_output = feature_extraction_output - denoising_output
        print("After feature extraction subtraction:", denoising_output.shape)
        
        # One Convolution Block
        conv_block_output = self.one_conv_block(denoising_output)
        print("After one convolution block:", conv_block_output.shape)
        
        # Cooperative Attention
        attention_output = self.cooperative_attention(conv_block_output)
        print("After cooperative attention:", attention_output.shape)
        
        # Diffusion Model Integration
        timesteps = torch.randint(0, self.diffusion_model.num_timesteps, (x.size(0),), device=x.device).long()
        diffusion_output = self.diffusion_model.get_noised_tensor(self.unet, attention_output, timesteps)
        print("After diffusion model integration:", diffusion_output.shape)
        
        # Residual Blocks
        residual_output = self.residual_blocks(diffusion_output)
        print("After residual blocks:", residual_output.shape)
        
        # Add residual output to conv_block_output
        combined_output = residual_output + conv_block_output
        print("After adding residual output:", combined_output.shape)
        
        # Deconvolution Blocks
        deconv_output = self.deconv_blocks(combined_output)
        print("After deconvolution blocks:", deconv_output.shape)
        
        # Add global cross-layer connection from input to the final output
        final_output = deconv_output + x
        print("After adding input to final output:", final_output.shape)
        
        # Apply final tanh activation to map to pixel value range
        output = self.final_activation(final_output)
        print("Output shape:", output.shape)
        
        return output

    
# Discriminator

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 32 * 32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x
    
###################### LOSS FUNCTIONS ######################

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=8):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:feature_layer]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        if img1.size(1) == 1:  # Convert single-channel to 3-channel
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        f1 = self.feature_extractor(img1)
        f2 = self.feature_extractor(img2)
        return F.mse_loss(f1, f2)

class TextureLoss(nn.Module):
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

    def forward(self, img1, img2):
        G1 = self.gram_matrix(img1)
        G2 = self.gram_matrix(img2)
        return F.mse_loss(G1, G2)

class ContentLoss(nn.Module):
    def forward(self, img1, img2):
        return torch.sqrt(F.l1_loss(img1, img2) ** 2 + 1e-8)

class WGAN_GP_Loss(nn.Module):
    def __init__(self, discriminator, lambda_gp=10):
        super(WGAN_GP_Loss, self).__init__()
        self.discriminator = discriminator
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, real_images, fake_images):
        batch_size, c, h, w = real_images.size()
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
        epsilon = epsilon.expand_as(real_images)
        interpolation = epsilon * real_images + (1 - epsilon) * fake_images
        interpolation.requires_grad_(True)

        interpolation_logits = self.discriminator(interpolation)
        gradients = torch.autograd.grad(
            outputs=interpolation_logits,
            inputs=interpolation,
            grad_outputs=torch.ones_like(interpolation_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, real_images, fake_images):
        d_real = self.discriminator(real_images).mean()
        d_fake = self.discriminator(fake_images).mean()
        gp = self.gradient_penalty(real_images, fake_images)
        return d_fake - d_real + gp

class MultimodalLoss(nn.Module):
    def __init__(self, discriminator, lambda1, lambda2, lambda3, lambda4):
        super(MultimodalLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.content_loss = ContentLoss()
        self.texture_loss = TextureLoss()
        self.adversarial_loss = WGAN_GP_Loss(discriminator)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

    def forward(self, generated_images, real_images, noisy_images):
        l_percep = self.perceptual_loss(real_images, generated_images)
        l_content = self.content_loss(generated_images, real_images)
        l_texture = self.texture_loss(generated_images, real_images)
        l_adversarial = self.adversarial_loss(real_images, generated_images)
        total_loss = self.lambda1 * l_percep + self.lambda2 * l_content + self.lambda3 * l_texture + self.lambda4 * l_adversarial
        return total_loss
    
###################### TRAINING LOOP ######################

# Denormalize function for TensorBoard visualization
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Training loop
def train_rca_gan(train_loader, val_loader, num_epochs=200, lambda_pixel=1, lambda_perceptual=0.01, lambda_texture=0.001,
                  lr=0.00005, betas=(0.5, 0.999), device=torch.device("cuda" if torch.cuda.is_available() else "mps")):
    # Initialize the models
    in_channels = 1
    out_channels = 1
    generator = GeneratorWithDiffusion(in_channels, out_channels).to(device)
    discriminator = Discriminator(in_channels).to(device)
    multimodal_loss = MultimodalLoss(discriminator, lambda_pixel, lambda_perceptual, lambda_texture, 1).to(device)

    log_dir = 'runs/paper_gan'

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Apply weights initialization
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif hasattr(m, 'bias') and 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

    global_step = 0

    for epoch in range(num_epochs):
        for i, (degraded_images, gt_images) in enumerate(train_loader):
            degraded_images = degraded_images.to(device)
            gt_images = gt_images.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Generate clean images
            gen_clean = generator(degraded_images)
            
            # Prepare real and fake data for discriminator
            real_data = gt_images
            fake_data = gen_clean.detach()
            
            # Get discriminator outputs
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data)
            
            # Calculate discriminator loss
            d_loss = -torch.mean(real_output) + torch.mean(fake_output)
            gp = WGAN_GP_Loss(discriminator).gradient_penalty(real_data, fake_data)
            d_loss += gp

            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(gen_clean)
            
            g_loss = multimodal_loss(gen_clean, gt_images, degraded_images)
            g_loss.backward()
            optimizer_G.step()

            # Print training progress
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            # Log batch losses to TensorBoard
            writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
            writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
            global_step += 1
            
        # Log example images to TensorBoard at the end of each epoch
        with torch.no_grad():
            example_degraded = degraded_images[:4].cpu()  # Take first 4 examples from the last batch
            example_gt = gt_images[:4].cpu()
            example_gen = generator(example_degraded.to(device)).cpu()
            
            # Denormalize images to [0, 1] for better visualization
            example_degraded = denormalize(example_degraded)
            example_gt = denormalize(example_gt)
            example_gen = denormalize(example_gen)

            # Log images
            writer.add_images('Degraded Images', example_degraded, epoch)
            writer.add_images('Generated Images', example_gen, epoch)
            writer.add_images('Ground Truth Images', example_gt, epoch)

        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

    print("Training finished.")
    writer.close()