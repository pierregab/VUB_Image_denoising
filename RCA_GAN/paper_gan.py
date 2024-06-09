import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

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
        #out = self.output_norm(out)  # Normalize the output
        return out

    def initialize_weights(self):
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeconvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

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
    
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        # Initial Conv Block
        self.initial_conv = MultiScaleConv(in_channels, 64)

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
            ConvBlock(32, 16, kernel_size=5, stride=1, padding=2),  # reduce to 16 channels, kernel size 5
            ConvBlock(16, 8, kernel_size=7, stride=1, padding=3),   # reduce to 8 channels, kernel size 7
            ConvBlock(8, 4, kernel_size=5, stride=1, padding=2),    # reduce to 4 channels, kernel size 5
            ConvBlock(4, out_channels, kernel_size=1, stride=1, padding=0),  # finally reduce to out_channels, kernel size 1
        )

        # Final tanh activation for output
        self.final_activation = nn.Tanh()

    def forward(self, x):
        intermediate_outputs = {}

        # Initial Conv Block
        initial_conv_output = self.initial_conv(x)
        #print(f'initial_conv_output: {initial_conv_output.shape}')
        intermediate_outputs['initial_conv_output'] = initial_conv_output

        # Feature Domain Denoising
        denoising_output = self.denoising_blocks(initial_conv_output)
        #print(f'denoising_output: {denoising_output.shape}')
        intermediate_outputs['denoising_output'] = denoising_output

        # Subtract initial conv result from denoising output
        denoising_output = denoising_output - initial_conv_output

        # One Convolution Block
        one_conv_output = self.one_conv_block(denoising_output)
        #print(f'one_conv_output: {one_conv_output.shape}')
        intermediate_outputs['one_conv_output'] = one_conv_output

        # Cooperative Attention
        attention_output = self.cooperative_attention(one_conv_output)
        #print(f'attention_output: {attention_output.shape}')
        intermediate_outputs['attention_output'] = attention_output

        # Residual Blocks
        residual_output = self.residual_blocks(attention_output)
        #print(f'residual_output: {residual_output.shape}')
        intermediate_outputs['residual_output'] = residual_output

        # Add residual output to attention_output
        combined_output = residual_output + one_conv_output
        #print(f'combined_output: {combined_output.shape}')
        intermediate_outputs['combined_output'] = combined_output

        # Convolution Blocks leading to a single-channel output
        conv_output = self.conv_blocks(combined_output)
        #print(f'conv_output: {conv_output.shape}')
        intermediate_outputs['conv_output'] = conv_output

        # Add global cross-layer connection from input to the final output
        final_output = conv_output + x
        #print(f'final_output: {final_output.shape}')
        intermediate_outputs['final_output'] = final_output

        # Apply final tanh activation to map to pixel value range
        output = self.final_activation(final_output)
        #print(f'output: {output.shape}')
        intermediate_outputs['output'] = output

        return output, intermediate_outputs

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
        return F.mse_loss(img1, img2)

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

    def discriminator_loss(self, real_images, fake_images):
        d_real = self.discriminator(real_images).mean()
        d_fake = self.discriminator(fake_images).mean()
        gp = self.gradient_penalty(real_images, fake_images)
        return d_fake - d_real + gp

    def generator_loss(self, fake_images):
        return -self.discriminator(fake_images).mean()

class MultimodalLoss(nn.Module):
    def __init__(self, discriminator, lambda1, lambda2, lambda3, lambda4):
        super(MultimodalLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.content_loss = ContentLoss()
        self.texture_loss = TextureLoss()
        self.adversarial_loss = WGAN_GP_Loss(discriminator)
        self.lambda1 = lambda1  # Weight for perceptual loss
        self.lambda2 = lambda2  # Weight for content loss
        self.lambda3 = lambda3  # Weight for texture loss
        self.lambda4 = lambda4  # Weight for adversarial loss

    def forward(self, generated_images, real_images, noisy_images):
        l_percep = self.perceptual_loss(real_images, generated_images)
        l_content = self.content_loss(generated_images, real_images)
        l_texture = self.texture_loss(generated_images, real_images)
        l_adversarial = self.adversarial_loss.generator_loss(generated_images)
        total_loss = self.lambda1 * l_percep + self.lambda2 * l_content + self.lambda3 * l_texture + self.lambda4 * l_adversarial
        return total_loss

# Denormalize function for TensorBoard visualization
def denormalize(tensor):
    return tensor * 0.5 + 0.5

def visualize_activation(name, writer, epoch):
    def hook(model, input, output):
        # Normalize the output for visualization
        output = output - output.min()
        output = output / output.max()
        
        # Ensure the output has 1 or 3 channels
        if output.size(1) > 3:
            output = output[:, :3, :, :]  # Take the first 3 channels
        elif output.size(1) == 1:
            output = output.repeat(1, 3, 1, 1)  # Convert single-channel to 3-channel
        
        grid = torchvision.utils.make_grid(output, normalize=True, scale_each=True)
        writer.add_image(name, grid, epoch)
    return hook

def register_hooks(generator, writer, epoch):
    hooks = {}
    
    # Register hook for the last deconv block only
    hooks["deconv_blocks_last"] = generator.deconv_blocks[-1].register_forward_hook(visualize_activation("Deconv Block Last", writer, epoch))
    hooks["initial_conv"] = generator.initial_conv.register_forward_hook(visualize_activation("Initial Conv Output", writer, epoch))
    
    return hooks

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, ResidualBlock):
        module.initialize_weights()

def train_rca_gan(train_loader, val_loader, num_epochs=1, lambda_perceptual=1.0, lambda_content=0.01, lambda_texture=0.001, lambda_adversarial=1.0,
                  lr=0.0005, betas=(0.5, 0.999), init_type='normal', log_dir='runs/paper_gan', use_tensorboard=True, device=torch.device("cuda" if torch.cuda.is_available() else "mps")):
    # Initialize the models
    in_channels = 1
    out_channels = 1
    generator = Generator(in_channels, out_channels).to(device)
    discriminator = Discriminator(in_channels).to(device)
    multimodal_loss = MultimodalLoss(discriminator, lambda_perceptual, lambda_content, lambda_texture, lambda_adversarial).to(device)

    if use_tensorboard:
        # Initialize TensorBoard writers
        writer = SummaryWriter(log_dir=log_dir)
        writer_debug = SummaryWriter(log_dir=f'{log_dir}/debug')

    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

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
            gen_clean, intermediate_outputs = generator(degraded_images)
            
            # Prepare real and fake data for discriminator
            real_data = gt_images
            fake_data = gen_clean.detach()
            
            # Calculate discriminator loss
            d_loss = multimodal_loss.adversarial_loss.discriminator_loss(real_data, fake_data)
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_data = gen_clean  # Not detached here
            
            g_loss = multimodal_loss(fake_data, gt_images, degraded_images)
            g_loss.backward()
            optimizer_G.step()

            # Print training progress
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            if use_tensorboard:
                # Log batch losses to TensorBoard
                writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
                writer.add_scalar('Loss/Perceptual', multimodal_loss.perceptual_loss(gt_images, gen_clean).item(), global_step)
                writer.add_scalar('Loss/Content', multimodal_loss.content_loss(gt_images, gen_clean).item(), global_step)
                writer.add_scalar('Loss/Texture', multimodal_loss.texture_loss(gt_images, gen_clean).item(), global_step)
                writer.add_scalar('Loss/Adversarial', multimodal_loss.adversarial_loss.generator_loss(gen_clean).item(), global_step)

            global_step += 1

        if use_tensorboard:
            # Log example images and intermediate outputs to TensorBoard at the end of each epoch
            with torch.no_grad():
                example_degraded = degraded_images[:4].cpu()  # Take first 4 examples from the last batch
                example_gt = gt_images[:4].cpu()
                example_gen, intermediate_outputs = generator(example_degraded.to(device))
                example_gen = example_gen.cpu()
                
                # Denormalize images to [0, 1] for better visualization
                example_degraded = denormalize(example_degraded)
                example_gt = denormalize(example_gt)
                example_gen = denormalize(example_gen)

                # Log images
                writer.add_images('Degraded Images', example_degraded, epoch)
                writer.add_images('Generated Images', example_gen, epoch)
                writer.add_images('Ground Truth Images', example_gt, epoch)
                
                # Log intermediate outputs
                for name, output in intermediate_outputs.items():
                    output = output[:4].cpu()  # Take first 4 examples
                    output = denormalize(output)
                    if output.size(1) == 1:  # Convert single-channel to 3-channel
                        output = output.repeat(1, 3, 1, 1)
                    elif output.size(1) > 3:  # If more than 3 channels, take only the first 3 channels
                        output = output[:, :3, :, :]
                    writer_debug.add_images(name, output, epoch)

        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"{log_dir}/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{log_dir}/discriminator_epoch_{epoch+1}.pth")

    print("Training finished.")
    
    if use_tensorboard:
        writer.close()
        writer_debug.close()
        
    return generator, discriminator