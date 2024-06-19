import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import psutil
import subprocess
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.models as models
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
        print(f"{prefix} CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
    elif torch.backends.mps.is_available():
        memory_info = psutil.virtual_memory()
        print(f"{prefix} System Memory Used: {memory_info.used / 1024 ** 3:.2f} GB")
        print(f"{prefix} System Memory Available: {memory_info.available / 1024 ** 3:.2f} GB")

class UNet_S(nn.Module):
    def __init__(self):
        super(UNet_S, self).__init__()
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
        dec4 = self.dec4(torch.cat((self.upconv4(enc4), enc3), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc2), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc1), dim=1))
        dec1 = self.dec1(dec2)
        return dec1

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
            x_t_unet = self.unet(x_t)
            x_tilde = (1 - alpha_t) * x_t_unet + alpha_t * noisy_image
            x_tilde_prev_unet = self.unet(x_t)
            x_tilde_prev = (1 - alpha_t_prev) * x_tilde_prev_unet + alpha_t_prev * noisy_image
            x_t = x_t - x_tilde + x_tilde_prev
        return x_t

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.improved_sampling(noisy_step_image)
        return denoised_image

# Define the PerceptualLoss class
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=8):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:feature_layer]).to(device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        if img1.size(1) == 1:  # Convert single-channel to 3-channel
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        f1 = self.feature_extractor(img1.to(device))
        f2 = self.feature_extractor(img2.to(device))
        return torch.norm(f1 - f2, p=2) ** 2  # L2 norm squared

# Define the TextureLoss class
class TextureLoss(nn.Module):
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

    def forward(self, img1, img2):
        G1 = self.gram_matrix(img1)
        G2 = self.gram_matrix(img2)
        return torch.norm(G1 - G2, p=2) ** 2  # L2 norm squared

# Define the ContentLoss class
class ContentLoss(nn.Module):
    def forward(self, img1, img2):
        epsilon = 1e-8  # small constant to prevent division by zero
        return torch.sqrt(torch.norm(img1 - img2, p=1) ** 2 + epsilon)  # L1 norm with small constant added

# Define the MultimodalLoss class
class MultimodalLoss(nn.Module):
    def __init__(self, lambda1, lambda2, lambda3):
        super(MultimodalLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.content_loss = ContentLoss()
        self.texture_loss = TextureLoss()
        self.lambda1 = lambda1  # Weight for perceptual loss
        self.lambda2 = lambda2  # Weight for content loss
        self.lambda3 = lambda3  # Weight for texture loss

    def forward(self, generated_images, real_images):
        l_percep = self.perceptual_loss(real_images, generated_images)
        l_content = self.content_loss(generated_images, real_images)
        l_texture = self.texture_loss(generated_images, real_images)
        total_loss = self.lambda1 * l_percep + self.lambda2 * l_content + self.lambda3 * l_texture
        return total_loss, l_percep, l_content, l_texture

# Define the model and optimizer
unet = UNet_S().to(device)
model = DiffusionModel(unet).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))

# Instantiate the multimodal loss
multimodal_loss = MultimodalLoss(lambda1=1.0, lambda2=0.01, lambda3=0.001)

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Sample training loop
def train_step(model, clean_images, noisy_images, optimizer, loss_fn, writer, epoch, batch_idx):
    model.train()
    optimizer.zero_grad()
    
    timesteps = model.timesteps
    clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)
    denoised_images = model(clean_images, noisy_images, timesteps)
    total_loss, l_percep, l_content, l_texture = loss_fn(denoised_images, clean_images)
    total_loss.backward()
    optimizer.step()
    
    writer.add_scalar('Loss/total', total_loss.item(), epoch * len(train_loader) + batch_idx)
    writer.add_scalar('Loss/perceptual', l_percep.item(), epoch * len(train_loader) + batch_idx)
    writer.add_scalar('Loss/content', l_content.item(), epoch * len(train_loader) + batch_idx)
    writer.add_scalar('Loss/texture', l_texture.item(), epoch * len(train_loader) + batch_idx)
    
    return total_loss.item()

def train_model(model, train_loader, optimizer, writer, loss_fn, num_epochs=10):
    for epoch in range(num_epochs):
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            loss = train_step(model, clean_images, noisy_images, optimizer, loss_fn, writer, epoch, batch_idx)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss:.4f}")
            
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

    log_dir = os.path.join("runs", "diffusion")
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)
    
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=1, augment=True, dataset_percentage=0.001)
    train_model(model, train_loader, optimizer, writer, multimodal_loss, num_epochs=10)
    writer.close()
