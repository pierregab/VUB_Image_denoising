import torch
import torch.nn as nn
import torch.optim as optim

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2 + x1)
        return x3

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=1000):
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
            x_tilde = (1 - alpha_t) * self.unet(x_t, t) + alpha_t * noisy_image
            x_tilde_prev = (1 - alpha_t_prev) * self.unet(x_t, t) + alpha_t_prev * noisy_image
            x_t = x_t - x_tilde + x_tilde_prev
        return x_t

    def forward(self, clean_image, noisy_image, t):
        noisy_step_image = self.forward_diffusion(clean_image, noisy_image, t)
        denoised_image = self.improved_sampling(noisy_step_image)
        return denoised_image

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

# Define the model and optimizer
unet = UNet()
model = DiffusionModel(unet)
optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))

# Sample training loop
def train_step(model, clean_images, noisy_images, optimizer):
    model.train()
    optimizer.zero_grad()
    
    timesteps = model.timesteps
    total_loss = 0
    for t in range(timesteps):
        denoised_images = model(clean_images, noisy_images, t)
        loss = charbonnier_loss(denoised_images, clean_images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / timesteps

# Dummy data for demonstration
batch_size = 32
image_size = 256
clean_images = torch.randn((batch_size, 3, image_size, image_size))  # Batch of clean images
noisy_images = clean_images + torch.randn((batch_size, 3, image_size, image_size)) * 0.1  # Batch of noisy images

# Training step
loss = train_step(model, clean_images, noisy_images, optimizer)
print(f"Training loss: {loss}")
