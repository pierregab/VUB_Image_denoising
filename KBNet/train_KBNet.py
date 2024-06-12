import torch
import subprocess
import sys
import os
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from KBNet import KBNet  # Ensure this path is correct
from torch.nn import functional as F

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Define paths to ground truth and degraded images
gt_folder = 'DIV2K_train_HR.nosync/resized_ground_truth_images'
degraded_folder = 'DIV2K_train_HR.nosync/degraded_images'

def start_tensorboard(log_dir):
    """Starts TensorBoard."""
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

def denormalize(tensor):
    """Denormalizes a tensor from [-1, 1] to [0, 1]."""
    return tensor * 0.5 + 0.5

# Define the PerceptualLoss class
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

# Define the TextureLoss class
class TextureLoss(nn.Module):
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

    def forward(self, img1, img2):
        if img1.size(1) == 1:  # Convert single-channel to 3-channel
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        G1 = self.gram_matrix(img1)
        G2 = self.gram_matrix(img2)
        return F.mse_loss(G1, G2)

# Define the ContentLoss class
class ContentLoss(nn.Module):
    def forward(self, img1, img2):
        return F.mse_loss(img1, img2)

def train_unet_kbnet(train_loader, val_loader, writer, num_epochs=20, learning_rate=1e-3):
    """Trains the UNet with KBNet module."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = KBNet(img_channel=1).to(device)  # Correct the argument here
    
    # Initialize loss functions
    perceptual_loss = PerceptualLoss().to(device)
    texture_loss = TextureLoss().to(device)
    content_loss = ContentLoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for i, (input_image, target_image) in enumerate(train_loader):
            input_image, target_image = input_image.to(device), target_image.to(device)
            optimizer.zero_grad()
            outputs = model(input_image)

            # Calculate multimodal loss
            p_loss = perceptual_loss(outputs, target_image)
            t_loss = texture_loss(outputs, target_image)
            c_loss = content_loss(outputs, target_image)
            loss = p_loss + t_loss + c_loss

            if torch.isnan(loss):
                print("NaN detected")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Output the loss after each batch
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            if (i + 1) % 10 == 0:
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Perceptual Loss', p_loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Texture Loss', t_loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Content Loss', c_loss.item(), epoch * len(train_loader) + i)

        model.eval()
        val_loss = 0
        val_p_loss = 0
        val_t_loss = 0
        val_c_loss = 0
        with torch.no_grad():
            for input_image, target_image in val_loader:
                input_image, target_image = input_image.to(device), target_image.to(device)
                outputs = model(input_image)

                # Calculate multimodal validation loss
                p_loss = perceptual_loss(outputs, target_image)
                t_loss = texture_loss(outputs, target_image)
                c_loss = content_loss(outputs, target_image)
                loss = p_loss + t_loss + c_loss
                
                val_loss += loss.item()
                val_p_loss += p_loss.item()
                val_t_loss += t_loss.item()
                val_c_loss += c_loss.item()

            val_loss /= len(val_loader)
            val_p_loss /= len(val_loader)
            val_t_loss /= len(val_loader)
            val_c_loss /= len(val_loader)

            print(f'Validation Loss after epoch {epoch + 1}: {val_loss:.4f}')
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Perceptual Loss', val_p_loss, epoch)
            writer.add_scalar('Validation Texture Loss', val_t_loss, epoch)
            writer.add_scalar('Validation Content Loss', val_c_loss, epoch)

        # Log images to TensorBoard
        with torch.no_grad():
            input_image, target_image = next(iter(val_loader))
            input_image, target_image = input_image.to(device), target_image.to(device)
            outputs = model(input_image)

            example_degraded = denormalize(input_image[:4].cpu())
            example_gt = denormalize(target_image[:4].cpu())
            example_gen = denormalize(outputs[:4].cpu())

            writer.add_images('Degraded Images', example_degraded, epoch)
            writer.add_images('Generated Images', example_gen, epoch)
            writer.add_images('Ground Truth Images', example_gt, epoch)

def main():
    log_dir = 'runs/unet_kbnet'
    writer = SummaryWriter(log_dir)
    start_tensorboard(log_dir)

    # Use num_workers=0 to avoid multiprocessing issues for debugging
    train_loader, val_loader = load_data(gt_folder, degraded_folder, batch_size=1, num_workers=0, 
                                         validation_split=0.2, augment=False, dataset_percentage=0.05)

    # Train the model with the current hyperparameters
    train_unet_kbnet(train_loader, val_loader, writer, num_epochs=20, learning_rate=1e-5)  # Reduced learning rate

    writer.close()

if __name__ == '__main__':
    main()
