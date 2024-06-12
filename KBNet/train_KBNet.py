import torch
import subprocess
import sys
import os
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from KBNet import KBNet  # Ensure this path is correct

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

def train_unet_kbnet(train_loader, val_loader, writer, num_epochs=20, learning_rate=1e-3):
    """Trains the UNet with KBNet module."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = KBNet(img_channel=1).to(device)  # Correct the argument here
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for i, (input_image, target_image) in enumerate(train_loader):
            input_image, target_image = input_image.to(device), target_image.to(device)
            optimizer.zero_grad()
            outputs = model(input_image)
            loss = criterion(outputs, target_image)
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_image, target_image in val_loader:
                input_image, target_image = input_image.to(device), target_image.to(device)
                outputs = model(input_image)
                val_loss += criterion(outputs, target_image).item()
            val_loss /= len(val_loader)
            print(f'Validation Loss after epoch {epoch + 1}: {val_loss:.4f}')
            writer.add_scalar('Validation Loss', val_loss, epoch)

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