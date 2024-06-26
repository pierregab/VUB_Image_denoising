import os
import torch
import subprocess
import sys
import optuna
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from paper_gan import MultimodalLoss

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data
from paper_gan import train_rca_gan, denormalize

# Define paths to ground truth and degraded images
image_folder = 'DIV2K_train_HR.nosync'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

def start_tensorboard(log_dir): 
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

def plot_losses(train_losses, val_losses, loss_names, output_path):
    for i, loss_name in enumerate(loss_names):
        plt.plot(train_losses[i], label=f'Train {loss_name}')
        plt.plot(val_losses[i], label=f'Val {loss_name}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def objective(trial, train_loader, val_loader):
    # Tune conv_block_channels
    conv_block_channels = [
        trial.suggest_int('conv_block_channels_0', 8, 64, log=True),
        trial.suggest_int('conv_block_channels_1', 4, 32, log=True),
        trial.suggest_int('conv_block_channels_2', 2, 16, log=True),
        trial.suggest_int('conv_block_channels_3', 1, 8, log=True)
    ]

    # Fixed hyperparameters
    lambda_perceptual = 1e-7
    lambda_content = 1e-4
    lambda_texture = 1e2
    lambda_adversarial = 1
    lr_G = 1e-3
    lr_D = 1e-6

    log_dir = f"runs/paper_gan/optuna_trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)
    
    generator, discriminator = train_rca_gan(
        train_loader, val_loader, num_epochs=20,
        lambda_perceptual=lambda_perceptual, lambda_content=lambda_content,
        lambda_texture=lambda_texture, lambda_adversarial=lambda_adversarial,
        lr_G=lr_G, lr_D=lr_D, log_dir=log_dir,
        early_stopping_patience=5,  # Adding early stopping patience
        trial=trial,  # Pass the trial for pruning
        conv_block_channels=conv_block_channels  # Pass the tuned conv_block_channels
    )

    multimodal_loss = MultimodalLoss(discriminator, lambda_perceptual, lambda_content, lambda_texture, lambda_adversarial).to(device)

    # Calculate validation loss
    val_loss = 0.0
    with torch.no_grad():
        for degraded_images, gt_images in val_loader:
            degraded_images = degraded_images.to(device)
            gt_images = gt_images.to(device)
            gen_clean, _ = generator(degraded_images)
            val_loss += multimodal_loss(gen_clean, gt_images, degraded_images).item()
    
    val_loss /= len(val_loader)
    
    # Save example generated images
    example_degraded = next(iter(val_loader))[0][:4].to(device)
    example_gen, _ = generator(example_degraded)
    example_gen = example_gen.cpu()
    example_gen = denormalize(example_gen)
    torchvision.utils.save_image(example_gen, os.path.join(log_dir, 'generated_images.png'))

    return val_loss

def main():
    log_dir_base = 'runs/paper_gan'
    start_tensorboard(log_dir_base)

    # Use num_workers=8 to utilize multiple workers for data loading
    train_loader, val_loader = load_data(image_folder, batch_size=8, num_workers=8, 
                                         validation_split=0.2, augment=False, dataset_percentage=0.001)

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=50)

    print("Hyperparameter tuning finished.")
    print("Best hyperparameters:", study.best_params)

if __name__ == '__main__':
    main()
